import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from tqdm.auto import tqdm
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# Setting the random seed
setup_seed(20)
removed = 20

class2id = {
    "None": 0,
    "D0": 1,
    "D1": 2,
    "D2": 3,
    "D3": 4,
    "D4": 5,
}

id2class = {v: k for k, v in class2id.items()}

from numpy import load

X_static_train = load("./Data/X_static_train.csv.npy")
X_time_train = load("./Data/X_time_train.csv.npy")
y_target_train = load("./Data/y_target_train.csv.npy")

X_static_valid = load("./Data/X_static_valid.csv.npy")
X_time_valid = load("./Data/X_time_valid.csv.npy")
y_target_valid = load("./Data/y_target_valid.csv.npy")

X_static_test = load("./Data/X_static_test.csv.npy")
X_time_test = load("./Data/X_time_test.csv.npy")
y_target_test = load("./Data/y_target_test.csv.npy")

v_fips = load("./Data/valid_fips.csv.npy", allow_pickle=True)
t_fips = load("./Data/test_fips.csv.npy", allow_pickle=True)

valid_fips = v_fips.tolist()
test_fips = t_fips.tolist()

X_time_train = np.swapaxes(X_time_train, 1, 2)
X_time_valid = np.swapaxes(X_time_valid, 1, 2)
X_time_test = np.swapaxes(X_time_test, 1, 2)

print(
    "Train dataset shape: ",
    X_time_train.shape,
    X_static_train.shape,
    y_target_train.shape,
)
print(
    "Validation dataset shape: ",
    X_time_valid.shape,
    X_static_valid.shape,
    y_target_valid.shape,
)
print(
    "Test dataset shape: ", X_time_test.shape, X_static_test.shape, y_target_test.shape
)
print("Fips shape: ", len(valid_fips), len(test_fips))


X_time_train = np.delete(X_time_train, [20], axis=1)
X_time_valid = np.delete(X_time_valid, [20], axis=1)
X_time_test = np.delete(X_time_test, [20], axis=1)

print(
    "Train dataset shape: ",
    X_time_train.shape,
    X_static_train.shape,
    y_target_train.shape,
)
print(
    "Validation dataset shape: ",
    X_time_valid.shape,
    X_static_valid.shape,
    y_target_valid.shape,
)
print(
    "Test dataset shape: ", X_time_test.shape, X_static_test.shape, y_target_test.shape
)
print("Fips shape: ", len(valid_fips), len(test_fips))

batch_size = 16
output_weeks = 6
use_static = True
hidden_dim = 512
n_layers = 2
ffnn_layers = 2
dropout = 0.1
one_cycle = True
lr = 7e-5
epochs = 10
clip = 5
kernel_size = 2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.utils.data import TensorDataset, DataLoader

train_data = TensorDataset(
    torch.tensor(X_time_train),
    torch.tensor(X_static_train),
    torch.tensor(y_target_train[:, :output_weeks]),
)

# DataLoader will reseed workers following Randomness in multi-process data loading algorithm.
# Use worker_init_fn() and generator to preserve reproducibility:

train_loader = DataLoader(
    train_data, shuffle=False, batch_size=batch_size, drop_last=False
)

valid_data = TensorDataset(
    torch.tensor(X_time_valid),
    torch.tensor(X_static_valid),
    torch.tensor(y_target_valid[:, :output_weeks]),
)
valid_loader = DataLoader(
    valid_data, shuffle=False, batch_size=batch_size, drop_last=False
)


test_data = TensorDataset(
    torch.tensor(X_time_test),
    torch.tensor(X_static_test),
    torch.tensor(y_target_test[:, :output_weeks]),
)

test_loader = DataLoader(
    test_data, shuffle=False, batch_size=batch_size, drop_last=False
)


size = 50 * (((180 - kernel_size + 1) // 2 - kernel_size + 1) // 2)


class ConvNet(nn.Module):
    def __init__(self):
        super()
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(X_time_train.shape[1], 40, kernel_size)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(40, 50, kernel_size)
        self.fc1 = nn.Linear(size + 30, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 6)

    def forward(self, x, static=None):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, size)
        if use_static:
            x = torch.cat([x, static], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def predict(x, static=None):
    if static is None:
        out = model(torch.tensor(x))
    else:
        out = model(torch.tensor(x), static)
    return out


model = ConvNet().to(device)
best_mae_sum = float("inf")
loss_function = nn.MSELoss()
if one_cycle:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs
    )
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


counter = 0
valid_loss_min = np.Inf
for i in range(epochs):
    current_mae_sum = 0

    for k, (inputs, static, labels) in tqdm(
        enumerate(train_loader),
        desc=f"epoch {i+1}/{epochs}",
        total=len(train_loader),
    ):
        model.train()
        counter += 1
        inputs, labels, static = (
            inputs.to(device=device, dtype=torch.float),
            labels.to(device=device, dtype=torch.float),
            static.to(device=device, dtype=torch.float),
        )
        if use_static:
            output = model(inputs, static)
        else:
            output = model(inputs)
        loss = loss_function(output, labels.float())
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if one_cycle:
            scheduler.step()

    print("Finished Training")

    model.eval()
    with torch.no_grad():
        val_losses = []
        model.eval()
        labels = []
        preds = []
        raw_labels = []
        raw_preds = []
        for inp, stat, lab in valid_loader:
            inp, lab, stat = (
                inp.to(device=device, dtype=torch.float),
                lab.to(device=device, dtype=torch.float),
                stat.to(device=device, dtype=torch.float),
            )
            if use_static:
                out = model(inp, stat)
            else:
                out = model(inp)
            val_loss = loss_function(out, lab.float())
            val_losses.append(val_loss.item())
            for labs in lab:
                labels.append([int(l.round()) for l in labs])
                raw_labels.append([float(l) for l in labs])
            for pred in out:
                preds.append([int(p.round()) for p in pred])
                raw_preds.append([float(p) for p in pred])
                # log data
        labels = np.array(labels)
        preds = np.clip(np.array(preds), 0, 5)
        raw_preds = np.array(raw_preds)
        raw_labels = np.array(raw_labels)
        for k in range(output_weeks):
            log_dict = {
                "loss": float(loss),
                "epoch": counter / len(train_loader),
                "step": counter,
                "lr": optimizer.param_groups[0]["lr"],
                "week": k + 1,
            }
            # w = f'week_{k+1}_'
            w = ""
            # log_dict[f"{w}validation_loss"] = np.mean(val_losses)
            log_dict[f"{w}mae"] = mean_absolute_error(
                raw_labels[:, k], raw_preds[:, k]
            ).round(5)
            current_mae_sum += log_dict[f"{w}mae"]
            # log_dict[f"{w}mae"] = mean_absolute_error(
            # raw_labels[:, k], raw_preds[:, k]
            # )
            print(log_dict)
        if current_mae_sum < best_mae_sum:
            best_mae_sum = current_mae_sum
            torch.save(model.state_dict(), f"./state_dict_{i}.pt")
            print(
                "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
                    current_mae_sum, best_mae_sum
                )
            )
            print("Making predictions on the test set now")
            dict_map = {
                "y_pred": [],
                "y_pred_rounded": [],
                "fips": [],
                "date": [],
                "y_true": [],
                "week": [],
            }
            i = 0
            count = 0
            for x, static, y in tqdm(
                test_loader,
                desc="test predictions...",
            ):
                x = x.cuda()
                static = static.cuda()
                with torch.no_grad():
                    x = x.to(device=device, dtype=torch.float)
                    static = static.to(device=device, dtype=torch.float)
                    if use_static:
                        pred = predict(x, static).clone().detach()
                    else:
                        pred = predict(x).clone().detach()
                    # pred is the prediction of the model (it's 16 * 6, since batch size is 16, and we are predicting for 6 weeks)
                for w in range(output_weeks):
                    dict_map["y_pred"] += [float(p[w]) for p in pred]
                    dict_map["y_pred_rounded"] += [int(p.round()[w]) for p in pred]
                    dict_map["fips"] += [f[1][0] for f in test_fips[i : i + len(x)]]
                    dict_map["date"] += [f[1][1] for f in test_fips[i : i + len(x)]]
                    dict_map["y_true"] += [float(item[w]) for item in y]
                    dict_map["week"] += [w] * len(x)
                i += len(x)
            df = pd.DataFrame(dict_map)

            print("Test results: ")
            for w in range(6):
                wdf = df[df["week"] == w]
                mae = mean_absolute_error(wdf["y_true"], wdf["y_pred"]).round(5)
                f1 = f1_score(
                    wdf["y_true"].round(), wdf["y_pred"].round(), average="macro"
                ).round(5)
                print(f"Week {w+1}", f"MAE {mae}", f"F1 {f1}")
                # with open(f"/content/gdrive/MyDrive/Drought/Positive_Round_2/CNN/file{removed}.txt",'a') as f:
                #  f.write(f"Week {w+1}"+f" MAE {mae}"+f" F1 {f1}"+"\n")
            # valid_loss_min = np.mean(val_losses)

for i in range(len(dict_map["fips"])):
    if dict_map["fips"][i] not in graph_data["fips"]:  # combined 2019 and 2020 data
        graph_data["fips"].append(dict_map["fips"][i])
        graph_data["date"].append(dict_map["date"][i])
        graph_data["y_pred"].append(dict_map["y_pred"][i])
        graph_data["y_true"].append(dict_map["y_true"][i])

print(len(graph_data["fips"]))
