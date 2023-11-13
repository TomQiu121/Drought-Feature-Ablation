import numpy as np
import pandas as pd
import json
import os
from numpy import load
from tqdm.auto import tqdm
from tqdm.notebook import tqdm
from datetime import datetime
from scipy.interpolate import interp1d
from sklearn.preprocessing import RobustScaler
import torch
import random

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

removed = 18


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# Set the random seed
setup_seed(20)

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

X_time_train = np.delete(X_time_train, [4, 18], axis=2)
X_time_valid = np.delete(X_time_valid, [4, 18], axis=2)
X_time_test = np.delete(X_time_test, [4, 18], axis=2)

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

class2id = {
    "None": 0,
    "D0": 1,
    "D1": 2,
    "D2": 3,
    "D3": 4,
    "D4": 5,
}

id2class = {v: k for k, v in class2id.items()}


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

import torch
from torch import nn
from sklearn.metrics import f1_score, mean_absolute_error


class DroughtNetLSTM(nn.Module):
    def __init__(
        self,
        output_size,
        num_input_features,
        hidden_dim,
        n_layers,
        ffnn_layers,
        drop_prob,
        static_dim=0,
    ):
        super(DroughtNetLSTM, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            num_input_features,
            hidden_dim,
            n_layers,
            dropout=drop_prob,
            batch_first=True,
        )
        self.dropout = nn.Dropout(drop_prob)
        self.fflayers = []
        for i in range(ffnn_layers - 1):
            if i == 0:
                self.fflayers.append(nn.Linear(hidden_dim + static_dim, hidden_dim))
            else:
                self.fflayers.append(nn.Linear(hidden_dim, hidden_dim))
        self.fflayers = nn.ModuleList(self.fflayers)
        self.final = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden, static=None):
        batch_size = x.size(0)
        x = x.to(dtype=torch.float32)
        if static is not None:
            static = static.to(dtype=torch.float32)
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out[:, -1, :]

        out = self.dropout(lstm_out)
        for i in range(len(self.fflayers)):
            if i == 0 and static is not None:
                out = self.fflayers[i](torch.cat((out, static), 1))
            else:
                out = self.fflayers[i](out)
        out = self.final(out)

        out = out.view(batch_size, -1)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
        )
        return hidden


def predict(x, static=None):
    if static is None:
        out, _ = model(torch.tensor(x), val_h)
    else:
        out, _ = model(torch.tensor(x), val_h, static)
    return out


best_mae_sum = float("inf")
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("using GPU")
else:
    device = torch.device("cpu")
    print("using CPU")
static_dim = 0
if use_static:
    static_dim = X_static_train.shape[-1]
model = DroughtNetLSTM(
    output_weeks,
    X_time_train.shape[-1],
    hidden_dim,
    n_layers,
    ffnn_layers,
    dropout,
    static_dim,
)
model.to(device)
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
    h = model.init_hidden(batch_size)

    for k, (inputs, static, labels) in tqdm(
        enumerate(train_loader),
        desc=f"epoch {i+1}/{epochs}",
        total=len(train_loader),
    ):
        model.train()
        counter += 1
        if len(inputs) < batch_size:
            h = model.init_hidden(len(inputs))
        h = tuple([e.data for e in h])
        inputs, labels, static = (
            inputs.to(device),
            labels.to(device),
            static.to(device),
        )
        model.zero_grad()
        if use_static:
            output, h = model(inputs, h, static)
        else:
            output, h = model(inputs, h)
        loss = loss_function(output, labels.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if one_cycle:
            scheduler.step()

    # using validation set to find the best model in training
    model.eval()
    with torch.no_grad():
        val_h = model.init_hidden(batch_size)
        val_losses = []
        model.eval()
        labels = []
        preds = []
        raw_labels = []
        raw_preds = []
        for inp, stat, lab in valid_loader:
            if len(inp) < batch_size:
                val_h = model.init_hidden(len(inp))
            val_h = tuple([each.data for each in val_h])
            inp, lab, stat = inp.to(device), lab.to(device), stat.to(device)
            if use_static:
                out, val_h = model(inp, val_h, stat)
            else:
                out, val_h = model(inp, val_h)
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
            for x, static, y in tqdm(
                test_loader,
                desc="test predictions...",
            ):
                x = x.cuda()
                static = static.cuda()
                val_h = tuple([each.data.cuda() for each in model.init_hidden(len(x))])
                with torch.no_grad():
                    x = x.to(device)
                    static = static.to(device)
                    if use_static:
                        pred = predict(x, static).clone().detach()
                    else:
                        pred = predict(x).clone().detach()
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

for i in range(len(dict_map["fips"])):
    if dict_map["fips"][i] not in graph_data["fips"]:
        graph_data["fips"].append(dict_map["fips"][i])
        graph_data["date"].append(dict_map["date"][i])
        graph_data["y_pred"].append(dict_map["y_pred"][i])
        graph_data["y_true"].append(dict_map["y_true"][i])

print(len(graph_data["fips"]))
graph_data = pd.DataFrame(graph_data)
print(graph_data)
graph_data.to_csv("heatmap.csv")
