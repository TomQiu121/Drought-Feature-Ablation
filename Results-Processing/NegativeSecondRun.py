import pandas as pd

rows, cols = 21, 5
arr = []
titles = [
    "None removed",
    "Precipitation",
    "Surface pressure",
    "Humidity at 2m",
    "Temperature at 2m",
    "Dew/Frost Point at 2m",
    "Wet Bulb Temperature at 2m",
    "Maximum Temperature at 2m",
    "Minimum temperature at 2m",
    "Temperature Range at 2m",
    "Earth skin temperature",
    "Wind speed at 10m",
    "Maximum wind speed at 10m",
    "Minimum wind speed at 10m",
    "Wind speed range at 10m",
    "Wind speed at 50m",
    "Maximum wind speed at 50m",
    "Minimum wind speed at 50m",
    "Wind speed range at 50m",
    "Sine Date encoding",
    "Cosine Date encoding",
]

for i in range(rows):
    col = []
    col.append(titles[i])
    accuracy_values = pd.read_csv(
        "Results/Negative2ndRun.csv", skiprows=(i) * 12, nrows=12, header=None
    )
    accuracy_values.columns = [0, 1]
    lstm_mae = 0
    cnn_mae = 0
    lstm_ = 0
    cnn_ = 0
    for j in range(6):
        lstm_mae += accuracy_values.loc[j].iat[0]
        lstm_ += accuracy_values.loc[j].iat[1]
        cnn_mae += accuracy_values.loc[j + 6].iat[0]
        cnn_ += accuracy_values.loc[j + 6].iat[1]
    col.append(lstm_mae / 6)
    col.append(cnn_mae / 6)
    col.append(lstm_ / 6)
    col.append(cnn_ / 6)
    arr.append(col)

df = pd.DataFrame(
    arr, columns=["Attribute", "LSTM MAE", "CNN MAE", "LSTM F1", "CNN F1"]
)
print(df.sort_values(by=["CNN MAE"]).to_latex())

arr2 = []
for i in range(21):
    # lst = [
    #  titles[i],
    #  df.loc[0].iat[1] - df.loc[i].iat[1],
    #  df.loc[0].iat[2] - df.loc[i].iat[2],
    #  df.loc[i].iat[3] - df.loc[0].iat[3],
    #  df.loc[i].iat[4] - df.loc[0].iat[4],
    # ]
    lst = []
    lst.append(titles[i])
    # lst.append((df.loc[0].iat[1] - df.loc[i].iat[1]) / df.loc[0].iat[1] * 100)
    # lst.append((df.loc[0].iat[2] - df.loc[i].iat[2]) / df.loc[0].iat[2] * 100)
    # lst.append((df.loc[i].iat[3] - df.loc[0].iat[3]) / df.loc[0].iat[3] * 100)
    # lst.append((df.loc[i].iat[4] - df.loc[0].iat[4]) / df.loc[0].iat[4] * 100)
    lst.append((df.loc[0].iat[1] - df.loc[i].iat[1]))
    lst.append((df.loc[0].iat[2] - df.loc[i].iat[2]))
    lst.append((df.loc[i].iat[3] - df.loc[0].iat[3]))
    lst.append((df.loc[i].iat[4] - df.loc[0].iat[4]))
    # currently using Percent Difference, above is Percent Error
    # lst.append(
    #     (df.loc[0].iat[1] - df.loc[i].iat[1])
    #     / (df.loc[0].iat[1] + df.loc[i].iat[1])
    #     * 200
    # )
    # lst.append(
    #     (df.loc[0].iat[2] - df.loc[i].iat[2])
    #     / (df.loc[0].iat[2] + df.loc[i].iat[2])
    #     * 200
    # )
    # lst.append(
    #     (df.loc[i].iat[3] - df.loc[0].iat[3])
    #     / (df.loc[i].iat[3] + df.loc[0].iat[3])
    #     * 200
    # )
    # lst.append(
    #     (df.loc[i].iat[4] - df.loc[0].iat[4])
    #     / (df.loc[i].iat[4] + df.loc[0].iat[4])
    #     * 200
    # )
    arr2.append(lst)
    # df_temp = pd.DataFrame(
    #    lst,
    #    columns=["Attribute", "LSTM MAE", "CNN MAE", "LSTM  ", "CNN  "],
    # )
    # df_diff = pd.concat([df_diff, df_temp])

df_diff = pd.DataFrame(
    arr2, columns=["Attribute", "LSTM MAE", "CNN MAE", "LSTM F1", "CNN F1"]
)

print("\n" + "Difference dataframe:")
print(df_diff)

lstm_df = df_diff.sort_values(by=["LSTM MAE"])
print("\n" + "Sorted by LSTM MAE values")
print(lstm_df)

cnn_df = df_diff.sort_values(by=["CNN MAE"])
print("\n" + "Sorted by CNN MAE values")
print(cnn_df)

same_list = []
not_same_list = []
for i in range(1, 21):
    if df_diff.loc[i].iat[1] * df_diff.loc[i].iat[2] > 0:
        if df_diff.loc[i].iat[1] > 0:
            same_list.append(
                "Removing " + df_diff.loc[i].iat[0] + " improved the model accuracy"
            )  # how to get this to work, it still rearranges the index
        else:
            same_list.append(
                "Removing " + df_diff.loc[i].iat[0] + " worsened the model accuracy"
            )
    else:
        not_same_list.append(df_diff.loc[i].iat[0])

print("\n Attributes that cause both models to behave similarly:")
print(same_list)
print("\n Attributes that cause models to behave differently:")
print(not_same_list)
