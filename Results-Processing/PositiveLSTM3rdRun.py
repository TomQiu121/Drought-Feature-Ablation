import pandas as pd

rows, cols = 20, 2
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
    "Using past drought observation scores",
]

for i in range(rows):
    col = []
    col.append(titles[i])
    accuracy_values = pd.read_csv(
        "Results/PositiveLSTM3rdRun.csv", skiprows=(i) * 6, nrows=6, header=None
    )
    accuracy_values.columns = [0, 1]
    lstm_mae = 0
    lstm_ = 0
    for j in range(6):
        lstm_mae += accuracy_values.loc[j].iat[0]
        lstm_ += accuracy_values.loc[j].iat[1]
    col.append(lstm_mae / 6)
    col.append(lstm_ / 6)
    arr.append(col)

df = pd.DataFrame(arr, columns=["Attribute", "LSTM MAE", "LSTM F1"])
print(df.sort_values(by=["LSTM MAE"]).to_latex())

arr2 = []
for i in range(20):
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
    lst.append((df.loc[0].iat[1] - df.loc[i].iat[1]))
    lst.append((df.loc[0].iat[2] - df.loc[i].iat[2]))
    arr2.append(lst)
    # df_temp = pd.DataFrame(
    #    lst,
    #    columns=["Attribute", "LSTM MAE", "lstm MAE", "LSTM  ", "lstm  "],
    # )
    # df_diff = pd.concat([df_diff, df_temp])

df_diff = pd.DataFrame(arr2, columns=["Attribute", "LSTM MAE", "LSTM F1"])

print("\n" + "Difference dataframe:")
print(df_diff)

lstm_df = df_diff.sort_values(by=["LSTM MAE"])
print("\n" + "Sorted by LSTM MAE values")
print(lstm_df)
