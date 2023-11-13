import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from numpy import load


# import relevant data
us_map = gpd.read_file("./Heatmap/CountiesBEA.shp")
results = pd.read_csv("./Results/NegativeLSTM3rdRun.csv")

us_map.plot()
print(results)

for i in range(len(results)):
    results.loc[i, "y_pred"] = (results.loc[i, "y_pred"] + 0.5) // 1
    results.loc[i, "y_true"] = (results.loc[i, "y_true"] + 0.5) // 1

for i in range(len(results)):
    results.loc[i, "y_pred"] -= results.loc[i, "y_true"]

x = results["y_pred"]
x = x.sort_values()
print(x)

# # change column name and merge data
results = results.rename(columns={"fips": "FIPS_BEA"})
map = us_map.merge(results, on="FIPS_BEA")


fig, ax = plt.subplots(1, figsize=(10, 10))
bar_info = plt.cm.ScalarMappable(
    cmap="seismic", norm=plt.Normalize(vmin=-2.5, vmax=2.5)
)
bar_info._A = []
cbar = fig.colorbar(bar_info)

map.plot(column="y_pred", cmap="seismic", linewidth=0.5, ax=ax, edgecolor=".5")
