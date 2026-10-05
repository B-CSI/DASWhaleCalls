# %% Imports
# Standard library
import datetime as dt
import os
import shutil
import sys
import time
from pathlib import Path

# Third-party libraries
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN
from dataclasses import asdict
from tqdm import tqdm

from config.parameters import *
from scripts.kvp_clustering import *
# %% Load KVP picks
csv_path = r".\results\2024_01_13_07h57m34s_HDAS_SAFE_DASWhaleCalls_example_KVPpicks.csv"
CSVfile_raw_data = pd.read_csv(csv_path)
print("\n=== KVP PICKS ===")
print(f"Loaded: {csv_path}")
print(f"Number of picks: {len(CSVfile_raw_data)}")
print(f"Columns: {CSVfile_raw_data.columns.tolist()}")
print(CSVfile_raw_data.head())

# Plot X vs Y
plt.figure(figsize=(12, 7))
plt.scatter(
    CSVfile_raw_data["time_rel"],
    CSVfile_raw_data["Channel"],
    s=15,
    alpha=0.7
)
plt.xlabel("Time [s]")
plt.ylabel("Channel")
plt.title("KVP Picks: Relative Arrival Time vs Channel")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% CSVfile filtering: 
CSVfile_raw_data_filt = CSVfile_raw_data.copy()
Nbands_max = 4
LBand, Hband = 15, 30
CSVfile_raw_data_filt = CSVfile_raw_data_filt[CSVfile_raw_data_filt["Number_bands"]<=Nbands_max]
CSVfile_raw_data_filt = CSVfile_raw_data_filt[CSVfile_raw_data_filt["Lowest_band"]>=LBand]
CSVfile_raw_data_filt = CSVfile_raw_data_filt[CSVfile_raw_data_filt["Highest_band"]>=LBand]
CSVfile_raw_data_filt = CSVfile_raw_data_filt[CSVfile_raw_data_filt["Lowest_band"]<=Hband]
CSVfile_raw_data_filt = CSVfile_raw_data_filt[CSVfile_raw_data_filt["Highest_band"]<=Hband]
# print(f"CSVfile_raw_data_filt: {len(CSVfile_raw_data_filt)} samples in {str(dt.timedelta(seconds=float(np.max(CSVfile_raw_data_filt['time_rel']) - np.min(CSVfile_raw_data_filt['time_rel'])))).split('.')[0]} ({len(CSVfile_raw_data_filt)/(np.max(CSVfile_raw_data_filt['time_rel']) - np.min(CSVfile_raw_data_filt['time_rel'])):.1f}/s)")

CSVfile_raw_data_filt = CSVfile_raw_data.copy()
# %% Prepare coordinates for clustering:
x_raw, y_raw = CSVfile_raw_data_filt["time_rel"],(CSVfile_raw_data_filt["Channel"]+1000)*dist_ch
x, y = x_raw.copy(), y_raw.copy()


# %% DBSCAN application:
print("\n=== DBSCAN PARAMETERS ===")
print(f"eps = {eps_m}m")
print(f"min_pts = {min_pts}")

x_prima = c_sound * x #[m]
y_prima = y.copy() #[m]
dbscan_df, n_noise = run_dbscan(x_prima, y_prima, dist_eps=eps_m, min_pts=min_pts)

dbscan_df['x0'] /= c_sound
dbscan_df['Xsize'] /= c_sound

print("\n=== DBSCAN RESULT ===")
print(f"Number of DBSCAN clusters: {len(dbscan_df)}")
print(f"Number of noise points: {n_noise}")
# print(dbscan_df)
# %% Representing dbscan_df:
unique_clusters = np.sort(dbscan_df["cluster_id"].unique())
FigName = "dbscan_df"
title_str = f"DBSCAN: {dbscan_df['cluster_id'].nunique()} clusters | {n_noise} noise | eps = {eps_m} m\n"

fig, ax = plt.subplots(figsize=(8, 8))
# Plot all KVP picks used for clustering
ax.scatter(x,y,s=8,marker=".",color="grey",label=f"KVP picks ({len(x)})")
# Plot original DBSCAN bounding boxes
for cluster in dbscan_df.itertuples():
    ax.add_patch(
        patches.Rectangle(
            (cluster.x0, cluster.y0),
            cluster.Xsize,
            cluster.Ysize,
            linewidth=2.5,
            linestyle="--",
            edgecolor="red",
            facecolor="none",
            alpha=0.9,
        )
    )
    # Cluster ID label
    ax.text(
        cluster.x0 + cluster.Xsize,
        cluster.y0,
        str(cluster.cluster_id),
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="red",
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.7,
            pad=1.5,
        ),
    )
# Dummy artist for legend
ax.add_patch(
    patches.Rectangle(
        (0, 0),
        0,
        0,
        linewidth=2.5,
        linestyle="--",
        edgecolor="red",
        facecolor="none",
        label=f"DBSCAN ({eps_m} m)",
    )
)
# Figure formatting
ax.set_title(title_str, fontsize=TITLE_SIZE, fontweight="bold")
ax.set_xlabel("Time [s]", fontsize=LABEL_SIZE)
ax.set_ylabel("Distance [m]", fontsize=LABEL_SIZE)
ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
ax.tick_params(axis="both", which="minor", labelsize=TICK_SIZE)
ax.grid(True, linewidth=GRID_WIDTH, alpha=GRID_ALPHA)
legend = ax.legend(loc="upper center",ncols=2,fontsize=LEGEND_SIZE,frameon=True)
legend.get_frame().set_edgecolor("black")
legend.get_frame().set_alpha(0.5)
fig.tight_layout()
fig.savefig(
    os.path.join(output_results, f"{FigName}.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# %% Cluster merging:
# Apply cluster merging
dbscan_merged_df = merge_dbscan_clusters(
    dbscan_df,
    x_dist=Xdist,     
    y_dist=Ydist,   
)

print("\n=== CLUSTER MERGING RESULT ===")
print(f"DBSCAN clusters: {len(dbscan_df)}")
print(f"Merged clusters: {len(dbscan_merged_df)}")

print("\n=== SOURCE CLUSTERS ===")
for row in dbscan_merged_df.itertuples():
    print(
        f"Merged {row.cluster_id}: "
        f"{row.source_clusters}"
    )

# %% Representing DBSCAN+merged results:
# Plot DBSCAN and merged cluster bounding boxes
FigName = "dbscan_merged_df"
title_str = (
    f"DBSCAN: {dbscan_df['cluster_id'].nunique()} clusters | {n_noise} noise | eps = {eps_m} m\n"
    f"Merged: {len(dbscan_merged_df)} clusters | Xdist = ±{Xdist} s | Ydist = ±{Ydist} m"
)

fig, ax = plt.subplots(figsize=(8, 8))
# Plot all KVP picks used for clustering
ax.scatter(x,y,s=8,marker=".",color="grey",label=f"KVP picks ({len(x)})")
# Plot original DBSCAN bounding boxes
for cluster in dbscan_df.itertuples():
    ax.add_patch(
        patches.Rectangle(
            (cluster.x0, cluster.y0),
            cluster.Xsize,
            cluster.Ysize,
            linewidth=2.5,
            linestyle="--",
            edgecolor="red",
            facecolor="none",
            alpha=0.9,
        )
    )
# Dummy artist for legend
ax.add_patch(
    patches.Rectangle(
        (0, 0),
        0,
        0,
        linewidth=2.5,
        linestyle="--",
        edgecolor="red",
        facecolor="none",
        label=f"DBSCAN ({eps_m} m)",
    )
)
# Plot merged clusters
# Plot merged clusters
for cluster in dbscan_merged_df.itertuples():
    ax.add_patch(
        patches.Rectangle(
            (cluster.x0, cluster.y0),
            cluster.Xsize,
            cluster.Ysize,
            linewidth=2.5,
            linestyle="-",
            edgecolor="black",
            facecolor="none",
            alpha=0.8,
        )
    )
    ax.text(
        cluster.x0 + cluster.Xsize / 2,
        cluster.y0 + cluster.Ysize / 2,
        str(cluster.cluster_id),
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="black",
    )
# Dummy artists for legend
ax.add_patch(
    patches.Rectangle(
        (0, 0),
        0,
        0,
        linewidth=2.5,
        edgecolor="black",
        facecolor="none",
        label="Merged clusters",
    )
)
ax.scatter([],[],s=14,marker=".",color="tab:blue",label=f"Clustered KVP picks ({len(dbscan_merged_df)})")
# Figure formatting
ax.set_title(title_str, fontsize=TITLE_SIZE, fontweight="bold")
ax.set_xlabel("Time [s]", fontsize=LABEL_SIZE)
# ax.set_xlim(305,330)
ax.set_ylabel("Distance [m]", fontsize=LABEL_SIZE)
ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
ax.tick_params(axis="both", which="minor", labelsize=TICK_SIZE)
ax.grid(True, linewidth=GRID_WIDTH, alpha=GRID_ALPHA)
legend = ax.legend(loc="upper center",ncols=2,fontsize=LEGEND_SIZE,frameon=True)
legend.get_frame().set_edgecolor("black")
legend.get_frame().set_alpha(0.5)
fig.tight_layout()
fig.savefig(
    os.path.join(output_results, f"{FigName}.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)
# %% 
FALTA APLICAR A CSVfile_raw_data_filt EL CLUSTER_ID AL QUE PERTENCE DE dbscan_merged_df PARA PODER SEGUIR
SERÁ A ESOS PUNTOS A LOS QUE LE APLIQUEMOS EL HIPERBOLIC 
# %% Hyperbolic fit application:
from scripts.kvp_clustering import fit_cluster

fits=[]
for cluster_id, cluster_data_df in dbscan_merged_df.groupby("cluster_id"):
    fit =  fit_cluster(
            cluster_data=cluster_data_df,
            min_points=min_fit_points,
            grid_size=fit_grid_size,
            early_time_weight=early_time_weight,
            early_time_weight_power=early_time_weight_power,
            refit=True,
            refit_time_threshold=refit_time_threshold,
            refit_no_early_weight=True,
        )
    fits.append(fit)
HyperFits_df = pd.DataFrame([asdict(f) for f in fits if f is not None])

# %%
from scripts.kvp_clustering import find_crossing_points

CSVfile_data = dbscan_merged_df.merge(
    HyperFits_df,
    on="cluster_id",
    how="left"
)
CSVfile_data = CSVfile_data.dropna(subset=["y0"])
unique_clusters_raw, counts_raw = np.unique(CSVfile_data["cluster_id"].values,return_counts=True)

filtered_rows = []
for cid in np.unique(CSVfile_data["cluster_id"].values):
    rows = CSVfile_data[CSVfile_data["cluster_id"] == cid]
    cluster_x = rows["x"].values 
    cluster_y = rows["Channel"].values
    # model prediction
    t_pred = np.sqrt(rows["t0"].iloc[0] ** 2 + ((cluster_y - rows["y0"].iloc[0]) ** 2) / (rows["vapp"].iloc[0] ** 2))
    # sort by channel (o y)
    idx = np.argsort(cluster_y)
    t_pred = t_pred[idx]
    cluster_x = cluster_x[idx]
    cluster_y = cluster_y[idx]
    # crossing points
    x_cross, y_cross = find_crossing_points(t_pred,cluster_x,cluster_y,tol=2*refit_time_threshold)
    # FILTER: keep only crossing points
    cross_set = set(zip(x_cross, y_cross))
    mask = np.array([(x, y) in cross_set for x, y in zip(cluster_x, cluster_y)])
    # build filtered cluster dataframe
    row_filt = rows.iloc[idx].iloc[mask]
    filtered_rows.append(row_filt)
# final filtered dataframe:
CSVfile_data = pd.concat(filtered_rows, ignore_index=True)

# %% Representing hyperbolic fit results:
# Plot DBSCAN and merged cluster bounding boxes
unique_clusters = np.sort(dbscan_merged_df["cluster_id"].unique())
FigName = "dbscan_merged_hyperfit_df"
title_str = (
    f"{len(CSVfile_data)} KVP picks in "
    f"{HyperFits_df['cluster_id'].nunique()} clusters\n"
    f"Min. fit points = {min_fit_points} | "
    f"Early-time weight = {early_time_weight} "
    f"(power = {early_time_weight_power})\n"
    f"Refit th. = ±{refit_time_threshold}s | "
    f"Crossing-point th. = ±{2 * refit_time_threshold}s"
)

fig, ax = plt.subplots(figsize=(8, 8))
# Plot all KVP picks used for clustering
ax.scatter(x,y,s=8,marker=".",color="grey",label=f"KVP picks ({len(x)})")
unique_clusters_plot = np.sort(CSVfile_data["cluster_id"].unique())
colors = mpl.colormaps.get_cmap('jet').resampled(len(unique_clusters_plot))
# Clusters
for i, cid in enumerate(unique_clusters_plot):
    rows = CSVfile_data.loc[CSVfile_data["cluster_id"] == cid]
    if rows.empty:
        continue
    x_cross = rows["x"].values 
    y_cross = rows["y"].values
    t_pred = np.sqrt(
        rows["t0"].iloc[0]**2 +
        (((y_cross/dist_ch) - rows["y0"].iloc[0])**2) /
        (rows["vapp"].iloc[0]**2)
    )
    idx = np.argsort(y_cross)
    t_pred = t_pred[idx]
    y_pred = y_cross[idx]
    ax.plot(
        t_pred,
        y_pred,
        color=colors(i),
        linewidth=2.0,
        alpha=0.35
    )
    ax.scatter(
        rows["t0"].iloc[0],
        rows["y0"].iloc[0]*dist_ch,
        marker="*",
        s=100,
        color=colors(i),
        edgecolors="black",
        linewidths=0.8,
        zorder=5
    )
    ax.scatter(
        x_cross,
        y_cross,
        s=14,
        color=colors(i),
        marker=".",
        alpha=0.9
    )
# Figure formatting
ax.set_title(title_str, fontsize=TITLE_SIZE, fontweight="bold")
ax.set_xlabel("Time [s]", fontsize=LABEL_SIZE)
ax.set_ylim(10e3,27e3)
ax.set_ylabel("Distance [m]", fontsize=LABEL_SIZE)
ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
ax.tick_params(axis="both", which="minor", labelsize=TICK_SIZE)
ax.grid(True, linewidth=GRID_WIDTH, alpha=GRID_ALPHA)
# legend = ax.legend(loc="upper center",ncols=2,fontsize=LEGEND_SIZE,frameon=True)
legend.get_frame().set_edgecolor("black")
legend.get_frame().set_alpha(0.5)
fig.tight_layout()
fig.savefig(
    os.path.join(output_results, f"{FigName}.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)
# %%
