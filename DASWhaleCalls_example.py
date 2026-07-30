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

from config.parameters import *

# %% CSVfile reading:
CSVfile = r"data_test\20240113_kvp_raw_picks.csv"
CSVfile_raw_data = pd.read_csv(CSVfile, sep=";", comment="#")

print(f"CSVfile_raw_data: {len(CSVfile_raw_data)} samples in {str(dt.timedelta(seconds=float(np.max(CSVfile_raw_data['time_rel']) - np.min(CSVfile_raw_data['time_rel'])))).split('.')[0]} ({len(CSVfile_raw_data)/(np.max(CSVfile_raw_data['time_rel']) - np.min(CSVfile_raw_data['time_rel'])):.1f}/s)")

# %% CSVfile filtering: 
CSVfile_raw_data_filt = CSVfile_raw_data.copy()
Nbands_max = 4
LBand, Hband = 15, 30
CSVfile_raw_data_filt = CSVfile_raw_data_filt[CSVfile_raw_data_filt["Number_bands"]<=Nbands_max]
CSVfile_raw_data_filt = CSVfile_raw_data_filt[CSVfile_raw_data_filt["Lowest_band"]>=LBand]
CSVfile_raw_data_filt = CSVfile_raw_data_filt[CSVfile_raw_data_filt["Highest_band"]>=LBand]
CSVfile_raw_data_filt = CSVfile_raw_data_filt[CSVfile_raw_data_filt["Lowest_band"]<=Hband]
CSVfile_raw_data_filt = CSVfile_raw_data_filt[CSVfile_raw_data_filt["Highest_band"]<=Hband]
print(f"CSVfile_raw_data_filt: {len(CSVfile_raw_data_filt)} samples in {str(dt.timedelta(seconds=float(np.max(CSVfile_raw_data_filt['time_rel']) - np.min(CSVfile_raw_data_filt['time_rel'])))).split('.')[0]} ({len(CSVfile_raw_data_filt)/(np.max(CSVfile_raw_data_filt['time_rel']) - np.min(CSVfile_raw_data_filt['time_rel'])):.1f}/s)")

# %% Representing data: 
x_raw, y_raw = CSVfile_raw_data["time_rel"],CSVfile_raw_data["Channel"]*dist_ch
x, y = CSVfile_raw_data_filt["time_rel"],CSVfile_raw_data_filt["Channel"]*dist_ch

FigName = "CSVfile_raw_data"

fig, ax = plt.subplots(figsize=(10, 6))
scatter = plt.scatter(x_raw, y_raw, marker='.', s=6, c='grey', alpha=0.5, label=f"{len(x_raw)} KVP raw picks")
scatter = plt.scatter(x, y, marker='.', s=8, c='black', alpha=1, label=f"{len(x)} samples to analysis")
ax.set_xlabel("Time [s]", fontsize=LABEL_SIZE)
ax.set_ylabel("Distance [m]", fontsize=LABEL_SIZE)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.legend(ncols=2,loc="lower left",fontsize=LEGEND_SIZE,frameon=True)
ax.grid(True,linewidth=0.3,alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="both",which="major",labelsize=TICK_SIZE)
ax.tick_params(axis="both", which="minor", labelsize=TICK_SIZE)
fig.tight_layout()
fig.savefig(os.path.join(output_results,f"{FigName}.png"),dpi=300,bbox_inches="tight")
plt.close(fig)

# %% Spatio-temporal clustering and hyperbolic fitting of KVP picks
# %% DBSCAN application:
from scripts.kvp_clustering import run_dbscan

x_prima = c_ref * x #[m]
y_prima = y.copy() #[m]
dbscan_df, n_noise = run_dbscan(x_prima, y_prima, dist_eps=eps_m, min_pts=min_pts)

dbscan_df['x0'] /= c_ref
dbscan_df['Xsize'] /= c_ref

# %% Bounding boxes for clusters creation:
dbscan_df["xmin"] = dbscan_df["x0"] - marginBB_x
dbscan_df["xmax"] = dbscan_df["x0"] + dbscan_df["Xsize"] + marginBB_x
dbscan_df["ymin"] = dbscan_df["y0"] - marginBB_y
dbscan_df["ymax"] = dbscan_df["y0"] + dbscan_df["Ysize"] + marginBB_y

# %% Cluster merging:
from scripts.kvp_clustering import merge_dbscan_clusters

dbscan_merged_df = merge_dbscan_clusters(dbscan_df, x_dist=Xdist, y_dist=Ydist)

# Count the number of picks inside each merged cluster
x_array = CSVfile_raw_data_filt["time_rel"].to_numpy()
y_array = (CSVfile_raw_data_filt["Channel"] * dist_ch).to_numpy()

cluster_sizes = []
for cluster in dbscan_merged_df.itertuples():
    inside = (
        (x_array >= cluster.xmin) & (x_array <= cluster.xmax) &
        (y_array >= cluster.ymin) & (y_array <= cluster.ymax)
    )
    cluster_sizes.append(np.count_nonzero(inside))
dbscan_merged_df["Npts"] = cluster_sizes

# Assign a cluster ID to each filtered KVP pick
clustered_data_df = CSVfile_raw_data_filt.copy()
clustered_data_df["x"] = clustered_data_df["time_rel"]
clustered_data_df["y"] = clustered_data_df["Channel"] * dist_ch
clustered_data_df["cluster_id"] = np.nan

for cluster in dbscan_merged_df.itertuples():
    inside = (
        (clustered_data_df["x"] >= cluster.xmin) &
        (clustered_data_df["x"] <= cluster.xmax) &
        (clustered_data_df["y"] >= cluster.ymin) &
        (clustered_data_df["y"] <= cluster.ymax)
    )
    clustered_data_df.loc[inside, "cluster_id"] = cluster.cluster_id
# Keep only picks assigned to a merged cluster
clustered_data_df = (
    clustered_data_df
    .dropna(subset=["cluster_id"])
    .reset_index(drop=True)
)
clustered_data_df["cluster_id"] = clustered_data_df["cluster_id"].astype(int)

# Renumber clusters from left to right (increasing time)
cluster_order = (
    clustered_data_df
    .groupby("cluster_id")["x"]
    .mean()
    .sort_values()
    .index
)
cluster_mapping = {
    old_id: new_id
    for new_id, old_id in enumerate(cluster_order)
}
clustered_data_df["cluster_id"] = (
    clustered_data_df["cluster_id"]
    .map(cluster_mapping)
)

# %% Representing DBSCAN+merged results:
# Plot DBSCAN and merged cluster bounding boxes
unique_clusters = np.sort(clustered_data_df["cluster_id"].unique())
FigName = "dbscan_merged_df"
title_str = (
    f"DBSCAN: {dbscan_df['cluster_id'].nunique()} clusters | {n_noise} noise | eps = {eps_m} m\n"
    f"Merged: {len(unique_clusters)} clusters | Xdist = ±{Xdist} s | Ydist = ±{Ydist} m"
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
for cluster_id in unique_clusters:
    cluster = clustered_data_df.loc[
        clustered_data_df["cluster_id"] == cluster_id
    ]
    cluster_x = cluster["x"].to_numpy()
    cluster_y = cluster["y"].to_numpy()
    ax.scatter(cluster_x,cluster_y,s=14,marker=".",color="tab:blue")
    ax.add_patch(
        patches.Rectangle(
            (cluster_x.min(), cluster_y.min()),
            cluster_x.max() - cluster_x.min(),
            cluster_y.max() - cluster_y.min(),
            linewidth=2.5,
            linestyle="-",
            edgecolor="black",
            facecolor="none",
            alpha=0.8,
        )
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
ax.scatter([],[],s=14,marker=".",color="tab:blue",label=f"Clustered KVP picks ({len(clustered_data_df)})")
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

# %% Hyperbolic fit application:
from scripts.kvp_clustering import fit_cluster

fits=[]
for cluster_id, cluster_data_df in clustered_data_df.groupby("cluster_id"):
    # cluster_id = 115
    # data_cluster = data[data["cluster_id"] == cluster_id]
    fit =  fit_cluster(
            cluster_data=cluster_data_df,
            min_points=min_fit_points,
            grid_size=fit_grid_size,
            early_time_weight=early_time_weight,
            early_time_weight_power=early_time_weight_power,
            refit=True,
            refit_time_threshold=refit_time_threshold,
            refit_no_early_weight=False,
        )
    fits.append(fit)
HyperFits_df = pd.DataFrame([asdict(f) for f in fits if f is not None])

# %%
from scripts.kvp_clustering import find_crossing_points

CSVfile_data = clustered_data_df.merge(
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
    x_cross, y_cross = find_crossing_points(t_pred,cluster_x,cluster_y,tol=refit_time_threshold)
    # FILTER: keep only crossing points
    cross_set = set(zip(x_cross, y_cross))
    mask = np.array([(x, y) in cross_set for x, y in zip(cluster_x, cluster_y)])
    # build filtered cluster dataframe
    row_filt = rows.iloc[idx].iloc[mask]
    filtered_rows.append(row_filt)
# final filtered dataframe:
CSVfile_data = pd.concat(filtered_rows, ignore_index=True)

# %% Cleaning CSVfile_data:
minNpts_HyperBranch = 5
# ---------------------------------------------------------------------
# Filter clusters before hyperbola analysis
# ---------------------------------------------------------------------

# Start from the complete dataset
CSVfile_data_filt = CSVfile_data.copy()

# ---------------------------------------------------------------------
# 1. Keep only clusters with enough picks on both hyperbola branches
# ---------------------------------------------------------------------
valid_clusters = []

for cluster_id, cluster in CSVfile_data_filt.groupby("cluster_id"):

    y0 = cluster["y0"].iloc[0]

    n_left = (cluster["Channel"] < y0).sum()
    n_right = (cluster["Channel"] > y0).sum()

    if (
        n_left >= minNpts_HyperBranch
        and n_right >= minNpts_HyperBranch
    ):
        valid_clusters.append(cluster_id)

CSVfile_data_filt = CSVfile_data_filt[
    CSVfile_data_filt["cluster_id"].isin(valid_clusters)
]

# ---------------------------------------------------------------------
# 2. Remove clusters with outlying y0 values (IQR criterion)
# ---------------------------------------------------------------------
cluster_summary = (
    CSVfile_data_filt
    .groupby("cluster_id")[["y0", "vapp"]]
    .first()
)

q1 = cluster_summary["y0"].quantile(0.25)
q3 = cluster_summary["y0"].quantile(0.75)
iqr = q3 - q1

valid_clusters = cluster_summary[
    cluster_summary["y0"].between(
        q1 - 1.5 * iqr,
        q3 + 1.5 * iqr
    )
].index

CSVfile_data_filt = CSVfile_data_filt[
    CSVfile_data_filt["cluster_id"].isin(valid_clusters)
]

# ---------------------------------------------------------------------
# 3. Remove clusters with outlying apparent velocities (IQR criterion)
# ---------------------------------------------------------------------
cluster_summary = (
    CSVfile_data_filt
    .groupby("cluster_id")[["vapp"]]
    .first()
)

q1 = cluster_summary["vapp"].quantile(0.25)
q3 = cluster_summary["vapp"].quantile(0.75)
iqr = q3 - q1

valid_clusters = cluster_summary[
    cluster_summary["vapp"].between(
        q1 - 1.5 * iqr,
        q3 + 1.5 * iqr
    )
].index

CSVfile_data_filt = CSVfile_data_filt[
    CSVfile_data_filt["cluster_id"].isin(valid_clusters)
]

print(
    f"Filtered dataset: {len(CSVfile_data_filt)} picks "
    f"in {CSVfile_data_filt['cluster_id'].nunique()} clusters"
)
CSVfile_data = CSVfile_data_filt.copy()
# %% Representing hyperbolic fit results:
# Plot DBSCAN and merged cluster bounding boxes
unique_clusters = np.sort(clustered_data_df["cluster_id"].unique())
FigName = "dbscan_merged_hyperfit_df"
title_str = (
    f"DBSCAN: {dbscan_df['cluster_id'].nunique()} clusters | {n_noise} noise | eps = {eps_m} m\n"
    f"Merged: {len(unique_clusters)} clusters | Xdist = ±{Xdist} s | Ydist = ±{Ydist} m"
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
