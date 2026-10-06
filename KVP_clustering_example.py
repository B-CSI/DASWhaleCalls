# %% Imports
# Standard library
import datetime as dt
import os
import shutil
import sys
import time
import re
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
from src.utils import *

# %% Load KVP picks
csv_path = os.path.join(".","results","2024_01_13_07h57m34s_HDAS_SAFE_DASWhaleCalls_example_KVPpicks.csv")
CSVfile_raw_data = pd.read_csv(csv_path,comment="#")

print("\n=== KVP PICKS ===")
print(f"Loaded: {csv_path}")
print(f"Number of picks: {len(CSVfile_raw_data)}")
print(f"Columns: {CSVfile_raw_data.columns.tolist()}")
print(CSVfile_raw_data.head())

# Plot X vs Y
FigName = "1_kvp_output"
title_str = "KVP picks output\n"

fig, ax = plt.subplots(figsize=(12, 7))
# Plot KVP picks
ax.scatter(
    CSVfile_raw_data["time_rel"],
    CSVfile_raw_data["Channel"],
    s=15,
    alpha=0.7,
    color="grey",
)
# Figure formatting
ax.set_title(
    title_str,
    fontsize=TITLE_SIZE,
    fontweight="bold"
)
ax.set_xlabel(
    "Time [s]",
    fontsize=LABEL_SIZE
)
ax.set_ylabel(
    "Channel",
    fontsize=LABEL_SIZE
)

ax.tick_params(
    axis="both",
    which="major",
    labelsize=TICK_SIZE
)

ax.tick_params(
    axis="both",
    which="minor",
    labelsize=TICK_SIZE
)

ax.grid(
    True,
    linewidth=GRID_WIDTH,
    alpha=GRID_ALPHA
)

fig.tight_layout()

fig.savefig(
    os.path.join(output_results, f"{FigName}.png"),
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)


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
x_raw, y_raw = CSVfile_raw_data_filt["time_rel"],(CSVfile_raw_data_filt["Channel"])*dist_ch
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
FigName = "2_run_dbscan_output"
title_str = f"DBSCAN: {dbscan_df['cluster_id'].nunique()} clusters | {n_noise} noise | eps = {eps_m} m\n"

fig, ax = plt.subplots(figsize=(12,7))

# Plot all KVP picks used for clustering
ax.scatter(
    x, y,
    s=8,
    marker=".",
    color="grey",
    label=f"KVP picks ({len(x)})"
)

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

    # Cluster ID label: bottom-right, outside the rectangle
    ax.annotate(
        str(cluster.cluster_id),
        xy=(
            cluster.x0 + cluster.Xsize,
            cluster.y0
        ),
        xytext=(5, -5),
        textcoords="offset points",
        ha="left",
        va="top",
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

legend = ax.legend(
    loc="upper center",
    ncols=2,
    fontsize=LEGEND_SIZE,
    frameon=True
)
legend.get_frame().set_edgecolor("black")
legend.get_frame().set_alpha(0.5)

fig.tight_layout()

fig.savefig(
    os.path.join(output_results, f"{FigName}.png"),
    dpi=300,
    bbox_inches="tight",
)

plt.show()
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
# %% Picks asociates with a cluster:
# Coordinates of the original KVP picks
CSVfile_clustered = CSVfile_raw_data_filt.copy()
CSVfile_clustered["x"] = CSVfile_clustered["time_rel"]
CSVfile_clustered["y"] = CSVfile_clustered["Channel"] * dist_ch
# Initialize final cluster ID
CSVfile_clustered["cluster_id"] = -1
for cluster in dbscan_merged_df.itertuples():
    x_min = cluster.x0
    x_max = cluster.x0 + cluster.Xsize
    y_min = cluster.y0
    y_max = cluster.y0 + cluster.Ysize
    mask = (
        (CSVfile_clustered["x"] >= x_min) &
        (CSVfile_clustered["x"] <= x_max) &
        (CSVfile_clustered["y"] >= y_min) &
        (CSVfile_clustered["y"] <= y_max)
    )
    CSVfile_clustered.loc[mask, "cluster_id"] = cluster.cluster_id
CSVfile_clustered = CSVfile_clustered[CSVfile_clustered["cluster_id"] >= 0].copy()

print("\n=== FINAL CLUSTERED PICKS ===")
print(f"Original picks: {len(CSVfile_raw_data_filt)}")
print(f"Clustered picks: {len(CSVfile_clustered)}")
print(f"Unclustered/noise picks: {len(CSVfile_raw_data_filt) - len(CSVfile_clustered)}")

print("\n=== PICKS PER FINAL CLUSTER ===")
print(
    CSVfile_clustered["cluster_id"]
    .value_counts()
    .sort_index()
)

# %% Representing DBSCAN+merged results:
# Plot DBSCAN and merged cluster bounding boxes
FigName = "3_merge_dbscan_clusters_output"
title_str = (
    f"DBSCAN: {dbscan_df['cluster_id'].nunique()} clusters | {n_noise} noise | eps = {eps_m} m\n"
    f"Merged: {len(dbscan_merged_df)} clusters | Xdist = ±{Xdist} s | Ydist = ±{Ydist} m"
)

fig, ax = plt.subplots(figsize=(12,7))

# Plot all KVP picks used for clustering
ax.scatter(
    x, y,
    s=8,
    marker=".",
    color="grey",
    label=f"KVP picks ({len(x)})"
)

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
        alpha=0.5,
    )
)

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
            alpha=0.5,
        )
    )

    # cluster_id: bottom-right, outside the rectangle
    ax.annotate(
        str(cluster.cluster_id),
        xy=(
            cluster.x0 + cluster.Xsize,
            cluster.y0
        ),
        xytext=(5, -5),              # desplazamiento hacia fuera
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="black",
        alpha=0.8,
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
        alpha=0.5,
    )
)

ax.scatter(
    CSVfile_clustered["x"],
    CSVfile_clustered["y"],
    s=14,
    marker=".",
    color="tab:blue",
    label=f"Clustered KVP picks ({len(CSVfile_clustered)})"
)

# Figure formatting
ax.set_title(title_str, fontsize=TITLE_SIZE, fontweight="bold")
ax.set_xlabel("Time [s]", fontsize=LABEL_SIZE)
# ax.set_xlim(305, 330)
ax.set_ylabel("Distance [m]", fontsize=LABEL_SIZE)

ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
ax.tick_params(axis="both", which="minor", labelsize=TICK_SIZE)

ax.grid(True, linewidth=GRID_WIDTH, alpha=GRID_ALPHA)

legend = ax.legend(
    loc="upper center",
    ncols=2,
    fontsize=LEGEND_SIZE,
    frameon=True
)
legend.get_frame().set_edgecolor("black")
legend.get_frame().set_alpha(0.5)

fig.tight_layout()

fig.savefig(
    os.path.join(output_results, f"{FigName}.png"),
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)


# %% Hyperbolic fit application (using channels):
# from scripts.kvp_clustering import fit_cluster

# fits=[]
# for cluster_id, cluster_data_df in CSVfile_clustered.groupby("cluster_id"):
#     fit =  fit_cluster(
#             cluster_data=cluster_data_df,
#             min_points=min_fit_points,
#             grid_size=fit_grid_size,
#             early_time_weight=early_time_weight,
#             early_time_weight_power=early_time_weight_power,
#             refit=True,
#             refit_time_threshold=refit_time_threshold,
#             refit_no_early_weight=True,
#         )
#     fits.append(fit)
# HyperFits_df = pd.DataFrame([asdict(f) for f in fits if f is not None])

# # %% Representing hyperbolic fit results:
# FigName = "4_fit_cluster_output"
# title_str = (
#     f"{len(CSVfile_raw_data)} KVP picks in "
#     f"{HyperFits_df['cluster_id'].nunique()} clusters\n"
#     f"Min. fit points = {min_fit_points} | "
#     f"Early-time weight = {early_time_weight} "
#     f"(power = {early_time_weight_power})\n"
#     f"Refit th. = ±{refit_time_threshold}s | "
#     f"Crossing-point th. = ±{2 * refit_time_threshold}s"
# )

# fig, ax = plt.subplots(figsize=(12,7))
# # Plot all KVP picks used for clustering
# ax.scatter(
#     x,
#     y,
#     s=8,
#     marker=".",
#     color="grey",
#     label=f"KVP picks ({len(x)})"
# )
# # Colormap: one colour per fitted cluster
# unique_clusters_plot = np.sort(
#     HyperFits_df["cluster_id"].unique()
# )
# colors = mpl.colormaps.get_cmap("jet").resampled(
#     len(unique_clusters_plot)
# )
# # Plot hyperbolic fits
# for i, cid in enumerate(unique_clusters_plot):
#     # Get the fit parameters for this cluster
#     fit_row = HyperFits_df.loc[HyperFits_df["cluster_id"] == cid].iloc[0]
#     t0 = fit_row["t0"]
#     y0 = fit_row["y0"]
#     vapp = fit_row["vapp"]
#     # Get the original KVP picks belonging to this cluster
#     rows = CSVfile_clustered.loc[CSVfile_clustered["cluster_id"] == cid].copy()
#     if rows.empty:
#         continue
#     y_data = rows["Channel"].to_numpy(dtype=float)
#     y_curve = np.linspace(np.min(y_data),np.max(y_data),300)
#     t_curve = np.sqrt(t0**2 + ((y_curve - y0)**2) / (vapp**2))
#     # Convert channel -> distance [m]
#     y_curve_plot = y_curve * dist_ch
#     # Plot fitted hyperbola
#     ax.plot(t_curve,y_curve_plot,color=colors(i),linewidth=2.0,alpha=0.5)
#     # Plot hyperbola apex
#     ax.scatter(
#         t0,
#         y0 * dist_ch,
#         marker="*",
#         s=100,
#         color=colors(i),
#         edgecolors="black",
#         linewidths=0.8,
#         zorder=5,
#     )
#     # Plot KVP picks belonging to this cluster
#     ax.scatter(
#         rows["x"],
#         rows["y"],
#         s=14,
#         color=colors(i),
#         marker=".",
#         alpha=0.9,
#     )
# # Figure formatting
# ax.set_title(
#     title_str,
#     fontsize=TITLE_SIZE,
#     fontweight="bold"
# )
# ax.set_xlabel(
#     "Time [s]",
#     fontsize=LABEL_SIZE
# )
# ax.set_ylabel(
#     "Distance [m]",
#     fontsize=LABEL_SIZE
# )
# ax.set_ylim(
#     10e3,
#     27e3
# )
# ax.tick_params(
#     axis="both",
#     which="major",
#     labelsize=TICK_SIZE
# )
# ax.tick_params(
#     axis="both",
#     which="minor",
#     labelsize=TICK_SIZE
# )
# ax.grid(
#     True,
#     linewidth=GRID_WIDTH,
#     alpha=GRID_ALPHA
# )
# # Save figure
# fig.tight_layout()
# fig.savefig(
#     os.path.join(
#         output_results,
#         f"{FigName}.png"
#     ),
#     dpi=300,
#     bbox_inches="tight",
# )
# plt.show()
# plt.close(fig)
# # %%

# %% Hyperbolic fit application — using physical distance [m]

from dataclasses import asdict
from scripts.kvp_clustering import fit_cluster


# ============================================================
# Prepare data for fitting
# ============================================================
#
# IMPORTANT:
# fit_cluster() now works entirely in physical units.
#
# x = time [s]
# y = distance [m]
#
# Therefore, the fitting routine must receive:
#
#     cluster_data["x"]        -> time [s]
#     cluster_data["y_fit"]    -> distance [m]
#
# We keep the original "y" column used for plotting, but create
# an explicit column for the fit to avoid ambiguity.
# ============================================================

CSVfile_clustered["y_fit"] = (
    CSVfile_clustered["Channel"] * dist_ch
)


# ============================================================
# Run hyperbolic fit for every cluster
# ============================================================

fits = []

for cluster_id, cluster_data_df in (
    CSVfile_clustered.groupby("cluster_id")
):

    fit = fit_cluster(
        cluster_data=cluster_data_df.rename(
            columns={"y_fit": "y"}
        ),
        min_points=min_fit_points,
        grid_size=fit_grid_size,
        early_time_weight=early_time_weight,
        early_time_weight_power=early_time_weight_power,
        refit=True,
        refit_time_threshold=refit_time_threshold,
        refit_no_early_weight=True,
    )

    fits.append(fit)


# ============================================================
# Create DataFrame with fit results
# ============================================================

HyperFits_df = pd.DataFrame(
    [
        asdict(f)
        for f in fits
        if f is not None
    ]
)

print("\n=== HYPERBOLIC FIT RESULTS ===")

if HyperFits_df.empty:

    print("No valid hyperbolic fits were obtained.")

else:

    print(
        f"Number of fitted clusters: "
        f"{HyperFits_df['cluster_id'].nunique()}"
    )

    print(
        HyperFits_df[
            [
                "cluster_id",
                "y0",
                "t0",
                "vapp",
                "rmse",
                "r2",
                "n_points",
                "n_removed",
            ]
        ]
    )


# %% Representing hyperbolic fit results

FigName = "4_fit_cluster_output"

title_str = (
    f"{len(CSVfile_raw_data)} KVP picks in "
    f"{HyperFits_df['cluster_id'].nunique()} clusters\n"
    f"Min. fit points = {min_fit_points} | "
    f"Early-time weight = {early_time_weight} "
    f"(power = {early_time_weight_power})\n"
    f"Refit th. = ±{refit_time_threshold}s | "
    f"Crossing-point th. = ±{2 * refit_time_threshold}s"
)


fig, ax = plt.subplots(
    figsize=(12, 7)
)


# ============================================================
# Plot all KVP picks used for clustering
# ============================================================

ax.scatter(
    x,
    y,
    s=8,
    marker=".",
    color="grey",
    label=f"KVP picks ({len(x)})"
)


# ============================================================
# Colormap: one colour per fitted cluster
# ============================================================

unique_clusters_plot = np.sort(
    HyperFits_df["cluster_id"].unique()
)

colors = mpl.colormaps.get_cmap(
    "jet"
).resampled(
    len(unique_clusters_plot)
)


# ============================================================
# Plot hyperbolic fits
# ============================================================

for i, cid in enumerate(unique_clusters_plot):

    # --------------------------------------------------------
    # Get fit parameters
    #
    # y0  -> metres
    # t0  -> seconds
    # vapp -> m/s
    # --------------------------------------------------------

    fit_row = HyperFits_df.loc[
        HyperFits_df["cluster_id"] == cid
    ].iloc[0]

    t0 = float(
        fit_row["t0"]
    )

    y0 = float(
        fit_row["y0"]
    )

    vapp = float(
        fit_row["vapp"]
    )


    # --------------------------------------------------------
    # Get KVP picks belonging to this cluster
    # --------------------------------------------------------

    rows = CSVfile_clustered.loc[
        CSVfile_clustered["cluster_id"] == cid
    ].copy()

    if rows.empty:
        continue


    # --------------------------------------------------------
    # Distance range of this cluster
    #
    # IMPORTANT:
    # Use physical distance [m].
    #
    # No channel conversion is needed here.
    # --------------------------------------------------------

    y_data = rows["y_fit"].to_numpy(
        dtype=float
    )

    y_curve = np.linspace(
        np.min(y_data),
        np.max(y_data),
        300
    )


    # --------------------------------------------------------
    # Evaluate fitted hyperbola
    #
    # t² = t0² + (y - y0)² / vapp²
    #
    # y is already in metres.
    # --------------------------------------------------------

    t_curve = np.sqrt(
        t0**2
        + (
            (y_curve - y0)**2
            / vapp**2
        )
    )


    # --------------------------------------------------------
    # Plot fitted hyperbola
    # --------------------------------------------------------

    ax.plot(
        t_curve,
        y_curve,
        color=colors(i),
        linewidth=2.0,
        alpha=0.5,
    )


    # --------------------------------------------------------
    # Plot hyperbola apex
    # --------------------------------------------------------

    ax.scatter(
        t0,
        y0,
        marker="*",
        s=100,
        color=colors(i),
        edgecolors="black",
        linewidths=0.8,
        zorder=5,
    )


    # --------------------------------------------------------
    # Plot KVP picks belonging to this cluster
    #
    # rows["y"] is already the physical distance [m]
    # used during clustering.
    # --------------------------------------------------------

    ax.scatter(
        rows["x"],
        rows["y"],
        s=14,
        color=colors(i),
        marker=".",
        alpha=0.9,
    )


# ============================================================
# Figure formatting
# ============================================================

ax.set_title(
    title_str,
    fontsize=TITLE_SIZE,
    fontweight="bold"
)

ax.set_xlabel(
    "Time [s]",
    fontsize=LABEL_SIZE
)

ax.set_ylabel(
    "Distance [m]",
    fontsize=LABEL_SIZE
)

ax.set_ylim(
    10e3,
    27e3
)

ax.tick_params(
    axis="both",
    which="major",
    labelsize=TICK_SIZE
)

ax.tick_params(
    axis="both",
    which="minor",
    labelsize=TICK_SIZE
)

ax.grid(
    True,
    linewidth=GRID_WIDTH,
    alpha=GRID_ALPHA
)


# ============================================================
# Save figure
# ============================================================

fig.tight_layout()

fig.savefig(
    os.path.join(
        output_results,
        f"{FigName}.png"
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)

# %%
