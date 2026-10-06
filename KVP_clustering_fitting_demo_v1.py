# %% Imports
# Standard library
import os
import numpy as np 

# Third-party libraries
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from config.parameters import *
from src.utils import *

# %% Load KVP picks
csv_path = os.path.join(".","results","2024_01_13_07h57m34s_HDAS_SAFE_DASWhaleCalls_example_KVPpicks.csv")
CSVfile_raw_data = pd.read_csv(csv_path,comment="#")

print("\n=== KVP PICKS ===")
print(f"Loaded: {csv_path}")
print(f"Number of picks: {len(CSVfile_raw_data)}")
print(f"Columns: {CSVfile_raw_data.columns.tolist()}")
print(CSVfile_raw_data.head())

# %% Representing the inut data (KVP picks):
plot_config = PlotConfig()
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
    fontsize=plot_config.title_size,
    fontweight="bold"
)
ax.set_xlabel(
    "Time [s]",
    fontsize=plot_config.label_size
)
ax.set_ylabel(
    "Channel",
    fontsize=plot_config.label_size
)
ax.tick_params(
    axis="both",
    which="major",
    labelsize=plot_config.tick_size
)
ax.tick_params(
    axis="both",
    which="minor",
    labelsize=plot_config.tick_size
)
ax.grid(
    True,
    linewidth=plot_config.grid_width,
    alpha=plot_config.grid_alpha
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
KVP_raw_picks_filt = CSVfile_raw_data.copy()
Nbands_max = 4
LBand, Hband = 15, 30
KVP_raw_picks_filt = KVP_raw_picks_filt[KVP_raw_picks_filt["Number_bands"]<=Nbands_max]
KVP_raw_picks_filt = KVP_raw_picks_filt[KVP_raw_picks_filt["Lowest_band"]>=LBand]
KVP_raw_picks_filt = KVP_raw_picks_filt[KVP_raw_picks_filt["Highest_band"]<=Hband]
# print(f"CSVfile_raw_data_filt: {len(CSVfile_raw_data_filt)} samples in {str(dt.timedelta(seconds=float(np.max(CSVfile_raw_data_filt['time_rel']) - np.min(CSVfile_raw_data_filt['time_rel'])))).split('.')[0]} ({len(CSVfile_raw_data_filt)/(np.max(CSVfile_raw_data_filt['time_rel']) - np.min(CSVfile_raw_data_filt['time_rel'])):.1f}/s)")

KVP_raw_picks = CSVfile_raw_data.copy()

# x and y for the spatio-temporal clustering and hyperbolic fitting of KVP picks:
KVP_raw_picks["x"] = KVP_raw_picks["time_rel"]
KVP_raw_picks["y"] = KVP_raw_picks["Channel"] * dist_ch

# %% Clustering:
from scripts.kvp_clustering_fitting import KVPpicks_clustering
KVP_clustered_picks = KVPpicks_clustering(
        KVP_raw_picks,
        eps_m,
        min_pts,
        c_sound,
        Xdist,
        Ydist,
        plotter=True,
        plt_saver=os.path.join(output_results, f"2_KVPpicks_clustering_output.png"),
        plt_config = plot_config
    )

# %% Hyperbolic fit:
from scripts.kvp_clustering_fitting import KVPpicks_Hfitting
KVP_Hfitted_picks = KVPpicks_Hfitting(
    KVP_raw_picks,
    KVP_clustered_picks,
    min_fit_points,
    fit_grid_size,
    early_time_weight,
    early_time_weight_power,
    refit_time_threshold,
    refit_bool,
    marginBB_y,
    plotter=True,
    plt_saver=os.path.join(output_results, f"3_KVPpicks_Hfitting_output.png"),
    plt_config = plot_config
)

# %% "Manual" selection of KVP picks
# Selection criteria.
y0_target = 18000.0          # Target hyperbola apex position [m]
y0_tolerance = 2000.0        # Allowed deviation from target y0 [m]

min_cluster_Xsize = 0.5      # Minimum cluster width [s]
max_cluster_Ysize = 10000.0   # Maximum cluster height [m]

# Boolean mask for clusters satisfying all criteria.
cluster_selection_mask = (
    (KVP_Hfitted_picks["cluster_Xsize"] >= min_cluster_Xsize)
    &
    (KVP_Hfitted_picks["cluster_Ysize"] <= max_cluster_Ysize)
    &
    (
        np.abs(
            KVP_Hfitted_picks["y0"] - y0_target
        )
        <= y0_tolerance
    )
)

# Keep only picks belonging to selected clusters.
KVP_selected_picks = (
    KVP_Hfitted_picks.loc[cluster_selection_mask]
    .copy()
)

print("\n=== KVP CLUSTER SELECTION ===")
print(
    f"Target y0 = {y0_target:.1f} m "
    f"± {y0_tolerance:.1f} m"
)
print(
    f"Minimum cluster Xsize = {min_cluster_Xsize:.3f} s"
)
print(
    f"Maximum cluster Ysize = {max_cluster_Ysize:.1f} m"
)

print("\nSelected clusters:")
print(
    KVP_selected_picks[
        [
            "cluster_id",
            "y0",
            "t0",
            "vapp",
            "cluster_Xsize",
            "cluster_Ysize",
        ]
    ]
    .drop_duplicates("cluster_id")
    .sort_values("cluster_id")
    .to_string(index=False)
)

print(
    f"\nSelected clusters: "
    f"{KVP_selected_picks['cluster_id'].nunique()}"
)

print(
    f"Selected KVP picks: "
    f"{len(KVP_selected_picks)}"
)

# %% Represent selected KVP picks
FigName = "4_KVPpicks_selected_output"
title_str = (
    f"Selected clusters: "
    f"{KVP_selected_picks['cluster_id'].nunique()} | "
    f"Selected KVP picks: {len(KVP_selected_picks)}\n"
    f"y0 = {y0_target:.0f} ± {y0_tolerance:.0f} m | "
    f"Xsize ≥ {min_cluster_Xsize:.2f} s | "
    f"Ysize ≤ {max_cluster_Ysize:.0f} m"
)

fig, ax = plt.subplots(
    figsize=(12, 7)
)

# ------------------------------------------------------------------
# Plot all raw KVP picks.
# ------------------------------------------------------------------

ax.scatter(
    KVP_raw_picks["x"],
    KVP_raw_picks["y"],
    s=8,
    marker=".",
    color="grey",
    alpha=0.6,
    label=f"Raw KVP picks ({len(KVP_raw_picks)})",
    zorder=1,
)

# ------------------------------------------------------------------
# Obtain selected cluster IDs.
# ------------------------------------------------------------------

selected_cluster_ids = np.sort(
    KVP_selected_picks["cluster_id"].unique()
)

# Create one color for each selected hyperbola.
colors = mpl.colormaps["jet"].resampled(
    max(len(selected_cluster_ids), 1)
)

cluster_colors = {
    cid: colors(i)
    for i, cid in enumerate(selected_cluster_ids)
}

# ------------------------------------------------------------------
# Plot selected fitted hyperbolas.
# ------------------------------------------------------------------

for cid in selected_cluster_ids:

    fit_row = KVP_Hfitted_picks.loc[
        KVP_Hfitted_picks["cluster_id"] == cid
    ].iloc[0]

    t0 = float(fit_row["t0"])
    y0 = float(fit_row["y0"])
    vapp = float(fit_row["vapp"])

    color = cluster_colors[cid]

    # Retrieve the merged cluster geometry.
    cluster_rows = KVP_clustered_picks.loc[
        KVP_clustered_picks["cluster_id"] == cid
    ]

    if cluster_rows.empty:
        continue

    cluster_y_min = float(
        cluster_rows["cluster_y0"].iloc[0]
    )

    cluster_y_max = (
        cluster_y_min
        + float(
            cluster_rows["cluster_Ysize"].iloc[0]
        )
    )

    # Extend fitted hyperbola vertically.
    y_min_fit = (
        cluster_y_min
        - marginBB_y
    )

    y_max_fit = (
        cluster_y_max
        + marginBB_y
    )

    # Evaluate fitted hyperbola.
    y_curve = np.linspace(
        y_min_fit,
        y_max_fit,
        1000,
    )

    t_curve = np.sqrt(
        t0**2
        + ((y_curve - y0) ** 2 / vapp**2)
    )

    ax.plot(
        t_curve,
        y_curve,
        color=color,
        linewidth=2.2,
        alpha=0.85,
        zorder=3,
    )

    # Plot hyperbola apex / vertex.
    ax.scatter(
        t0,
        y0,
        marker="*",
        s=120,
        color=color,
        edgecolors="black",
        linewidths=0.8,
        zorder=5,
    )

# ------------------------------------------------------------------
# Plot selected KVP picks.
# ------------------------------------------------------------------

if not KVP_selected_picks.empty:

    for cid, selected_cluster in (
        KVP_selected_picks.groupby("cluster_id")
    ):

        color = cluster_colors.get(
            cid,
            "black",
        )

        ax.scatter(
            selected_cluster["x"],
            selected_cluster["y"],
            s=22,
            marker=".",
            color=color,
            alpha=0.95,
            zorder=4,
        )

# ------------------------------------------------------------------
# Figure formatting.
# ------------------------------------------------------------------

ax.set_title(
    title_str,
    fontsize=plot_config.title_size,
    fontweight="bold",
)

ax.set_xlabel(
    "Time [s]",
    fontsize=plot_config.label_size,
)

ax.set_ylabel(
    "Distance [m]",
    fontsize=plot_config.label_size,
)

ax.tick_params(
    axis="both",
    which="major",
    labelsize=plot_config.tick_size,
)

ax.tick_params(
    axis="both",
    which="minor",
    labelsize=plot_config.tick_size,
)

ax.grid(
    True,
    linewidth=plot_config.grid_width,
    alpha=plot_config.grid_alpha,
)

fig.tight_layout()
fig.savefig(
    os.path.join(output_results, f"{FigName}.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.show()
plt.close(fig)
