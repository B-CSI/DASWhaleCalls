# %% Imports
# Standard library
import os
import numpy as np 

# Third-party libraries
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from datetime import datetime

from config.parameters import *

SCRIPT_NAME = "KVP_clustering_fitting_demo.py"
# %% Load KVP picks
csv_path = os.path.join(".","data_example","2024_01_13_07h57m44s_HDAS_SAFE_DASWhaleCalls_example.csv")
CSVfile_raw_data = pd.read_csv(csv_path,sep=";",comment="#")
print("\n=== KVP PICKS ===")
print(f"Loaded: {csv_path}")
print(f"Number of picks: {len(CSVfile_raw_data)}")
print(f"Columns: {CSVfile_raw_data.columns.tolist()}")
print(CSVfile_raw_data.head())

# %% Representing the inut data (KVP picks):
plot_config = PlotConfig()
FigName = "0_KVPpicks_raw"
title_str = "KVP picks raw\n"
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

# %% Data for the analysis:
KVP_raw_picks = CSVfile_raw_data.copy()
# x and y for the spatio-temporal clustering and hyperbolic fitting of KVP picks:
KVP_raw_picks["x"] = KVP_raw_picks["time_rel"]
KVP_raw_picks["y"] = KVP_raw_picks["Channel"] * dist_ch

# %% KVP_raw_picks filtering (not necessary with the kvp v0.2.0): 
# KVP_raw_picks_filt = CSVfile_raw_data.copy()
# Nbands_max = 4
# LBand, Hband = 15, 30
# KVP_raw_picks_filt = KVP_raw_picks_filt[KVP_raw_picks_filt["Number_bands"]<=Nbands_max]
# KVP_raw_picks_filt = KVP_raw_picks_filt[KVP_raw_picks_filt["Lowest_band"]>=LBand]
# KVP_raw_picks_filt = KVP_raw_picks_filt[KVP_raw_picks_filt["Highest_band"]<=Hband]
# print(f"Points: {len(KVP_raw_picks_filt)} before filtering → {len(KVP_raw_picks_filt)} after filtering | Removed: {len(KVP_raw_picks_filt)-len(KVP_raw_picks_filt)} ({100*(len(KVP_raw_picks_filt)-len(KVP_raw_picks_filt))/len(KVP_raw_picks_filt):.1f}%)")

# %% KVP_raw_picks filtering (necessary with the kvp v0.2.0): 
from scripts.kvp_clustering_fitting import KVPpicks_noise_filter
KVP_raw_picks_filt, KVP_noise_picks = KVPpicks_noise_filter(
    KVP_raw_picks,
    time_window=0.1,     # [s]
    min_span_m=10000.0,   # [m]
    min_picks=30,
    plotter=True,
    plt_saver=os.path.join(output_results, "1_KVPpicks_filt.png"),
    plt_config=plot_config,
)

# %% Clustering:
from scripts.kvp_clustering_fitting import KVPpicks_clustering
KVP_clustered_picks = KVPpicks_clustering(
        KVP_raw_picks_filt,
        eps_m,
        min_pts,
        c_sound,
        Xdist,
        Ydist,
        plotter=True,
        plt_saver=os.path.join(output_results, f"2_KVPpicks_clustering.png"),
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
    plt_saver=os.path.join(output_results, f"3_KVPpicks_Hfitting.png"),
    plt_config = plot_config
)

# %% Results Filt+Cluster+HFitting:
print("\nResults Filt+Cluster+HFitting:")
print(
    KVP_Hfitted_picks[
        [
            "cluster_id",
            "n_points",
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

# %% "Manual" selection of KVP picks
# Selection criteria:
# y0_target = 18000.0          # Target hyperbola apex position [m]
# y0_tolerance = 2000.0        # Allowed deviation from target y0 [m]
min_cluster_Xsize = 0.5      # Minimum cluster width [s]
min_cluster_Ysize = 1000.0   # Minimum cluster height [m]
cluster_selection_mask = (
    (KVP_Hfitted_picks["cluster_Xsize"] >= min_cluster_Xsize)
    &
    (KVP_Hfitted_picks["cluster_Ysize"] >= min_cluster_Ysize)
    # &
    # (
    #     np.abs(
    #         KVP_Hfitted_picks["y0"] - y0_target
    #     )
    #     <= y0_tolerance
    # )
)
# cluster_selection_mask = KVP_Hfitted_picks["n_points"] >= (2*min_pts)

# Keep only picks belonging to selected clusters.
KVP_selected_picks = (
    KVP_Hfitted_picks.loc[cluster_selection_mask]
    .copy()
)

print("\nSelected clusters:")
print(
    KVP_selected_picks[
        [
            "cluster_id",
            "n_points",
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
FigName = "4_KVPpicks_selected"
title_str = (
    f"Selected clusters: "
    f"{KVP_selected_picks['cluster_id'].nunique()} | "
    f"Selected KVP picks: {len(KVP_selected_picks)}\n"
    f"X-size ≥ {min_cluster_Xsize} s | "
    f"Y-size ≥ {min_cluster_Ysize:.0f} m"
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
    cluster_rows = KVP_selected_picks.loc[
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
        linewidth=3,
        alpha=0.3,
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

# %%


# %% Save the CSV with the KVP picks selected:
input_stem = os.path.splitext(
    os.path.basename(csv_path)
)[0]

# Remove "_rawKVPpicks" from the input filename
input_stem = input_stem.replace(
    "_rawKVPpicks",
    "",
)

output_filename = (
    input_stem
    + "_selectedKVPpicks.csv"
)

output_file = os.path.join(
    ".",
    "data_example",
    output_filename,
)

# ------------------------------------------------------------
# Creation date
# ------------------------------------------------------------

creation_date = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S"
)

cluster_id_order = (
    KVP_selected_picks[
        ["cluster_id", "t0"]
    ]
    .drop_duplicates("cluster_id")
    .sort_values("t0")
    ["cluster_id"]
    .tolist()
)
cluster_id_mapping = {
    old_id: new_id
    for new_id, old_id in enumerate(cluster_id_order)
}
KVP_selected_picks["cluster_id"] = (
    KVP_selected_picks["cluster_id"]
    .map(cluster_id_mapping)
)

# ------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------
with open(
    output_file,
    "w",
    encoding="utf-8",
    newline="",
) as f:

    f.write(
        "# KVP SELECTED PICKS\n"
    )

    f.write(
        f"# created by {SCRIPT_NAME} ({creation_date})\n"
    )

    f.write(
        f"# input_file: "
        f"{os.path.basename(csv_path)}\n"
    )

    f.write(
        "# description = Selected KVP picks after "
        "spatio-temporal clustering and hyperbolic fitting.\n"
    )

    f.write(
        "# DATA\n"
    )

    KVP_selected_picks.to_csv(
        f,
        index=False,
        sep=";",
    )


print(
    f"\nCSV saved to:\n"
    f"{output_file}"
)

print(
    f"Rows written: "
    f"{len(KVP_selected_picks)}"
)