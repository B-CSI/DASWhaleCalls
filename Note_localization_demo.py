# %% Imports
# Standard library
import os
import numpy as np 
import ast

# Third-party libraries
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from config.parameters import *

import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import datetime as dt

SCRIPT_NAME = "Note_localization_demo.py"
plot_config = PlotConfig()
# %% Load KVP picks
csv_path = os.path.join(".","data_example","2024_01_13_07h57m44s_HDAS_SAFE_DASWhaleCalls_example_selectedKVPpicks.csv")
KVP_selected_picks = pd.read_csv(csv_path,sep=";",comment="#")

print("\n=== KVP SELECTED PICKS ===")
print(
    f"Loaded file:       {csv_path}"
)
print(
    f"Number of picks:    {len(KVP_selected_picks)}"
)

print(f"Number of clusters: {KVP_selected_picks['cluster_id'].nunique()}")
picks_per_cluster = (
    KVP_selected_picks
    .groupby("cluster_id")
    .size()
)
print(
    f"Picks/cluster:      "
    f"[{picks_per_cluster.min()},{picks_per_cluster.max()}] "
    f"(median = {picks_per_cluster.median():.0f})"
)


# %% Cable coordinates:
CSVfile_cable = "data_test/cable_geometry.csv"
CSVfile_cable_data = pd.read_csv(CSVfile_cable, sep=";", comment="#") 

dist_cable = CSVfile_cable_data["Channel"].values * dist_ch

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# # ------------------------------------------------------------------
# # Cable geometry in the local Cartesian reference system
# # ------------------------------------------------------------------
# axes[0].plot(
#     CSVfile_cable_data["X[m]"],
#     CSVfile_cable_data["Y[m]"],
#     "-k",
#     linewidth=1.5,
# )
# axes[0].scatter(
#     CSVfile_cable_data["X[m]"],
#     CSVfile_cable_data["Y[m]"],
#     s=8,
#     c="tab:blue",
# )
# axes[0].set_xlabel("X [m]", fontsize=LABEL_SIZE)
# axes[0].set_ylabel("Y [m]", fontsize=LABEL_SIZE)
# axes[0].set_title("Cable geometry", fontsize=TITLE_SIZE)
# axes[0].set_aspect("equal")
# axes[0].grid(True, linewidth=0.3, alpha=0.4)

# # ------------------------------------------------------------------
# # Elevation profile
# # ------------------------------------------------------------------
# axes[1].plot(
#     CSVfile_cable_data["Channel"],
#     CSVfile_cable_data["Z[m]"],
#     "-k",
#     linewidth=1.5,
# )
# axes[1].scatter(
#     CSVfile_cable_data["Channel"],
#     CSVfile_cable_data["Z[m]"],
#     s=8,
#     c="tab:blue",
# )
# axes[1].set_xlabel("Channel", fontsize=LABEL_SIZE)
# axes[1].set_ylabel("Z [m]", fontsize=LABEL_SIZE)
# axes[1].set_title("Cable elevation", fontsize=TITLE_SIZE)
# axes[1].grid(True, linewidth=0.3, alpha=0.4)

# for ax in axes:
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)

# fig.tight_layout()
# plt.show()

# %% Positioning 
from matplotlib.colors import LogNorm
from scripts.kvp_positioning import gridCreator2Dsquare, localization_misfit

Results_df_columns = (
    ['cluster_id', 'unixtime', 'Npicks', 'Prob', 'x', 'y', 'dx', 'dy', 'Ax', 'Ay']
)
Results_df = [] # one per cluster_id

Nx = int(2 * maxX / dx_res) + 1
Ny = int(2 * maxY / dy_res) + 1
Ntotal = Nx * Ny

fig_counter = 5
for idx, (cluster, KVPpicks_cluster) in enumerate(tqdm(KVP_selected_picks.groupby('cluster_id'), total=KVP_selected_picks['cluster_id'].nunique(), desc="DAS positioning")):
    # cluster, KVPpicks_cluster = list(KVP_selected_picks.groupby('cluster_id'))[idx]
    times_obs = KVPpicks_cluster["time_rel"].values 
    channels_obs = KVPpicks_cluster["Channel"].values

    dist_cable_obs = channels_obs*dist_ch
    R_obs_id = channels_obs.copy()     # Receiver IDs from observations
    T_obs_s  = times_obs.copy()        # Observed arrival times [s]

    R_ref_id = R_obs_id[np.argmin(T_obs_s)]
    row = CSVfile_cable_data.loc[CSVfile_cable_data["Channel"] == R_ref_id]
    lon_closest_m = row["X[m]"].iloc[0]
    lat_closest_m = row["Y[m]"].iloc[0]

    points_cable = np.stack([CSVfile_cable_data["X[m]"], CSVfile_cable_data["Y[m]"]], axis=1)
    R_calc_xy_m = points_cable.copy() # All receivers (channels)

    indxs_obs = np.array([np.argmin(np.abs(dist_cable - d)) for d in dist_cable_obs])

    central_point_m = np.array([lon_closest_m, lat_closest_m])
    E_calc_xy_m = gridCreator2Dsquare(central_point_m, dx_res, dy_res, minX, maxX, minY, maxY, plotter=False)
    # water_points = WaterPointsFilt(points_grid, plotter=0, lon_center=lon_ref, lat_center=lat_ref)
    # E_calc_xy_m = water_points

    dist_ER_m = np.linalg.norm(R_calc_xy_m[:, None, :] - E_calc_xy_m[None, :, :], axis=2)
    T_calc_s = (dist_ER_m - np.min(dist_ER_m, axis=1, keepdims=True)) / c_sound
    T_calc_s = T_calc_s[indxs_obs, :]

    # print(f"Grid: {len(E_calc_xy_m)} samples; dx: {int(dx_res)}m; dy: {int(dy_res)}m\n[+/- {np.abs(int(maxX))},+/- {np.abs(int(maxY))}]m")
    
    misfit_s = np.empty(len(E_calc_xy_m))
    for i in range(len(E_calc_xy_m)):
        misfit_s[i], _, _ = localization_misfit(T_obs_s,T_calc_s[:, i],fs=fs,n_sigma=n_sigma_terr)

    best_idx = np.where(misfit_s == np.min(misfit_s))[0][0]
    best_point = E_calc_xy_m[best_idx, :]

    # Residuals at the best localization point
    T_calc_best = T_calc_s[:, best_idx]
    misfit, best_shift, inliers = localization_misfit(
        T_obs_s,
        T_calc_best,
        fs=fs,
        n_sigma=n_sigma_terr
    )
    # Residual = calculated - observed (after optimal time shift)
    residuals = T_calc_best - (T_obs_s - best_shift)
    # Horizontal bar plot
    inlier_mask = np.zeros(len(R_obs_id), dtype=bool)
    inlier_mask[inliers] = True


    # Figure: Localization time and residual analysis
    fig, axes = plt.subplots(1, 3,figsize=(18, 7),sharey=True)
    y = np.arange(len(R_obs_id))
    # AXIS 1: All channels (inliers + outliers)
    ax = axes[0]
    ax.barh(
        y[inlier_mask],
        residuals[inlier_mask] * 1000,
        height=0.7,
        label="Inliers",
        color="tab:blue",
    )
    ax.barh(
        y[~inlier_mask],
        residuals[~inlier_mask] * 1000,
        height=0.7,
        label="Outliers",
        color="tab:orange",
    )
    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
    )
    ax.set_title(
        "All channels",
        fontsize=plot_config.title_size,
        fontweight="bold",
    )
    ax.set_xlabel(
        "T_calc - T_obs [ms]",
        fontsize=plot_config.label_size,
    )
    ax.set_ylabel(
        "Channel",
        fontsize=plot_config.label_size,
    )
    ax.legend()

    # AXIS 2: Observed vs calculated arrival times
    ax = axes[1]
    # Calculated times
    ax.plot(
        T_calc_best - np.min(T_calc_best),
        y,
        linestyle="-",
        marker="",
        color="black",
        label="Calculated",
    )
    # Observed times (inliers only)
    ax.plot(
        T_obs_s[inlier_mask] - best_shift - np.min(T_calc_best),
        y[inlier_mask],
        linestyle="",
        marker="x",
        markersize=6,
        markeredgewidth=1.5,
        color="tab:blue",
        label="Observed",
    )
    ax.set_title(
        "Observed vs calculated",
        fontsize=plot_config.title_size,
        fontweight="bold",
    )
    ax.set_xlabel(
        "Time [s]",
        fontsize=plot_config.label_size,
    )
    ax.legend()

    # AXIS 3: Inliers residuals only
    ax = axes[2]

    ax.barh(
        y[inlier_mask],
        residuals[inlier_mask] * 1000,
        height=0.7,
        color="tab:blue",
        label="Inliers",
    )
    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
    )
    ax.set_title(
        "Inliers only",
        fontsize=plot_config.title_size,
        fontweight="bold",
    )
    ax.set_xlabel(
        "T_calc - T_obs [ms]",
        fontsize=plot_config.label_size,
    )
    ax.legend()

    # COMMON FORMATTING
    for ax in axes:
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

    # Overall figure title
    fig.suptitle(
        f"Cluster: {int(cluster)} | "
        f"Shift = {best_shift:.3f} s | "
        f"MAE = {misfit:.4f} s | "
        f"Inliers = {np.sum(inlier_mask)}/{len(inlier_mask)}",
        fontsize=plot_config.title_size,
        fontweight="bold",
    )

    plt.tight_layout()
    fig.savefig(
        os.path.join(
            output_results,
            f"5_NoteLoc_cluster{int(cluster)}_times.png",
        ),
        dpi=150,
        bbox_inches="tight",
    )
    # plt.show()
    plt.close(fig)

    # Figure: Localization probability
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        E_calc_xy_m[:, 0],
        E_calc_xy_m[:, 1],
        # c = misfit_s,
        c=np.maximum(misfit_s, 1e-6),       # Representation for better visualization
        norm=LogNorm(
            vmin=np.min(np.maximum(misfit_s, 1e-6)),
            vmax=np.max(np.maximum(misfit_s, 1e-6)),
        ),
        cmap="viridis_r",
        marker="s",
    )
    ax.plot(
        CSVfile_cable_data["X[m]"],
        CSVfile_cable_data["Y[m]"],
        color="red",
        linestyle="-",
        linewidth=2,
        label="DAS interrogation",
    )
    ax.plot(
        best_point[0],
        best_point[1],
        color="red",
        marker="*",
        linestyle="",
        label=f"Best: {misfit_s[best_idx]:.3f}s",
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(
        # r"MAE [s]",    
        r"MAE [s] (log scale)",     # Representation for better visualization
        rotation=270,
        labelpad=15,
        fontsize=plot_config.label_size,
    )
    cbar.ax.tick_params(
        labelsize=plot_config.tick_size,
    )
    ax.set_xlim(
        np.min(E_calc_xy_m[:, 0]),
        np.max(E_calc_xy_m[:, 0]),
    )
    ax.set_ylim(
        np.min(E_calc_xy_m[:, 1]),
        np.max(E_calc_xy_m[:, 1]),
    )
    # Figure formatting.
    ax.set_title(
        f"Cluster: {cluster}\n"
        f"Grid: {len(E_calc_xy_m)} samples; "
        f"dx: {dx_res}m; dy: {dy_res}m\n"
        f"[$\\pm${np.abs(int(maxX))},"
        f"$\\pm${np.abs(int(maxY))}]m",
        fontsize=plot_config.title_size,
        fontweight="bold",
    )
    ax.set_xlabel(
        "X [m]",
        fontsize=plot_config.label_size,
    )
    ax.set_ylabel(
        "Y [m]",
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
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    fig.savefig(
        os.path.join(
            output_results,
            f"6_NoteLoc_cluster{int(cluster)}.png",
        ),
        dpi=150,
        bbox_inches="tight",
    )
    # plt.show()
    plt.close(fig)
    plt.close("all")

    E_calc_x_m, E_calc_y_m = E_calc_xy_m[:,0], E_calc_xy_m[:,1]
    best_results = {
        'cluster_id': cluster,
        'unixtime': np.min(KVPpicks_cluster['unix_time']),
        'Npicks': len(KVPpicks_cluster),
        'Prob': misfit_s.tolist(),
        'x': E_calc_x_m.tolist(),
        'y': E_calc_y_m.tolist(),
        'dx': dx_res,
        'dy': dy_res,
        'Ax': np.abs(int(maxX)),
        'Ay': np.abs(int(maxY)),
    }
    if best_results is not None:
        Results_df.append(best_results)
        # Output data saved as one CSV per cluster
        CSVbase = os.path.basename(csv_path).split('.')[0]
        CSVname = f"{CSVbase}_NoteLoc_cluster{int(cluster)}.csv"
        metadata = [
            f"# Input: {os.path.basename(csv_path)}",
            f"# Output: {CSVname}",
            "# ----------------------------------------",
            "# Local tangent plane projection, Units: meters",
            f"# Cluster: {int(cluster)}/{len(KVP_selected_picks['cluster_id'].unique())}"
        ]
        params_line1 = (
            f"# dx={int(dx_res)}m; dy={int(dy_res)}m; "
            f"maxX={np.abs(int(maxX))}m; maxY={np.abs(int(maxY))}m"
        )
        metadata.append(params_line1)
        params_line2 = (
            f"# Npts_min_HyperBranch={int(minNpts_HyperBranch)}; "
            f"Npts_grid={int(Ntotal)} ({int(Nx)} x {int(Ny)}); "
            f"sigma_err={n_sigma_terr};"
        )
        metadata.append(params_line2)
        # DataFrame solo de este cluster
        cluster_df = pd.DataFrame([best_results], columns=Results_df_columns)
        with open(os.path.join(output_results, CSVname),'w',encoding='utf-8') as f:
            # Metadata:
            for line in metadata:
                f.write(line + "\n")
            # Data:
            cluster_df.to_csv(f, sep=";", index=False)
        print('%s created!' % CSVname)

Results_df = pd.DataFrame(Results_df, columns=Results_df_columns)

# %% Reading the csv files to (re-)create the Results_df
base_path = "results/2024_01_13_07h57m44s_HDAS_SAFE_DASWhaleCalls_example_selectedKVPpicks_NoteLoc_cluster"
dfs = []
for cluster in range(6):
    filename = f"{base_path}{cluster}.csv"
    df = pd.read_csv(
        filename,
        sep=';',
        comment='#'
    )
    # Convert string representations of lists into Python lists
    for col in ['x', 'y', 'Prob']:
        df[col] = df[col].apply(ast.literal_eval)

    df['Cluster'] = cluster
    dfs.append(df)
Results_df = pd.concat(dfs, ignore_index=True)

# %% Spatial representation using misfit percentile
percentile = 95  # Select the best (100-percentile)% (lowest misfit values)

fig, ax = plt.subplots(figsize=(8, 5))

x_all, y_all, t_all, misfit_all, cluster_all = [], [], [], [], []
cluster_stats = []

n_clusters_total = len(Results_df)
n_clusters_selected = 0
for cluster_id, row in Results_df.iterrows():
    x = np.asarray(row['x'], dtype=float)
    y = np.asarray(row['y'], dtype=float)
    misfit = np.asarray(row['Prob'], dtype=float)

    time = float(row['unixtime'])
    t = np.full(len(x), time)

    # Valid positions
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(misfit)
        & np.isfinite(t)
    )
    n_total = len(x)
    n_valid = np.sum(valid)
    # Percentile calculated over valid positions
    misfit_threshold = np.nanpercentile(
        misfit[valid],
        100 - percentile,
    )
    mask = valid & (misfit <= misfit_threshold)
    n_selected = np.sum(mask)
    if n_selected == 0:
        continue
    n_clusters_selected += 1

    x_all.append(x[mask])
    y_all.append(y[mask])
    t_all.append(t[mask])
    misfit_all.append(misfit[mask])
    cluster_all.append(np.full(n_selected, cluster_id))
    cluster_stats.append({
        "cluster": cluster_id,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_selected": n_selected,
        "misfit_th": misfit_threshold,
        "misfit_min": np.min(misfit[mask]),
        "misfit_max": np.max(misfit[mask]),
    })

# Concatenate selected clusters
x_filt = np.concatenate(x_all)
y_filt = np.concatenate(y_all)
t_filt = np.concatenate(t_all)
misfit_all = np.concatenate(misfit_all)
cluster_all = np.concatenate(cluster_all)

# Absolute and relative time reference
t_start = np.min(t_filt)
t_end = np.max(t_filt)
duration_seconds = int(round(t_end - t_start))
t_start_datetime = dt.datetime.fromtimestamp(
    t_start,
    tz=dt.timezone.utc
)
# Relative time for colorbar
t_filt = t_filt - t_start

# Global minimum and maximum
idx_min = np.argmin(misfit_all)
idx_max = np.argmax(misfit_all)

misfit_min_selected = misfit_all[idx_min]
misfit_max_selected = misfit_all[idx_max]

cluster_min = cluster_all[idx_min]
cluster_max = cluster_all[idx_max]

# Summary
n_selected_per_cluster = cluster_stats[0]["n_selected"]
n_total_per_cluster = cluster_stats[0]["n_valid"]

print("\n" + "="*80)
print("          DAS SPATIAL REPRESENTATION SUMMARY")
print("="*80)

print(
    f"Misfit percentile        : P{percentile} "
    f"(selected points {n_selected_per_cluster}/{n_total_per_cluster})"
)

print(f"Total clusters           : {n_clusters_total}")

print(
    f"Misfit value             : "
    f"[{misfit_min_selected:.6f}, {misfit_max_selected:.6f}] s "
    f"([{misfit_min_selected*1000:.3f}, "
    f"{misfit_max_selected*1000:.3f}] ms) "
    f"| [Cluster {cluster_min}, Cluster {cluster_max}]"
)

print("-"*80)

print(
    f"{'Cluster':>8} "
    f"{'P threshold':>14} "
    f"{'Min misfit [s]':>17} "
    f"{'Max misfit [s]':>17}"
)

print("-"*80)
for stat in cluster_stats:
    print(
        f"{stat['cluster']:>8} "
        f"{stat['misfit_th']:>14.6f} "
        f"{stat['misfit_min']:>17.6f} "
        f"{stat['misfit_max']:>17.6f}"
    )
print("="*80 + "\n")

# Save summary as ASCII text file
summary_file = os.path.join(
    output_results,
    f"7_NoteLoc_P{percentile}.txt",
)

hours = duration_seconds // 3600
minutes = (duration_seconds % 3600) // 60
seconds = duration_seconds % 60

duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

start_str = (
    t_start_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + "Z"
)

with open(summary_file, "w", encoding="ascii") as f:

    f.write("="*80 + "\n")
    f.write("          DAS SPATIAL REPRESENTATION SUMMARY\n")
    f.write("="*80 + "\n")

    f.write(f"Start time               : {start_str}\n")
    f.write(f"Duration                 : {duration_str} (HH:mm:SS)\n")

    f.write("-"*80 + "\n")

    f.write(
        f"Misfit percentile        : P{percentile} "
        f"(selected points {n_selected_per_cluster}/{n_total_per_cluster})\n"
    )

    f.write(f"Total clusters           : {n_clusters_total}\n")

    f.write(
        f"Misfit value             : "
        f"[{misfit_min_selected:.6f}, {misfit_max_selected:.6f}] s "
        f"([{misfit_min_selected*1000:.3f}, "
        f"{misfit_max_selected*1000:.3f}] ms) "
        f"| [Cluster {cluster_min}, Cluster {cluster_max}]\n"
    )

    f.write("-"*80 + "\n")

    f.write(
        f"{'Cluster':>8} "
        f"{'P threshold':>14} "
        f"{'Min misfit [s]':>17} "
        f"{'Max misfit [s]':>17}\n"
    )

    f.write("-"*80 + "\n")

    for stat in cluster_stats:
        f.write(
            f"{stat['cluster']:>8} "
            f"{stat['misfit_th']:>14.6f} "
            f"{stat['misfit_min']:>17.6f} "
            f"{stat['misfit_max']:>17.6f}\n"
        )

    f.write("="*80 + "\n")

print(f"Summary saved: {summary_file}")

# Spatial scatter
sc = ax.scatter(
    x_filt,
    y_filt,
    c=t_filt,
    cmap="viridis",
    s=12,
    alpha=0.8,
    edgecolors="none",
    zorder=3,
)
# DAS cable
ax.plot(
    CSVfile_cable_data["X[m]"],
    CSVfile_cable_data["Y[m]"],
    color="black",
    linestyle="-",
    linewidth=2,
    label="DAS cable",
    zorder=2,
)
# Colorbar
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label(
    "Time [HH:mm:SS]",
    rotation=270,
    labelpad=15,
    fontsize=plot_config.label_size,
)
def format_time(seconds, pos):
    seconds = int(round(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
cbar.ax.yaxis.set_major_formatter(
    FuncFormatter(format_time)
)
cbar.ax.tick_params(labelsize=plot_config.tick_size)

# Labels and title
ax.set_xlabel("X [m]", fontsize=plot_config.label_size)
ax.set_ylabel("Y [m]", fontsize=plot_config.label_size)

# ax.set_title(
#     f"Spatial distribution (lowest {100-percentile}% misfit)",
#     fontsize=plot_config.title_size,
#     fontweight="bold",
# )
ax.set_title(
    f"Spatial distribution (lowest {100-percentile}% misfit)\n"
    # f"Start: {t_start_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} UTC",
    f"Start: {t_start_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z",
    fontsize=plot_config.title_size,
    fontweight="bold",
)

# Axis limits
ax.set_xlim(np.min(x_filt)-1e3, np.max(x_filt)+1e3)
ax.set_ylim(np.min(y_filt)-1e3, np.max(y_filt)+1e3)

# Formatting
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

ax.set_aspect("equal", adjustable="box")
ax.legend(frameon=True)

plt.tight_layout()
# Save figure
fig.savefig(
    os.path.join(
        output_results,
        f"7_NoteLoc_P{percentile}.png",
    ),
    dpi=150,
    bbox_inches="tight",
)
plt.show()
plt.close(fig)

# %% Spatial representation - Misfit stacking between clusters
fig, ax = plt.subplots(figsize=(8, 5))
X_all, Y_all, Z_all = [], [], []
cluster_stats = []
n_clusters_total = len(Results_df)
n_clusters_selected = 0

for cluster_id, row in Results_df.iterrows():
    x = np.asarray(row["x"], dtype=float).ravel()
    y = np.asarray(row["y"], dtype=float).ravel()
    misfit = np.asarray(row["Prob"], dtype=float).ravel()
    dx = row["dx"]
    dy = row["dy"]
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(misfit)
    )
    n_total = len(x)
    n_valid = np.sum(valid)
    if n_valid == 0:
        continue
    # Select lowest misfit percentile
    misfit_threshold = np.nanpercentile(
        misfit[valid],
        100 - percentile
    )
    mask = valid & (misfit <= misfit_threshold)
    if not np.any(mask):
        continue
    # Selected positions
    x_selected = x[mask]
    y_selected = y[mask]
    z_selected = misfit[mask]
    # Snap coordinates to spatial grid
    x_selected = np.round(x_selected / dx) * dx
    y_selected = np.round(y_selected / dy) * dy
    X_all.append(x_selected)
    Y_all.append(y_selected)
    Z_all.append(z_selected)
    n_clusters_selected += 1
    cluster_stats.append({
        "cluster": cluster_id,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_selected": np.sum(mask),
        "misfit_th": misfit_threshold,
        "misfit_min": np.min(z_selected),
        "misfit_max": np.max(z_selected),
    })

X_all = np.concatenate(X_all)
Y_all = np.concatenate(Y_all)
Z_all = np.concatenate(Z_all)

df_grid = pd.DataFrame({
    "x": X_all,
    "y": Y_all,
    "z": Z_all
})

Z_grid = df_grid.pivot_table(
    index="y",
    columns="x",
    values="z",
    aggfunc="sum"
)

X = Z_grid.columns.values
Y = Z_grid.index.values

X_grid, Y_grid = np.meshgrid(X, Y)

Z = Z_grid.values

ax.plot(
    CSVfile_cable_data["X[m]"],
    CSVfile_cable_data["Y[m]"],
    color="black",
    linestyle="-",
    linewidth=2,
    label="DAS cable",
    zorder=3
)

contour = ax.contourf(
    X_grid,
    Y_grid,
    Z,
    levels=20,
    cmap="viridis",
    alpha=0.85,
    zorder=2
)

cbar = fig.colorbar(contour, ax=ax)
cbar.set_label(
    "Cumulative misfit [s]",
    rotation=270,
    labelpad=15,
    fontsize=plot_config.label_size
)
cbar.ax.tick_params(
    labelsize=plot_config.tick_size
)

# Labels and title
ax.set_xlabel("X [m]", fontsize=plot_config.label_size)
ax.set_ylabel("Y [m]", fontsize=plot_config.label_size)

ax.set_title(
    f"Spatial misfit stacking (P{percentile})",
    fontsize=plot_config.title_size,
    fontweight="bold"
)

ax.set_xlim(np.min(x_filt)-1e3, np.max(x_filt)+1e3)
ax.set_ylim(np.min(y_filt)-1e3, np.max(y_filt)+1e3)

# Formatting
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

ax.set_aspect("equal", adjustable="box")
ax.legend(frameon=True)

plt.tight_layout()
# Save figure
fig.savefig(
    os.path.join(
        output_results,
        f"8_NoteLoc_Stacking_P{percentile}.png"
    ),
    dpi=150,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# %%
