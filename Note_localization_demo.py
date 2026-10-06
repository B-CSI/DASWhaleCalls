# %% Imports
# Standard library
import os
import numpy as np 

# Third-party libraries
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from config.parameters import *

SCRIPT_NAME = "Note_localization_demo.py"
# %% Load KVP picks
csv_path = os.path.join(".","data_example","2024_01_13_07h57m34s_HDAS_SAFE_DASWhaleCalls_example_selectedKVPpicks.csv")
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
    f"{picks_per_cluster.min()} - {picks_per_cluster.max()} "
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
from scripts.kvp_positioning import gridCreator2Dsquare, localization_misfit

Results_df_columns = (
    ['cluster_id', 'unixtime', 'Npicks', 'Prob', 'x', 'y', 'dx', 'dy', 'Ax', 'Ay']
)
Results_df = [] # one per cluster_id

Nx = int(2 * maxX / dx_res) + 1
Ny = int(2 * maxY / dy_res) + 1
Ntotal = Nx * Ny

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
    
    Prob = np.empty(len(E_calc_xy_m))
    for i in range(len(E_calc_xy_m)):
        misfit, _, _ = localization_misfit(T_obs_s,T_calc_s[:, i],fs=fs,n_sigma=n_sigma_terr)
        Prob[i] = 1.0 / (1.0 + misfit**2)

    best_idx = np.where(Prob == np.max(Prob))[0][0]
    best_point = E_calc_xy_m[best_idx, :]

    fig = plt.figure(figsize=(8,6))
    sc = plt.scatter(E_calc_xy_m[:,0],E_calc_xy_m[:,1],c=Prob,cmap='viridis',marker='s')
    plt.plot(CSVfile_cable_data['X[m]'], CSVfile_cable_data['Y[m]'],
                color='red', linestyle='-', linewidth=2,
                label='DAS interrogation')
    plt.plot(best_point[0], best_point[1],color='red', marker='*', linestyle='',
                label=f'Best: {Prob[best_idx]:.3f}')
    # plt.plot(central_point_m[0]+maxX, central_point_m[1]+maxY,color='lime', marker='o', linestyle='')
    # plt.plot(central_point_m[0]+minX, central_point_m[1]+minY,color='lime', marker='o', linestyle='')
    # plt.plot(central_point_m[0]+maxX, central_point_m[1]+minY,color='lime', marker='o', linestyle='')
    # plt.plot(central_point_m[0]+minX, central_point_m[1]+maxY,color='lime', marker='o', linestyle='')
    cbar = plt.colorbar(sc)
    cbar.set_label('Probability',rotation=270,labelpad=15)
    plt.xlim(np.min(E_calc_xy_m[:, 0]),np.max(E_calc_xy_m[:, 0]))
    plt.ylim(np.min(E_calc_xy_m[:, 1]),np.max(E_calc_xy_m[:, 1]))
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.grid(True, alpha=0.3)
    # plt.legend(ncols=3)
    plt.title(f"Cluster: {cluster}\nGrid: {len(E_calc_xy_m)} samples; dx: {dx_res}m; dy: {dy_res}m\n[$\pm${np.abs(int(maxX))},$\pm${np.abs(int(maxY))}]m")
    # plt.axis('equal')
    plt.tight_layout()
    # plt.show()
    fig.savefig(
        os.path.join(output_results, f"DASpos_{int(cluster)}.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    plt.close('all')

    E_calc_x_m, E_calc_y_m = E_calc_xy_m[:,0], E_calc_xy_m[:,1]
    best_results = {
        'cluster_id': cluster,
        'unixtime': np.min(KVPpicks_cluster['unixtime']),
        'Npicks': len(KVPpicks_cluster),
        'Prob': Prob.tolist(),
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
        CSVbase = os.path.basename(CSVfile).split('.')[0]
        CSVname = f"{CSVbase}_DASpos_cluster{int(cluster)}.csv"
        metadata = [
            f"# Input: {os.path.basename(CSVfile)}",
            f"# Output: {CSVname}",
            "# ----------------------------------------",
            "# Local tangent plane projection, Units: meters",
            f"# Cluster: {int(cluster)}/{len(unique_clusters_raw)}"
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
