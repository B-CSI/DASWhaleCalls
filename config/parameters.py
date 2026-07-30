# %% Libs
import os

# %% Defined params
# gral. params:
dist_ch = 10        # [m]
FontSize = 16
output_results = "./results"
# DBSCAN:
c_ref = 1500        # [m/s]
eps_m = 500         # [m]
min_pts = 5
# Cluster merged:
Xdist = 2           # [s]
Ydist = 5000        # [m]
marginBB_x = 1      # [s]
marginBB_y = 1000   # [m]
# Hyperbolic fit:
min_fit_points = 5
fit_grid_size = 500
early_time_weight = 20
early_time_weight_power = 10
refit_time_threshold = 0.3

# %% Derivated params and others
TITLE_SIZE = FontSize 
LABEL_SIZE = FontSize - 4
TICK_SIZE = LABEL_SIZE - 1
LEGEND_SIZE = LABEL_SIZE - 2
LINEWIDTH = 2.0
GRID_WIDTH = 0.3
GRID_ALPHA = 0.6

os.makedirs(output_results, exist_ok=True)