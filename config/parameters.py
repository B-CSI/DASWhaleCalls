# %% Libs
import os
from dataclasses import dataclass

# %% Defined params
# gral. params:
dist_ch = 10        # [m]
fs = 100 # [Hz]
FontSize = 16
output_results = "./results"
# DBSCAN:
c_sound = 1500        # [m/s]
eps_m = 500         # [m]
min_pts = 5
# Cluster merged:
Xdist = 2           # [s]
Ydist = 5000        # [m]
marginBB_x = 1      # [s]
marginBB_y = 1000   # [m]
# Hyperbolic fit:
refit_bool = True
min_fit_points = 5
fit_grid_size = 500
early_time_weight = 20
early_time_weight_power = 10
refit_time_threshold = 0.5  # [s]
# Positioning note:
dx_res = 25         # [m]
dy_res = 25         # [m]
minX = -3e3         # [m]
maxX = 3e3          # [m]
minY = -3e3         # [m]
maxY = 3e3          # [m]
n_sigma_terr = 1
minNpts_HyperBranch = 3

# %% Derivated params and others
TITLE_SIZE = FontSize 
LABEL_SIZE = FontSize - 4
TICK_SIZE = LABEL_SIZE - 1
LEGEND_SIZE = LABEL_SIZE - 2
LINEWIDTH = 2.0
GRID_WIDTH = 0.3
GRID_ALPHA = 0.6

@dataclass(frozen=True)
class PlotConfig:
    """Configuration parameters for matplotlib figures."""
    title_size: int = TITLE_SIZE
    label_size: int = LABEL_SIZE
    tick_size: int = TICK_SIZE
    legend_size: int = LEGEND_SIZE

    linewidth: float = LINEWIDTH
    grid_width: float = GRID_WIDTH
    grid_alpha: float = GRID_ALPHA

os.makedirs(output_results, exist_ok=True)
# %%
