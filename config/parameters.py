# %% Libs
import os
from dataclasses import dataclass

# %% Defined params
# DAS acquisition (reference values; actual values read from HDF5)
dist_ch = 10                    # [m]
fs = 100 # [Hz]
FontSize = 16
output_results = "./results"    # Output
# KVP CONFIGURATION [used by 0_KVP_application_demo.py]:
fmin_filt = 19.0                # Bandpass filtering min [Hz]
fmax_filt = 23.0                # Bandpass filtering max [Hz]
nfilt = 6                       # Bandpass filter order
wavelet = "morlet"              # Wavelet configuration
freqmax = 25.0                  # Maximum central frequency [Hz]
octaves = 1                     # Number of frequency octaves
voices = 10                     # Number of voices per octave
cf_cycles = 150.0               # Window length in cycles 
jmp_cycles = 2.0                # Jump-window length in cycles
jump = 2.0                      # Jump detection threshold
mingap = 1.0                    # Minimum temporal gap between consecutive 
nbands = 2                      # Number of frequency bands required for grouping
w0 = 2 * 5.3364                 # Dimensionless central frequency parameter
# DBSCAN PARAMETERS [used by 1_KVP_clustering_fitting_demo.py]
c_sound = 1500                  # [m/s]
eps_m = 500                     # [m]
min_pts = 5
# CLUSTER MERGING PARAMETERS [used by 1_KVP_clustering_fitting_demo.py]
Xdist = 2                       # [s]
Ydist = 5000                    # [m]
marginBB_x = 1                  # [s]
marginBB_y = 1000               # [m]
# HYPERBOLIC FIT PARAMETERS [used by 1_KVP_clustering_fitting_demo.py]:
refit_bool = True
min_fit_points = 5
fit_grid_size = 500
early_time_weight = 20
early_time_weight_power = 10
refit_time_threshold = 0.5      # [s]
# POSITIONING PARAMETERS [used by 3_Note_localization_demo]:
dx_res = 25                     # [m]
dy_res = 25                     # [m]
minX = -3e3                     # [m]
maxX = 3e3                      # [m]
minY = -3e3                     # [m]
maxY = 3e3                      # [m]
n_sigma_terr = 1
minNpts_HyperBranch = 3

# %% Derivated params and others
# PLOTTING PARAMETERS
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