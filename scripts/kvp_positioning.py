# %% Libraries:
# %% Libraries
import numpy as np
import matplotlib.pyplot as plt


# %% gridCreator2Dsquare function: 
def gridCreator2Dsquare(central_point, dx, dy, minX, maxX, minY, maxY, plotter=False):
    """
    Generate 2D grid points within a square region centered at central_point.

    Parameters:
    - central_point (np.ndarray): Array-like [x, y] of the center.
    - dx (float): Step size in the X direction.
    - dy (float): Step size in the Y direction.
    - minX, maxX (float): X-coordinate range for the grid.
    - minY, maxY (float): Y-coordinate range for the grid.
    - plotter (bool): If True, plot the generated points and central point.

    Returns:
    - points (np.ndarray): Array of shape (N, 2) with generated grid points.
    
    Application:
    points = gridCreator2Dsquare(central_point, dx, dy, minX, maxX, minY, maxY, plotter=False)
    
    Created/Last modified: 2025-09-05
    """
    # Create grid within the defined X and Y range
    x_vals = np.arange(minX, maxX + dx, dx)
    y_vals = np.arange(minY, maxY + dy, dy)

    # Generate the grid of points
    points = []
    for x in x_vals:
        for y in y_vals:
            # Translate to the central point
            point = np.array([x, y]) + central_point[:2]
            points.append(point)
    
    points = np.array(points)

    if plotter:
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot center
        ax.scatter(central_point[0], central_point[1], color='red', marker='x', s=100, label='Central Point')

        # Plot generated points
        ax.scatter(points[:, 0], points[:, 1], color='gray', alpha=0.6, label='Generated Points', s=30)

        # Set axis labels
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title("gridCreator2Dsquare(): Points in Square Grid Around Central Point")
        ax.legend()
        ax.grid(True)

        # Set limits for the plot to visualize the grid
        ax.set_xlim(minX + central_point[0], maxX + central_point[0])
        ax.set_ylim(minY + central_point[1], maxY + central_point[1])

        plt.tight_layout()
        plt.show()

    return points

# %% localization_misfit function():
def localization_misfit(
    observed_times: np.ndarray,
    calculated_times: np.ndarray,
    fs: float,
    n_sigma: float,
) -> tuple[float, float, np.ndarray]:
    """
    Compute the localization misfit between observed and calculated
    relative arrival times.

    The observed arrival times are aligned to the calculated arrival times by
    applying a uniform temporal shift. For each tested shift, outliers are
    rejected using the median absolute deviation (MAD), and the misfit is
    computed as the sum of the absolute residuals of the retained
    observations.

    The optimal shift is the one that minimizes the localization misfit.

    Parameters
    ----------
    observed_times : ndarray
        Observed relative arrival times (s).

    calculated_times : ndarray
        Calculated relative arrival times (s).

    fs : float
        Sampling frequency (Hz). The temporal shift is evaluated at intervals
        of 1 / fs.

    n_sigma : float
        Outlier rejection threshold expressed as multiples of the robust
        standard deviation estimated from the MAD.

    Returns
    -------
    misfit : float
        Localization misfit,

            ε = Σ |T_calc - T_obs|,

        computed using only the retained observations.

    best_shift : float
        Temporal shift (s) applied to the observed arrival times.

    inlier_indices : ndarray
        Indices of the observations retained after MAD filtering.
    """

    observed_times = np.asarray(observed_times, dtype=float)
    calculated_times = np.asarray(calculated_times, dtype=float)

    # Search interval for the temporal shift
    shift_min = np.min(observed_times - calculated_times)
    shift_max = np.max(observed_times - calculated_times)

    shift_step = 1.0 / fs
    shifts = np.arange(shift_min, shift_max + shift_step, shift_step)

    best_misfit = np.inf
    best_shift = 0.0
    best_inliers = np.arange(len(observed_times))

    for shift in shifts:
        # Apply candidate temporal shift
        shifted_obs = observed_times - shift
        residuals = calculated_times - shifted_obs
        # Robust outlier rejection using the MAD
        median = np.median(residuals)
        mad = np.median(np.abs(residuals - median))
        robust_std = 1.4826 * mad
        lower = median - n_sigma * robust_std
        upper = median + n_sigma * robust_std
        inliers = np.where((residuals >= lower) & (residuals <= upper))[0]
        if len(inliers) == 0:
            continue
        misfit = np.sum(np.abs(residuals[inliers]))
        if misfit < best_misfit:
            best_misfit = misfit
            best_shift = shift
            best_inliers = inliers

    return best_misfit, best_shift, best_inliers