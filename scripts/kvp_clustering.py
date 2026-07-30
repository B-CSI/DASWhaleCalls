import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from dataclasses import dataclass

# %% run_dbscan function:
def run_dbscan(x, y, dist_eps, min_pts):
    """
    Run DBSCAN clustering and summarize each detected cluster.

    Parameters
    ----------
    x, y : array-like
        Coordinates of the points.
    dist_eps : float
        DBSCAN epsilon parameter.
    min_pts : int
        Minimum number of samples for a core point.
    
    Returns
    -------
    dbscan_df : pandas.DataFrame
        Summary of detected clusters with columns:
        ['cluster_id', 'Npts', 'x0', 'y0', 'Xsize', 'Ysize'].
    n_noise : int
        Number of points classified as noise.
    """
    X = np.column_stack((x, y))

    dbscan = DBSCAN(eps=dist_eps, min_samples=min_pts)
    labels = dbscan.fit_predict(X)

    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    n_noise = np.count_nonzero(labels == -1)

    rows = []
    cluster_id = -1

    for label in unique_labels:
        if label == -1:
            continue  # Skip noise

        cluster_id += 1
        mask = labels == label

        x_cluster = X[mask, 0]
        y_cluster = X[mask, 1]

        rows.append([
            cluster_id,
            len(x_cluster),
            np.min(x_cluster),
            np.min(y_cluster),
            np.max(x_cluster) - np.min(x_cluster),
            np.max(y_cluster) - np.min(y_cluster)
        ])

    dbscan_df = pd.DataFrame(
        rows,
        columns=['cluster_id', 'Npts', 'x0', 'y0', 'Xsize', 'Ysize']
    )

    return dbscan_df, n_noise

# %% merge_dbscan_clusters function:
def merge_dbscan_clusters(dbscan_df, x_dist=0, y_dist=0):
    """
    Merge nearby DBSCAN clusters based on the distance between their
    bounding boxes.

    Two clusters are merged when the horizontal and vertical gaps between
    their bounding boxes are smaller than the specified thresholds.

    Parameters
    ----------
    dbscan_df : pandas.DataFrame
        DBSCAN cluster summary. Must contain the columns:
        ['x0', 'y0', 'Xsize', 'Ysize'].
    x_dist : float
        Maximum horizontal separation between bounding boxes.
    y_dist : float
        Maximum vertical separation between bounding boxes.
   
    Returns
    -------
    pandas.DataFrame
        Merged cluster summary with columns:
        ['cluster_id', 'xmin', 'ymin', 'xmax', 'ymax',
         'x0', 'y0', 'Xsize', 'Ysize', 'Npts'].
    """

    # ------------------------------------------------------------------
    # Build bounding boxes
    # ------------------------------------------------------------------
    clusters = dbscan_df.copy()

    clusters["xmin"] = clusters["x0"]
    clusters["xmax"] = clusters["x0"] + clusters["Xsize"]
    clusters["ymin"] = clusters["y0"]
    clusters["ymax"] = clusters["y0"] + clusters["Ysize"]

    def rectangle_distance(box1, box2):
        xmin1, xmax1, ymin1, ymax1 = box1
        xmin2, xmax2, ymin2, ymax2 = box2

        dx = max(0, max(xmin2 - xmax1, xmin1 - xmax2))
        dy = max(0, max(ymin2 - ymax1, ymin1 - ymax2))

        return max(dx / x_dist, dy / y_dist)

    boxes = clusters[["xmin", "xmax", "ymin", "ymax"]].to_numpy()

    # ------------------------------------------------------------------
    # First merge using hierarchical clustering
    # ------------------------------------------------------------------
    distances = pdist(boxes, metric=rectangle_distance)
    linkage_matrix = linkage(distances, method="complete")
    clusters["cluster_id"] = fcluster(
        linkage_matrix,
        t=1,
        criterion="distance"
    )

    dbscan_merged_df = (
        clusters
        .groupby("cluster_id", as_index=False)
        .agg({
            "xmin": "min",
            "ymin": "min",
            "xmax": "max",
            "ymax": "max"
        })
    )

    # ------------------------------------------------------------------
    # Second merge to guarantee transitive connections
    # ------------------------------------------------------------------
    parent = np.arange(len(dbscan_merged_df))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(dbscan_merged_df)):
        for j in range(i + 1, len(dbscan_merged_df)):

            dx = max(
                0,
                max(
                    dbscan_merged_df.loc[j, "xmin"] - dbscan_merged_df.loc[i, "xmax"],
                    dbscan_merged_df.loc[i, "xmin"] - dbscan_merged_df.loc[j, "xmax"],
                ),
            )

            dy = max(
                0,
                max(
                    dbscan_merged_df.loc[j, "ymin"] - dbscan_merged_df.loc[i, "ymax"],
                    dbscan_merged_df.loc[i, "ymin"] - dbscan_merged_df.loc[j, "ymax"],
                ),
            )

            if dx <= x_dist and dy <= y_dist:
                union(i, j)

    dbscan_merged_df["cluster_id"] = [find(i) for i in range(len(dbscan_merged_df))]

    dbscan_merged_df = (
        dbscan_merged_df
        .groupby("cluster_id", as_index=False)
        .agg({
            "xmin": "min",
            "ymin": "min",
            "xmax": "max",
            "ymax": "max"
        })
    )

    # ------------------------------------------------------------------
    # Rebuild output table
    # ------------------------------------------------------------------
    dbscan_merged_df["x0"] = dbscan_merged_df["xmin"]
    dbscan_merged_df["y0"] = dbscan_merged_df["ymin"]
    dbscan_merged_df["Xsize"] = dbscan_merged_df["xmax"] - dbscan_merged_df["xmin"]
    dbscan_merged_df["Ysize"] = dbscan_merged_df["ymax"] - dbscan_merged_df["ymin"]

    dbscan_merged_df = dbscan_merged_df.sort_values("x0").reset_index(drop=True)
    dbscan_merged_df["cluster_id"] = np.arange(len(dbscan_merged_df))

    return dbscan_merged_df

# %% Hyperbolic fit:
@dataclass
class FitResult:
    """All parameters and quality measures from fitting one hyperbola to one cluster.

    After the fitting routine finishes, every piece of information about the
    best curve it found is stored here so it can be used for plotting and
    printed to the terminal.
    """
    cluster_id: int             # which cluster this result belongs to
    y0: float                   # channel position at the apex of the hyperbola
    t0: float                   # arrival time at the apex, in elapsed seconds
    vapp: float                 # apparent velocity along the fibre
    rmse: float                 # average time error of the fit (seconds); lower is better
    r2: float
    mse: float
    n_points: int               # number of picks used to produce this fit
    n_removed: int = 0          # picks discarded as outliers before the second fit
    initial_rmse: float | None = None  # RMSE from the first pass, before refitting

def x_fit_evaluation(x_raw,x_pred):
    x_residuals = x_raw - x_pred
    mse = np.mean(x_residuals**2)
    ss_res = np.sum(x_residuals**2)
    ss_tot = np.sum((x_raw - np.mean(x_raw))**2)
    r2 = (1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0)
    return x_residuals, r2, mse

def fit_cluster(
    cluster_data: pd.DataFrame,
    min_points: int,
    grid_size: int,
    early_time_weight: float,
    early_time_weight_power: float,
    refit: bool,
    refit_time_threshold: float,
    refit_no_early_weight: bool = False,
) -> FitResult | None:
    """Fit the NMO hyperbola  t² = t₀² + (y - y₀)² / v²  to one cluster.

    y₀ enters the model nonlinearly, so we fix it on a dense grid and solve
    the linear sub-problem for (t₀², 1/v²) at each trial value, keeping the
    best result. Optionally a second pass is run after removing picks whose
    time residual exceeds *refit_time_threshold*.

    Returns a FitResult, or None if the cluster has too few usable picks.
    """
    # Validate the tuning parameters up-front so we fail immediately with a
    # clear message rather than producing a silent bad result later.
    if early_time_weight < 1.0:
        raise ValueError("--early-time-weight must be >= 1.")
    if early_time_weight_power <= 0.0:
        raise ValueError("--early-time-weight-power must be > 0.")
    if refit_time_threshold <= 0.0:
        raise ValueError("--refit-time-threshold must be > 0.")

    if len(cluster_data) < min_points:
        return None
    # Extract the arrival times and channel positions as plain number arrays
    cluster_id_data = int(cluster_data["cluster_id"].iloc[0])
    t_arr = cluster_data["x"].to_numpy(dtype=float)  # arrival time of each pick (seconds)
    y_arr = cluster_data["Channel"].to_numpy(dtype=float)        # channel position of each pick
    # Remove any picks that have missing or infinite values in either column
    finite = np.isfinite(t_arr) & np.isfinite(y_arr)
    t_arr = t_arr[finite]
    y_arr = y_arr[finite]
    if len(t_arr) < min_points or np.min(y_arr) == np.max(y_arr):
        return None
    def fit_once(cluster_id_data, times: np.ndarray, values: np.ndarray, etw: float | None = None) -> FitResult | None:
        """Fit the hyperbola to the given picks and return the best FitResult.

        'times'  — arrival time of each pick (seconds)
        'values' — channel position of each pick
        'etw'    — override for early_time_weight; None means use the outer value
        """
        # Can't fit a curve with too few points or if all picks are on the same channel
        if len(times) < min_points or np.min(values) == np.max(values):
            return None
        # Build a grid of candidate apex positions (y0) spanning slightly beyond
        # the data range so the apex can sit just outside the observed channels
        y_span = float(np.max(values) - np.min(values))
        y_margin = max(1.0, 0.25 * y_span)  # at least 1 channel unit of margin
        y0_grid = np.linspace(float(np.min(values) - y_margin), float(np.max(values) + y_margin), grid_size)
        # Square the arrival times once; the model is linear in t² not in t
        t2 = times**2
        # Compute a per-pick weight: earlier picks get a higher weight when etw > 1
        effective_etw = early_time_weight if etw is None else etw
        t_span = float(np.max(times) - np.min(times))
        if t_span > 0.0 and effective_etw > 1.0:
            # Ramp weight from `effective_etw` at the earliest pick down to
            # 1.0 at the latest, with `early_time_weight_power` controlling the
            # decay shape (1 = linear, >1 = convex, <1 = concave).
            normalized_age = (times - np.min(times)) / t_span
            weights = 1.0 + (effective_etw - 1.0) * (1.0 - normalized_age) ** early_time_weight_power
        else:
            weights = np.ones_like(times)  # equal weight for every pick
        # Weighted least squares is equivalent to OLS on sqrt(w)-scaled data.
        sqrt_weights = np.sqrt(weights)
        best: FitResult | None = None
        best_rmse = np.inf
        # Try every candidate apex position and keep the one that gives the smallest error
        for y0 in y0_grid:
            # Squared distance from each pick to the candidate apex along the channel axis
            radius2 = (values - y0) ** 2
            # Set up the linear system:  t² ≈ t0² · 1  +  (1/v²) · radius²
            # 'design' has two columns: a column of ones (for t0²) and radius² (for 1/v²)
            design = np.column_stack([np.ones_like(radius2), radius2])
            # Apply the sqrt-weight scaling to convert weighted to ordinary least squares
            weighted_design = design * sqrt_weights[:, None]
            weighted_t2 = t2 * sqrt_weights
            try:
                # Solve for the two unknowns [t0², 1/v²] using least squares
                coeffs, _, _, _ = np.linalg.lstsq(weighted_design, weighted_t2, rcond=None)
            except np.linalg.LinAlgError:
                continue  # skip this y0 if the numerical solver fails
            t0_squared = float(coeffs[0])
            inv_v_squared = float(coeffs[1])
            # Reject non-physical solutions: t₀ and v must both be real and positive.
            if t0_squared < 0.0 or inv_v_squared <= 0.0:
                continue
            # Evaluate the fitted curve at each pick's channel position
            t_fit = np.sqrt(t0_squared + inv_v_squared * radius2)
            # RMSE: root-mean-square error between predicted and observed arrival times
            rmse = float(np.sqrt(np.average((times - t_fit) ** 2, weights=weights)))
            if rmse < best_rmse and y0>0:
                best_rmse = rmse
                _, r2, mse = x_fit_evaluation(times,t_fit)
                best = FitResult(
                    cluster_id = cluster_id_data,
                    y0=float(y0),
                    t0=float(np.sqrt(t0_squared)),          # recover t0 from t0²
                    vapp=float(1.0 / np.sqrt(inv_v_squared)),  # recover v  from 1/v²
                    rmse=rmse,
                    r2=float(r2),
                    mse=float(mse),
                    n_points=len(times),
                )
        return best
    # --- First pass: fit using all picks in the cluster ---
    first_fit = fit_once(cluster_id_data, t_arr, y_arr)
    if first_fit is None or not refit:
        return first_fit
    first_fit.initial_rmse = first_fit.rmse
    # Second pass: drop picks whose time residual exceeds the threshold, then
    # refit the cleaned subset. If nothing was removed or too few points remain,
    # return the first fit unchanged.
    first_t_fit = np.sqrt(first_fit.t0**2 + ((y_arr - first_fit.y0) ** 2) / (first_fit.vapp**2))
    keep_mask = np.abs(t_arr - first_t_fit) <= refit_time_threshold
    n_removed = int(len(t_arr) - np.count_nonzero(keep_mask))
    if n_removed == 0 or np.count_nonzero(keep_mask) < min_points:
        return first_fit
    # --- Second pass: fit using only the inlier picks ---
    second_fit = fit_once(cluster_id_data, t_arr[keep_mask], y_arr[keep_mask], etw=1.0 if refit_no_early_weight else None)
    if second_fit is None:
        return first_fit
    second_fit.n_removed = n_removed
    second_fit.initial_rmse = first_fit.rmse
    return second_fit

# %% find_crossing_points function:
def find_crossing_points(x_pred, x_data, y_data, tol=0.5, x_lim=None, y_lim=None):
    x_data = np.array(x_data).ravel()
    y_data = np.array(y_data).ravel()
    x_pred = np.array(x_pred).ravel()
    # Máscara por tolerancia en X
    mask_cross = np.abs(x_data - x_pred) < tol
    # Aplicar límites si se proporcionan
    if x_lim is not None:
        mask_cross &= (x_data >= x_lim[0]) & (x_data <= x_lim[1])
    if y_lim is not None:
        mask_cross &= (y_data >= y_lim[0]) & (y_data <= y_lim[1])
    x_cross = x_data[mask_cross]
    y_cross = y_data[mask_cross]
    return x_cross, y_cross