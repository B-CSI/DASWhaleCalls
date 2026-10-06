# %% Libraries
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from dataclasses import dataclass
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib as mpl
from dataclasses import asdict

from config.parameters import PlotConfig

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

    dbscan = DBSCAN(
        eps=dist_eps,
        min_samples=min_pts
    )

    labels = dbscan.fit_predict(X)

    unique_labels = set(labels)

    n_clusters = len(unique_labels - {-1})
    n_noise = np.count_nonzero(labels == -1)

    rows = []

    for label in unique_labels:

        if label == -1:
            continue  # Skip noise

        mask = labels == label

        x_cluster = X[mask, 0]
        y_cluster = X[mask, 1]

        rows.append([
            label,
            len(x_cluster),
            np.min(x_cluster),
            np.min(y_cluster),
            np.max(x_cluster) - np.min(x_cluster),
            np.max(y_cluster) - np.min(y_cluster)
        ])

    dbscan_df = pd.DataFrame(
        rows,
        columns=[
            'cluster_id',
            'Npts',
            'x0',
            'y0',
            'Xsize',
            'Ysize'
        ]
    )

    # ------------------------------------------------------------
    # Sort chronologically and reassign cluster IDs
    # ------------------------------------------------------------

    dbscan_df = (
        dbscan_df
        .sort_values("x0")
        .reset_index(drop=True)
    )

    dbscan_df["cluster_id"] = np.arange(
        len(dbscan_df)
    )

    return dbscan_df, n_noise

# %% merge_dbscan_clusters function:
def merge_dbscan_clusters(
    dbscan_df,
    x_dist=2.0,       # [s]
    y_dist=5000.0,    # [m]
):
    """
    Merge DBSCAN bounding boxes.

    X = time [s]
    Y = distance/channel [m]

    Two DBSCAN bounding boxes are connected when the gap between
    their temporal intervals is <= x_dist AND the gap between
    their spatial intervals is <= y_dist.

    Connected components are then merged transitively.
    """

    df = dbscan_df.copy().reset_index(drop=True)

    # Original bounding boxes
    df["xmin"] = df["x0"]
    df["xmax"] = df["x0"] + df["Xsize"]

    df["ymin"] = df["y0"]
    df["ymax"] = df["y0"] + df["Ysize"]

    # ------------------------------------------------------------
    # Union-Find
    # ------------------------------------------------------------

    parent = np.arange(len(df))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri = find(i)
        rj = find(j)

        if ri != rj:
            parent[rj] = ri

    # ------------------------------------------------------------
    # Pairwise BB comparison
    # ------------------------------------------------------------

    for i in range(len(df)):

        for j in range(i + 1, len(df)):

            # Temporal gap [s]
            dx = max(
                0.0,
                df.loc[j, "xmin"] - df.loc[i, "xmax"],
                df.loc[i, "xmin"] - df.loc[j, "xmax"],
            )

            # Spatial gap [m]
            dy = max(
                0.0,
                df.loc[j, "ymin"] - df.loc[i, "ymax"],
                df.loc[i, "ymin"] - df.loc[j, "ymax"],
            )

            # Connect boxes
            if dx <= x_dist and dy <= y_dist:
                union(i, j)

    # ------------------------------------------------------------
    # Connected components
    # ------------------------------------------------------------

    df["_merged_id"] = [
        find(i)
        for i in range(len(df))
    ]

    # ------------------------------------------------------------
    # Merge boxes
    # ------------------------------------------------------------

    merged = (
        df
        .groupby("_merged_id")
        .agg(
            Npts=("Npts", "sum"),
            x0=("xmin", "min"),
            xmax=("xmax", "max"),
            y0=("ymin", "min"),
            ymax=("ymax", "max"),
            source_clusters=(
                "cluster_id",
                lambda s: sorted(s.tolist())
            ),
        )
        .reset_index(drop=True)
    )

    merged["Xsize"] = (
        merged["xmax"] -
        merged["x0"]
    )

    merged["Ysize"] = (
        merged["ymax"] -
        merged["y0"]
    )

    # ------------------------------------------------------------
    # Sort chronologically
    # ------------------------------------------------------------

    merged = (
        merged
        .sort_values("x0")
        .reset_index(drop=True)
    )

    # New merged ID
    merged["cluster_id"] = np.arange(len(merged))

    # ------------------------------------------------------------
    # Final format
    # ------------------------------------------------------------

    merged = merged[
        [
            "cluster_id",
            "Npts",
            "source_clusters",
            "x0",
            "y0",
            "Xsize",
            "Ysize",
        ]
    ]

    return merged

# %% Hyperbolic fit (using physical distance in metres):
@dataclass
class FitResult:
    """Parameters and quality measures from fitting one NMO hyperbola.

    The fit is performed entirely in physical units:

        x  -> time [s]
        y  -> distance along the fibre [m]

    Hyperbolic model:

        t² = t₀² + (y - y₀)² / v²

    Therefore:

        y0   -> apex position [m]
        t0   -> arrival time at apex [s]
        vapp -> apparent velocity [m/s]
        rmse -> time error [s]
        mse  -> time error squared [s²]
    """

    cluster_id: int

    # Position of the hyperbola apex [m]
    y0: float

    # Arrival time at the apex [s]
    t0: float

    # Apparent velocity [m/s]
    vapp: float

    # Fit quality
    rmse: float
    r2: float
    mse: float

    # Number of picks used in the final fit
    n_points: int

    # Number of picks removed during refit
    n_removed: int = 0

    # RMSE before refitting
    initial_rmse: float | None = None


def x_fit_evaluation(
    x_raw,
    x_pred,
):
    """
    Evaluate residuals and goodness of fit.

    Parameters
    ----------
    x_raw : array-like
        Observed arrival times [s].

    x_pred : array-like
        Predicted arrival times [s].

    Returns
    -------
    residuals : ndarray
        Time residuals [s].

    r2 : float
        Coefficient of determination.

    mse : float
        Mean squared error [s²].
    """

    x_raw = np.asarray(
        x_raw,
        dtype=float,
    )

    x_pred = np.asarray(
        x_pred,
        dtype=float,
    )

    x_residuals = (
        x_raw - x_pred
    )

    mse = float(
        np.mean(
            x_residuals ** 2
        )
    )

    ss_res = np.sum(
        x_residuals ** 2
    )

    ss_tot = np.sum(
        (
            x_raw
            - np.mean(x_raw)
        ) ** 2
    )

    if ss_tot != 0.0:
        r2 = (
            1.0
            - ss_res / ss_tot
        )
    else:
        r2 = 0.0

    return (
        x_residuals,
        float(r2),
        mse,
    )


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
    """
    Fit an NMO hyperbola to one cluster using PHYSICAL DISTANCE.

    The fitted model is:

        t² = t₀² + (y - y₀)² / v²

    where:

        t  = arrival time [s]
        y  = distance along fibre [m]
        t0 = apex arrival time [s]
        y0 = apex position [m]
        v  = apparent velocity [m/s]

    IMPORTANT
    ---------
    This function does NOT use the HDF5 Channel index.

    The input DataFrame must contain:

        cluster_id
        x
        y

    with:

        x = time [s]
        y = distance [m]

    The function therefore works independently of how the original
    DAS channels were numbered or cropped.
    """

    # ========================================================
    # Validate parameters
    # ========================================================

    if early_time_weight < 1.0:
        raise ValueError(
            "early_time_weight must be >= 1."
        )

    if early_time_weight_power <= 0.0:
        raise ValueError(
            "early_time_weight_power must be > 0."
        )

    if refit_time_threshold <= 0.0:
        raise ValueError(
            "refit_time_threshold must be > 0."
        )

    # ========================================================
    # Minimum number of points
    # ========================================================

    if len(cluster_data) < min_points:
        return None

    # ========================================================
    # Cluster ID
    # ========================================================

    cluster_id_data = int(
        cluster_data[
            "cluster_id"
        ].iloc[0]
    )

    # ========================================================
    # Extract PHYSICAL coordinates
    # ========================================================

    # x = arrival time [s]
    t_arr = cluster_data[
        "x"
    ].to_numpy(
        dtype=float
    )

    # y = physical distance [m]
    y_arr = cluster_data[
        "y"
    ].to_numpy(
        dtype=float
    )

    # ========================================================
    # Remove invalid values
    # ========================================================

    finite = (
        np.isfinite(t_arr)
        & np.isfinite(y_arr)
    )

    t_arr = t_arr[finite]
    y_arr = y_arr[finite]

    if len(t_arr) < min_points:
        return None

    # All picks cannot be at exactly the same distance
    if np.min(y_arr) == np.max(y_arr):
        return None

    # ========================================================
    # Fit function
    # ========================================================

    def fit_once(
        cluster_id_data,
        times: np.ndarray,
        distances: np.ndarray,
        etw: float | None = None,
    ) -> FitResult | None:
        """
        Perform one hyperbolic fit.

        Parameters
        ----------
        times :
            Arrival times [s].

        distances :
            Physical positions along the fibre [m].

        etw :
            Optional override for early-time weighting.
        """

        # ----------------------------------------------------
        # Basic validation
        # ----------------------------------------------------

        if len(times) < min_points:
            return None

        if (
            np.min(distances)
            == np.max(distances)
        ):
            return None

        # ----------------------------------------------------
        # Candidate apex positions
        #
        # Everything here is in METRES.
        # ----------------------------------------------------

        y_span = float(
            np.max(distances)
            - np.min(distances)
        )

        # Allow the apex to lie slightly outside
        # the observed cluster.
        #
        # Minimum margin = 1 m
        y_margin = max(
            1.0,
            0.25 * y_span,
        )

        y0_grid = np.linspace(
            float(
                np.min(distances)
                - y_margin
            ),
            float(
                np.max(distances)
                + y_margin
            ),
            grid_size,
        )

        # ----------------------------------------------------
        # Square arrival times
        #
        # Model:
        #
        # t² = t0² + (y-y0)² / v²
        #
        # ----------------------------------------------------

        t2 = times ** 2

        # ----------------------------------------------------
        # Early-time weighting
        # ----------------------------------------------------

        effective_etw = (
            early_time_weight
            if etw is None
            else etw
        )

        t_span = float(
            np.max(times)
            - np.min(times)
        )

        if (
            t_span > 0.0
            and effective_etw > 1.0
        ):

            normalized_age = (
                times
                - np.min(times)
            ) / t_span

            weights = (
                1.0
                + (
                    effective_etw
                    - 1.0
                )
                * (
                    1.0
                    - normalized_age
                )
                ** early_time_weight_power
            )

        else:

            weights = np.ones_like(
                times
            )

        # ----------------------------------------------------
        # Weighted least squares
        # ----------------------------------------------------

        sqrt_weights = np.sqrt(
            weights
        )

        best = None
        best_rmse = np.inf

        # ====================================================
        # Search for best apex position
        # ====================================================

        for y0 in y0_grid:

            # ------------------------------------------------
            # Physical distance from candidate apex [m]
            # ------------------------------------------------

            radius2 = (
                distances - y0
            ) ** 2

            # ------------------------------------------------
            # Linearized model:
            #
            # t² = t0² + (1/v²) * radius²
            #
            # Unknowns:
            #
            # t0²
            # 1/v²
            # ------------------------------------------------

            design = np.column_stack([
                np.ones_like(
                    radius2
                ),
                radius2,
            ])

            weighted_design = (
                design
                * sqrt_weights[:, None]
            )

            weighted_t2 = (
                t2
                * sqrt_weights
            )

            try:

                coeffs, _, _, _ = (
                    np.linalg.lstsq(
                        weighted_design,
                        weighted_t2,
                        rcond=None,
                    )
                )

            except np.linalg.LinAlgError:

                continue

            # ------------------------------------------------
            # Recover physical parameters
            # ------------------------------------------------

            t0_squared = float(
                coeffs[0]
            )

            inv_v_squared = float(
                coeffs[1]
            )

            # ------------------------------------------------
            # Reject non-physical solutions
            # ------------------------------------------------

            if (
                t0_squared < 0.0
                or inv_v_squared <= 0.0
            ):
                continue

            # ------------------------------------------------
            # Predicted arrival times
            # ------------------------------------------------

            t_fit = np.sqrt(
                t0_squared
                + inv_v_squared
                * radius2
            )

            # ------------------------------------------------
            # RMSE [s]
            # ------------------------------------------------

            rmse = float(
                np.sqrt(
                    np.average(
                        (
                            times
                            - t_fit
                        ) ** 2,
                        weights=weights,
                    )
                )
            )

            # ------------------------------------------------
            # Keep best solution
            # ------------------------------------------------

            if rmse < best_rmse:

                best_rmse = rmse

                _, r2, mse = (
                    x_fit_evaluation(
                        times,
                        t_fit,
                    )
                )

                best = FitResult(

                    cluster_id=(
                        cluster_id_data
                    ),

                    # Physical distance [m]
                    y0=float(y0),

                    # Time [s]
                    t0=float(
                        np.sqrt(
                            t0_squared
                        )
                    ),

                    # Velocity [m/s]
                    vapp=float(
                        1.0
                        / np.sqrt(
                            inv_v_squared
                        )
                    ),

                    rmse=float(
                        rmse
                    ),

                    r2=float(
                        r2
                    ),

                    mse=float(
                        mse
                    ),

                    n_points=len(
                        times
                    ),
                )

        return best

    # ========================================================
    # FIRST FIT
    # ========================================================

    first_fit = fit_once(
        cluster_id_data,
        t_arr,
        y_arr,
    )

    if (
        first_fit is None
        or not refit
    ):
        return first_fit

    first_fit.initial_rmse = (
        first_fit.rmse
    )

    # ========================================================
    # Calculate first-fit prediction
    #
    # IMPORTANT:
    # y_arr and y0 are both in metres.
    # vapp is m/s.
    # ========================================================

    first_t_fit = np.sqrt(
        first_fit.t0 ** 2
        + (
            (
                y_arr
                - first_fit.y0
            ) ** 2
            / first_fit.vapp ** 2
        )
    )

    # ========================================================
    # Remove temporal outliers
    # ========================================================

    keep_mask = (
        np.abs(
            t_arr
            - first_t_fit
        )
        <= refit_time_threshold
    )

    n_removed = int(
        len(t_arr)
        - np.count_nonzero(
            keep_mask
        )
    )

    # Nothing removed or too few points remain
    if (
        n_removed == 0
        or np.count_nonzero(
            keep_mask
        ) < min_points
    ):

        return first_fit

    # ========================================================
    # SECOND FIT
    # ========================================================

    second_fit = fit_once(
        cluster_id_data,
        t_arr[keep_mask],
        y_arr[keep_mask],
        etw=(
            1.0
            if refit_no_early_weight
            else None
        ),
    )

    if second_fit is None:
        return first_fit

    second_fit.n_removed = (
        n_removed
    )

    second_fit.initial_rmse = (
        first_fit.rmse
    )

    return second_fit


# %% find_crossing_points
def find_crossing_points(
    x_pred,
    y_pred,
    x_data,
    y_data,
    tol=0.5,
    x_lim=None,
    y_lim=None,
):
    x_pred = np.asarray(x_pred).ravel()
    y_pred = np.asarray(y_pred).ravel()
    x_data = np.asarray(x_data).ravel()
    y_data = np.asarray(y_data).ravel()

    # ---------------------------------------------------------
    # Validate
    # ---------------------------------------------------------
    if len(x_data) != len(y_data):
        raise ValueError(
            "x_data and y_data must have the same length."
        )

    if len(x_pred) != len(y_pred):
        raise ValueError(
            "x_pred and y_pred must have the same length."
        )

    # Remove invalid curve points
    valid_curve = (
        np.isfinite(x_pred)
        & np.isfinite(y_pred)
    )

    x_pred = x_pred[valid_curve]
    y_pred = y_pred[valid_curve]

    # Remove invalid raw points
    valid_data = (
        np.isfinite(x_data)
        & np.isfinite(y_data)
    )

    x_data = x_data[valid_data]
    y_data = y_data[valid_data]

    if len(x_data) == 0 or len(x_pred) == 0:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    # ---------------------------------------------------------
    # Apply limits to raw data
    # ---------------------------------------------------------
    mask = np.ones(
        len(x_data),
        dtype=bool,
    )

    if x_lim is not None:
        mask &= (
            (x_data >= x_lim[0])
            & (x_data <= x_lim[1])
        )

    if y_lim is not None:
        mask &= (
            (y_data >= y_lim[0])
            & (y_data <= y_lim[1])
        )

    x_data = x_data[mask]
    y_data = y_data[mask]

    if len(x_data) == 0:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    # ---------------------------------------------------------
    # Sort curve by y
    # ---------------------------------------------------------
    order = np.argsort(y_pred)

    y_pred = y_pred[order]
    x_pred = x_pred[order]

    # Remove duplicate y values if necessary
    y_pred_unique, unique_idx = np.unique(
        y_pred,
        return_index=True,
    )

    x_pred_unique = x_pred[unique_idx]

    # ---------------------------------------------------------
    # Evaluate fitted curve at the SAME y position
    # as every raw pick
    # ---------------------------------------------------------
    inside = (
        (y_data >= y_pred_unique.min())
        & (y_data <= y_pred_unique.max())
    )

    x_fit_at_data_y = np.full(
        len(y_data),
        np.nan,
        dtype=float,
    )

    x_fit_at_data_y[inside] = np.interp(
        y_data[inside],
        y_pred_unique,
        x_pred_unique,
    )

    # ---------------------------------------------------------
    # Time residual at SAME spatial position
    # ---------------------------------------------------------
    time_residual = np.abs(
        x_data - x_fit_at_data_y
    )

    mask_cross = (
        np.isfinite(time_residual)
        & (time_residual <= tol)
    )

    return (
        x_data[mask_cross],
        y_data[mask_cross],
    )
# %%
def KVPpicks_clustering(
    KVP_raw_picks,
    eps_m,
    min_pts,
    c_sound,
    Xdist,
    Ydist,
    plotter=False,
    plt_saver=None,
    plt_config=None,
):
    """
    Cluster KVP picks using DBSCAN and merge nearby clusters.

    The input DataFrame is expected to contain the original KVP pick
    information, including the ``time_rel`` and ``Channel`` columns.
    The time and channel information are converted internally into
    the coordinates used for clustering.

    DBSCAN is performed in a physical coordinate system where the
    time axis is converted from seconds to metres using the sound
    velocity. After DBSCAN clustering, nearby clusters are merged
    according to the specified time and distance tolerances.

    Parameters
    ----------
    KVP_raw_picks : pandas.DataFrame
        DataFrame containing the raw KVP picks. It must contain:

        - ``time_rel`` : Arrival time [s].
        - ``Channel`` : Channel index.

        Additional columns are preserved in the returned DataFrame.

    eps_m : float
        DBSCAN epsilon parameter [m].

    min_pts : int
        Minimum number of samples required to define a DBSCAN core point.

    c_sound : float
        Sound velocity [m/s]. The time coordinate is multiplied by this
        value before applying DBSCAN so that both clustering axes are
        expressed in metres.

    Xdist : float
        Maximum allowed separation between clusters along the time axis [s]
        during the cluster-merging step.

    Ydist : float
        Maximum allowed separation between clusters along the distance axis [m]
        during the cluster-merging step.

    plotter : bool, optional
        If True, generate a figure showing the original KVP picks,
        DBSCAN bounding boxes, merged clusters, and final clustered picks.
        Default is False.

    plt_saver : str or pathlib.Path, optional
        Output path used to save the clustering figure. The figure is saved
        only when ``plotter=True`` and ``plt_saver`` is not None.

    Returns
    -------
    KVP_clustered_picks : pandas.DataFrame
        DataFrame containing the input KVP picks assigned to a final
        merged cluster.

        The original columns are preserved and the following columns
        are added:

        - ``x`` : Arrival time [s].
        - ``y`` : Physical distance [m].
        - ``cluster_id`` : Final merged cluster identifier.
        - ``cluster_x0`` : Lower time boundary of the merged cluster [s].
        - ``cluster_y0`` : Lower spatial boundary of the merged cluster [m].
        - ``cluster_Xsize`` : Time extent of the merged cluster [s].
        - ``cluster_Ysize`` : Spatial extent of the merged cluster [m].

        Picks classified as noise or not included in any final merged
        cluster are excluded.


    Notes
    -----
    DBSCAN is performed using the following transformed coordinates::

        x_prime = c_sound * x
        y_prime = y

    where ``x`` is the arrival time in seconds and ``y`` is the physical
    channel distance in metres.
    """

    # ------------------------------------------------------------------
    # Prepare input DataFrame
    # ------------------------------------------------------------------

    x = KVP_raw_picks["x"].to_numpy()
    y = KVP_raw_picks["y"].to_numpy()

    # ------------------------------------------------------------------
    # Run DBSCAN
    # ------------------------------------------------------------------

    # Convert the time coordinate from seconds to metres so that both
    # axes are expressed in the same physical units.
    x_prime = c_sound * x
    y_prime = y.copy()

    dbscan_df, n_noise = run_dbscan(
        x_prime,
        y_prime,
        dist_eps=eps_m,
        min_pts=min_pts,
    )

    # DBSCAN bounding boxes are initially expressed in metres along
    # both axes. Convert the time axis back to seconds.
    dbscan_df["x0"] /= c_sound
    dbscan_df["Xsize"] /= c_sound

    # ------------------------------------------------------------------
    # Merge nearby DBSCAN clusters
    # ------------------------------------------------------------------

    dbscan_merged_df = merge_dbscan_clusters(
        dbscan_df,
        x_dist=Xdist,
        y_dist=Ydist,
    )

    # ------------------------------------------------------------------
    # Assign KVP picks to final merged clusters
    # ------------------------------------------------------------------

    KVP_clustered_picks = KVP_raw_picks.copy()
    KVP_clustered_picks["cluster_x0"] = np.nan
    KVP_clustered_picks["cluster_y0"] = np.nan
    KVP_clustered_picks["cluster_Xsize"] = np.nan
    KVP_clustered_picks["cluster_Ysize"] = np.nan

    # Initialize all picks as unclustered/noise.
    KVP_clustered_picks["cluster_id"] = -1

    for cluster in dbscan_merged_df.itertuples():

        x_min = cluster.x0
        x_max = cluster.x0 + cluster.Xsize

        y_min = cluster.y0
        y_max = cluster.y0 + cluster.Ysize

        mask = (
            (KVP_clustered_picks["x"] >= x_min)
            & (KVP_clustered_picks["x"] <= x_max)
            & (KVP_clustered_picks["y"] >= y_min)
            & (KVP_clustered_picks["y"] <= y_max)
        )

        KVP_clustered_picks.loc[mask, "cluster_id"] = (
            cluster.cluster_id
        )

        KVP_clustered_picks.loc[mask, "cluster_x0"] = (
            cluster.x0
        )

        KVP_clustered_picks.loc[mask, "cluster_y0"] = (
            cluster.y0
        )

        KVP_clustered_picks.loc[mask, "cluster_Xsize"] = (
            cluster.Xsize
        )

        KVP_clustered_picks.loc[mask, "cluster_Ysize"] = (
            cluster.Ysize
        )

    # Keep only picks assigned to a final merged cluster.
    KVP_clustered_picks = KVP_clustered_picks[
        KVP_clustered_picks["cluster_id"] >= 0
    ].copy()

    # ------------------------------------------------------------------
    # Plot clustering results
    # ------------------------------------------------------------------

    if plotter:
        if plt_config is None:
            plt_config = PlotConfig()


        title_str = (
            f"DBSCAN: {dbscan_df['cluster_id'].nunique()} clusters | "
            f"{n_noise} noise | eps = {eps_m} m\n"
            f"Merged: {len(dbscan_merged_df)} clusters | "
            f"Xdist = ±{Xdist} s | Ydist = ±{Ydist} m"
        )

        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot all KVP picks used for clustering.
        ax.scatter(
            x,
            y,
            s=8,
            marker=".",
            color="grey",
            label=f"KVP picks ({len(x)})",
        )

        # Plot original DBSCAN bounding boxes.
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

        # Dummy artist for the DBSCAN legend entry.
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

        # Plot merged cluster bounding boxes.
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

            # Label the merged cluster at the bottom-right corner.
            ax.annotate(
                str(cluster.cluster_id),
                xy=(
                    cluster.x0 + cluster.Xsize,
                    cluster.y0,
                ),
                xytext=(5, -5),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=10,
                fontweight="bold",
                color="black",
                alpha=0.8,
            )

        # Dummy artist for the merged-cluster legend entry.
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

        # Plot picks assigned to final merged clusters.
        ax.scatter(
            KVP_clustered_picks["x"],
            KVP_clustered_picks["y"],
            s=14,
            marker=".",
            color="tab:blue",
            label=f"Clustered KVP picks ({len(KVP_clustered_picks)})",
        )

        # Figure formatting.
        ax.set_title(
            title_str,
            fontsize=plt_config.title_size,
            fontweight="bold",
        )

        ax.set_xlabel(
            "Time [s]",
            fontsize=plt_config.label_size,
        )

        ax.set_ylabel(
            "Distance [m]",
            fontsize=plt_config.label_size,
        )

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=plt_config.tick_size,
        )

        ax.tick_params(
            axis="both",
            which="minor",
            labelsize=plt_config.tick_size,
        )

        ax.grid(
            True,
            linewidth=plt_config.grid_width,
            alpha=plt_config.grid_alpha,
        )

        legend = ax.legend(
            loc="upper center",
            ncols=2,
            fontsize=plt_config.legend_size,
            frameon=True,
        )

        legend.get_frame().set_edgecolor("black")
        legend.get_frame().set_alpha(0.5)

        fig.tight_layout()

        # Save the figure only when an output path is provided.
        if plt_saver is not None:
            fig.savefig(
                plt_saver,
                dpi=300,
                bbox_inches="tight",
            )

        plt.show()
        plt.close(fig)

    return KVP_clustered_picks


def KVPpicks_Hfitting(
    KVP_raw_picks,
    KVP_clustered_picks,
    min_fit_points,
    fit_grid_size,
    early_time_weight,
    early_time_weight_power,
    refit_time_threshold,
    refit_bool,
    marginBB_y,
    plotter=False,
    plt_saver=None,
    plt_config=None,
):
    """
    Fit hyperbolas to clustered KVP picks and select raw picks
    consistent with the fitted hyperbolas.

    Parameters
    ----------
    KVP_raw_picks : pandas.DataFrame
        DataFrame containing the raw KVP picks. It must contain
        ``time_rel``, ``Channel``, ``x`` and ``y`` columns.

    KVP_clustered_picks : pandas.DataFrame
        DataFrame containing KVP picks assigned to final clusters.
        It must contain ``cluster_id``, ``time_rel``, ``Channel``,
        ``x``, ``y`` and the cluster geometry columns:

        - ``cluster_x0``
        - ``cluster_y0``
        - ``cluster_Xsize``
        - ``cluster_Ysize``

    min_fit_points : int
        Minimum number of picks required for a hyperbolic fit.

    fit_grid_size : int
        Number of grid points used during the hyperbolic fitting
        procedure.

    early_time_weight : float
        Weight applied to early-time picks during the hyperbolic fit.

    early_time_weight_power : float
        Power controlling the variation of the early-time weighting.

    refit_time_threshold : float
        Time tolerance [s] used during the refit and hyperbola
        crossing-points selection.

    refit_bool : bool
        If True, perform the optional hyperbola refit.

    marginBB_y : float
        Additional spatial margin [m] applied above and below the
        original cluster extent when evaluating the fitted hyperbola.

    plotter : bool, optional
        If True, generate a figure showing the fitted hyperbolas and
        the raw picks selected by the fits.
        Default is False.

    plt_saver : str or pathlib.Path, optional
        Output path used to save the figure. The figure is only saved
        when ``plotter=True`` and ``plt_saver`` is not None.

    plt_config : PlotConfig, optional
        Matplotlib plotting configuration. If None, a default
        ``PlotConfig`` is created.

    Returns
    -------
    KVP_Hfitted_picks : pandas.DataFrame
        Raw KVP picks selected as compatible with the fitted
        hyperbolas.

        The original columns are preserved, together with:

        - ``cluster_id``
        - ``cluster_x0``
        - ``cluster_y0``
        - ``cluster_Xsize``
        - ``cluster_Ysize``
        - ``y0``
        - ``t0``
        - ``vapp``
        - ``rmse``
        - ``r2``
        - ``n_points``
        - ``n_removed``

        The cluster geometry columns describe the merged DBSCAN
        cluster associated with each pick. The fit parameters describe
        the hyperbola fitted to that cluster.
    """

    # ------------------------------------------------------------------
    # Hyperbolic fitting
    # ------------------------------------------------------------------

    fits = []

    for cluster_id, cluster_data_df in (
        KVP_clustered_picks.groupby("cluster_id")
    ):

        fit = fit_cluster(
            cluster_data=cluster_data_df,
            min_points=min_fit_points,
            grid_size=fit_grid_size,
            early_time_weight=early_time_weight,
            early_time_weight_power=early_time_weight_power,
            refit=refit_bool,
            refit_time_threshold=refit_time_threshold,
            refit_no_early_weight=True,
        )

        if fit is not None:
            fits.append(fit)

    # Create DataFrame containing the fitted hyperbola parameters.
    HyperFits_df = pd.DataFrame(
        [asdict(f) for f in fits]
    )

    # Columns from the hyperbolic fit that will be propagated to
    # every raw pick selected by the corresponding fitted hyperbola.
    fit_attributes = [
        "cluster_id",
        "y0",
        "t0",
        "vapp",
        "rmse",
        "r2",
        "n_points",
        "n_removed",
    ]

    # ------------------------------------------------------------------
    # Select raw KVP picks following the fitted hyperbolas
    # ------------------------------------------------------------------

    selected_raw_picks = []

    # Convert raw-pick coordinates to NumPy arrays once.
    x_raw = KVP_raw_picks["x"].to_numpy()
    y_raw = KVP_raw_picks["y"].to_numpy()

    for _, fit_row in HyperFits_df.iterrows():

        cid = int(fit_row["cluster_id"])

        # Fitted hyperbola parameters.
        t0 = float(fit_row["t0"])
        y0 = float(fit_row["y0"])
        vapp = float(fit_row["vapp"])

        # Get the geometry of the merged cluster.
        cluster_rows = KVP_clustered_picks.loc[
            KVP_clustered_picks["cluster_id"] == cid
        ]

        if cluster_rows.empty:
            continue

        # The cluster geometry is already stored in the clustered
        # DataFrame, so there is no need to reconstruct it from
        # the individual picks.
        cluster_y_min = float(
            cluster_rows["cluster_y0"].iloc[0]
        )

        cluster_y_size = float(
            cluster_rows["cluster_Ysize"].iloc[0]
        )

        cluster_y_max = (
            cluster_y_min
            + cluster_y_size
        )

        # Extend the fitted hyperbola vertically.
        y_min_fit = (
            cluster_y_min
            - marginBB_y
        )

        y_max_fit = (
            cluster_y_max
            + marginBB_y
        )

        # Evaluate the fitted hyperbola over the required
        # spatial range.
        y_curve = np.linspace(
            y_min_fit,
            y_max_fit,
            2000,
        )

        t_curve = np.sqrt(
            t0**2
            + ((y_curve - y0) ** 2 / vapp**2)
        )

        # Find raw picks crossing / compatible with the
        # fitted hyperbola.
        x_cross, y_cross = find_crossing_points(
            x_pred=t_curve,
            y_pred=y_curve,
            x_data=x_raw,
            y_data=y_raw,
            tol=refit_time_threshold,
            y_lim=(
                y_min_fit,
                y_max_fit,
            ),
        )

        if len(x_cross) == 0:
            continue

        # Match the crossing points to the original raw picks.
        selected_mask = np.zeros(
            len(KVP_raw_picks),
            dtype=bool,
        )

        for x_sel, y_sel in zip(
            x_cross,
            y_cross,
        ):

            match = (
                np.isclose(
                    x_raw,
                    x_sel,
                    rtol=0.0,
                    atol=1e-10,
                )
                &
                np.isclose(
                    y_raw,
                    y_sel,
                    rtol=0.0,
                    atol=1e-10,
                )
            )

            selected_mask |= match

        selected_cluster = (
            KVP_raw_picks.loc[selected_mask].copy()
        )

        if selected_cluster.empty:
            continue

        # Associate each selected raw pick with the fitted cluster.
        selected_cluster["cluster_id"] = cid

        selected_raw_picks.append(
            selected_cluster
        )

    # ------------------------------------------------------------------
    # Combine selections from all fitted hyperbolas
    # ------------------------------------------------------------------

    if selected_raw_picks:

        KVP_Hfitted_picks = pd.concat(
            selected_raw_picks,
            ignore_index=True,
        )

    else:

        KVP_Hfitted_picks = pd.DataFrame(
            columns=[
                *KVP_raw_picks.columns,
                "cluster_id",
            ]
        )

    # A raw pick may be selected by more than one fitted hyperbola.
    # Keep only one occurrence of each original KVP pick.
    KVP_Hfitted_picks = (
        KVP_Hfitted_picks
        .drop_duplicates(
            subset=[
                "Channel",
                "time_rel",
            ],
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # Add cluster geometry and fitted hyperbola parameters
    # ------------------------------------------------------------------

    if not KVP_Hfitted_picks.empty:

        # Cluster geometry is unique for each cluster_id.
        cluster_geometry_attributes = [
            "cluster_id",
            "cluster_x0",
            "cluster_y0",
            "cluster_Xsize",
            "cluster_Ysize",
        ]

        cluster_geometry = (
            KVP_clustered_picks[
                cluster_geometry_attributes
            ]
            .drop_duplicates(
                subset="cluster_id"
            )
        )

        KVP_Hfitted_picks = KVP_Hfitted_picks.merge(
            cluster_geometry,
            on="cluster_id",
            how="left",
            validate="many_to_one",
        )

        # Add fitted hyperbola parameters.
        if not HyperFits_df.empty:

            KVP_Hfitted_picks = KVP_Hfitted_picks.merge(
                HyperFits_df[fit_attributes],
                on="cluster_id",
                how="left",
                validate="many_to_one",
            )

    # ------------------------------------------------------------------
    # Plot hyperbolic fitting results
    # ------------------------------------------------------------------

    if plotter:

        # Use default plotting parameters if none are provided.
        if plt_config is None:
            plt_config = PlotConfig()

        title_str = (
            f"Min. fit points = {min_fit_points} | "
            f"Early-time weight = {early_time_weight} "
            f"(power = {early_time_weight_power})\n"
            f"Refit th./cross. tol. = "
            f"±{refit_time_threshold:.3f} s | "
            f"Fit margin = ±{marginBB_y:.0f} m"
        )

        fig, ax = plt.subplots(
            figsize=(12, 7)
        )

        # Plot all raw KVP picks.
        ax.scatter(
            KVP_raw_picks["x"],
            KVP_raw_picks["y"],
            s=8,
            marker=".",
            color="grey",
            alpha=0.6,
            zorder=1,
        )

        # Obtain fitted cluster IDs.
        if not HyperFits_df.empty:

            cluster_ids = np.sort(
                HyperFits_df["cluster_id"].unique()
            )

        else:

            cluster_ids = np.array([])

        # Create one color for each fitted hyperbola.
        colors = mpl.colormaps["jet"].resampled(
            max(len(cluster_ids), 1)
        )

        cluster_colors = {
            cid: colors(i)
            for i, cid in enumerate(cluster_ids)
        }

        # Plot fitted hyperbolas.
        for cid in cluster_ids:

            fit_row = HyperFits_df.loc[
                HyperFits_df["cluster_id"] == cid
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

        # Plot raw picks selected by the fitted hyperbolas.
        if not KVP_Hfitted_picks.empty:

            for cid, selected_cluster in (
                KVP_Hfitted_picks.groupby("cluster_id")
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

        # Figure formatting.
        ax.set_title(
            title_str,
            fontsize=plt_config.title_size,
            fontweight="bold",
        )

        ax.set_xlabel(
            "Time [s]",
            fontsize=plt_config.label_size,
        )

        ax.set_ylabel(
            "Distance [m]",
            fontsize=plt_config.label_size,
        )

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=plt_config.tick_size,
        )

        ax.tick_params(
            axis="both",
            which="minor",
            labelsize=plt_config.tick_size,
        )

        ax.grid(
            True,
            linewidth=plt_config.grid_width,
            alpha=plt_config.grid_alpha,
        )

        fig.tight_layout()

        # Save the figure only when an output path is provided.
        if plt_saver is not None:

            fig.savefig(
                plt_saver,
                dpi=300,
                bbox_inches="tight",
            )

        plt.show()
        plt.close(fig)

    return KVP_Hfitted_picks
