# %%
from obspy import Trace, UTCDateTime
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.signal_functions import (
    SigFilt_HP,
    SigFilt_LP,
)
import matplotlib.pyplot as plt
# %% 
def run_kvp_signal(
    signal,
    fs,
    starttime,
    channel,
    kvp_picker,
):

    tr = Trace(
        data=np.asarray(
            signal,
            dtype=np.float64,
        )
    )

    tr.stats.sampling_rate = fs

    tr.stats.starttime = UTCDateTime(
        starttime
    )

    tr.stats.channel = (
        f"CH{channel}"
    )

    return kvp_picker.obspy(tr)

# OVER ALL DAS CHANNELS
def run_kvp_das(
    data,
    fs,
    distance,
    unix_time_start,
    array_indices,
    channel_indices,
    kvp_picker,
    show_progress=True
):

    rows = []

    duration_original = data.shape[1] / fs
    n_channels_to_process = len(array_indices)

    iterator = zip(array_indices, channel_indices)

    if show_progress:
        iterator = tqdm(
            iterator,
            total=n_channels_to_process,
            desc="Running KVP",
            unit="channel",
        )

    for array_index, channel in iterator:

        signal = data[array_index, :].astype(np.float64)

        # Filter the complete concatenated signal
        # signal, _ = CMremover(signal)

        # Filter the complete concatenated signal
        # signal = SigFilt_HP(
        #     signal, fs, nfilt, fmin_filt
        # )

        # signal = SigFilt_LP(
        #     signal, fs, nfilt, fmax_filt
        # )

        # Run KVP using real data only
        try:
            output = run_kvp_signal(
                signal=signal,
                fs=fs,
                starttime=unix_time_start,
                channel=channel,
                kvp_picker=kvp_picker,
            )

        except Exception as e:
            print(f"KVP failed on channel {channel}: {e}")
            continue

        if output is None or len(output) == 0:
            continue

        for pick in output:

            if not np.any(
                pick.singlepicks["idx_ref"] >= 0
            ):
                continue

            time_rel = float(pick.onset(posix=False))

            # Keep picks inside the concatenated interval
            if not (0 <= time_rel < duration_original):
                continue

            rows.append({
                "Channel": int(channel),
                "Distance_km": float(distance[array_index] / 1e3),
                "Number_bands": int(pick.nb),
                "Lowest_band": float(pick.centralfreqs[0]),
                "Highest_band": float(pick.centralfreqs[-1]),
                "time_rel": time_rel,
                "unix_time": float(unix_time_start + time_rel),
            })

    return pd.DataFrame(
        rows,
        columns=[
            "Channel",
            "Distance_km",
            "Number_bands",
            "Lowest_band",
            "Highest_band",
            "time_rel",
            "unix_time",
        ],
    )



def KVPpicks_vertical_noise_filter(
    kvp_picks_df,
    window_width=0.02,    # [s]
    min_span_m=10000.0,   # [m]
    min_picks=30,
    time_col="x",
    dist_col="y",
    plotter=False,
    plt_saver=None,
):
    """
    Detect and remove quasi-simultaneous, spatially extended KVP picks.

    Detection is performed using short, overlapping temporal windows.
    A window is classified as noise when it contains enough picks
    distributed over a sufficiently large spatial extent.

    Parameters
    ----------
    kvp_picks_df : pandas.DataFrame
        KVP picks. Requires time_col [s] and dist_col [m].
    window_width : float
        Temporal window width [s].
    min_span_m : float
        Minimum spatial extent [m].
    min_picks : int
        Minimum number of picks inside a window.
    time_col : str
        Time column [s].
    dist_col : str
        Distance column [m].
    plotter : bool
        Plot kept and removed picks.
    plt_saver : str or None
        Output figure path.

    Returns
    -------
    filtered_df : pandas.DataFrame
        Picks retained after filtering.
    noise_df : pandas.DataFrame
        Picks classified as noise.
    """

    if window_width <= 0 or min_span_m < 0 or min_picks < 1:
        raise ValueError("Invalid noise filter parameters.")

    if kvp_picks_df.empty:
        return kvp_picks_df.copy(), kvp_picks_df.copy()

    t = kvp_picks_df[time_col].to_numpy(dtype=float)
    y = kvp_picks_df[dist_col].to_numpy(dtype=float)

    # Ignore invalid coordinates
    valid = np.isfinite(t) & np.isfinite(y)

    # Sort valid picks by time
    valid_indices = np.flatnonzero(valid)
    order = np.argsort(t[valid], kind="stable")

    sorted_indices = valid_indices[order]
    t_sorted = t[sorted_indices]
    y_sorted = y[sorted_indices]

    noise_mask = np.zeros(len(kvp_picks_df), dtype=bool)

    # Overlapping windows (50% overlap)
    step = window_width / 2

    window_starts = np.arange(
        t_sorted[0],
        t_sorted[-1] + step,
        step,
    )

    # Detect spatially extended temporal concentrations
    for t_start in window_starts:

        t_end = t_start + window_width

        lo = np.searchsorted(t_sorted, t_start, side="left")
        hi = np.searchsorted(t_sorted, t_end, side="left")

        n_win = hi - lo

        if n_win < min_picks:
            continue

        y_win = y_sorted[lo:hi]

        # Robust spatial extent (5th to 95th percentile)
        y_low, y_high = np.percentile(y_win, [5, 95])

        span = y_high - y_low

        if span >= min_span_m:

            # Flag all picks inside the detected window
            noise_mask[sorted_indices[lo:hi]] = True

    # Separate original DataFrame
    filtered_df = kvp_picks_df.iloc[~noise_mask].copy()
    noise_df = kvp_picks_df.iloc[noise_mask].copy()

    # Summary
    print("\n=== KVP VERTICAL NOISE FILTER ===")
    print(f"Window width: {window_width} s")
    print(f"Minimum spatial span: {min_span_m} m")
    print(f"Minimum picks: {min_picks}")
    print(f"Input picks: {len(kvp_picks_df)}")
    print(f"Removed (noise): {len(noise_df)}")
    print(f"Kept: {len(filtered_df)}")

    # Plot
    if plotter:

        fig, ax = plt.subplots(figsize=(12, 7))

        ax.scatter(
            filtered_df[time_col],
            filtered_df[dist_col],
            s=12,
            c="grey",
            alpha=0.7,
            label=f"Kept ({len(filtered_df)})",
        )

        ax.scatter(
            noise_df[time_col],
            noise_df[dist_col],
            s=12,
            c="red",
            alpha=0.8,
            label=f"Removed ({len(noise_df)})",
        )

        ax.set_title(
            "KVP vertical noise detection\n"
            f"Window: {window_width} s | "
            f"Span ≥ {min_span_m} m | "
            f"Picks ≥ {min_picks}",
            fontsize=13,
            fontweight="bold",
        )

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Distance [m]")
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()

        if plt_saver is not None:
            fig.savefig(
                plt_saver,
                dpi=300,
                bbox_inches="tight",
            )

        plt.show()
        plt.close(fig)

    return filtered_df, noise_df


# %%
