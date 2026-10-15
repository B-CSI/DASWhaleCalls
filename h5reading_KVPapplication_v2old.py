# %%
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from obspy import Trace, UTCDateTime
import pandas as pd

from src.signal_functions import (
    SigFilt_HP,
    SigFilt_LP,
)

from kvp import KVP

SCRIPT_NAME = "h5reading_KVPapplication.py"
# %%
# ============================================================
# LOAD HDF5
# ============================================================

file_path = os.path.join(
    ".",
    "data_example",
    "2024_01_13_07h57m34s_HDAS_SAFE_DASWhaleCalls_example.h5",
)

with h5py.File(file_path, "r") as f:

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    data = f["data"][:]

    # --------------------------------------------------------
    # Acquisition metadata
    # --------------------------------------------------------

    fs = float(
        f.attrs["sampling_frequency"]
    )

    dx = float(
        f.attrs["spatial_sampling"]
    )

    unix_time_start = float(
        f.attrs["unix_time_start"]
    )

    distance_start = float(
        f.attrs["distance_start"]
    )

    # --------------------------------------------------------
    # Original channel metadata
    # --------------------------------------------------------

    original_channel_start = int(
        f.attrs["original_channel_start"]
    )

    original_channel_end = int(
        f.attrs["original_channel_end"]
    )

    if "original_number_of_channels" in f.attrs:

        original_number_of_channels = int(
            f.attrs["original_number_of_channels"]
        )

    else:

        original_number_of_channels = None


# %%
# ============================================================
# DATA GEOMETRY
# ============================================================

n_channels, n_time = data.shape

dt = 1.0 / fs

time = np.arange(
    n_time
) * dt


# ------------------------------------------------------------
# Local HDF5 row indices
# ------------------------------------------------------------

array_indices = np.arange(
    n_channels,
    dtype=int,
)


# ------------------------------------------------------------
# Original DAS channel numbers
# ------------------------------------------------------------

channel_indices = (
    original_channel_start
    + array_indices
)


# ------------------------------------------------------------
# Consistency check
# ------------------------------------------------------------

expected_channel_end = (
    original_channel_start
    + n_channels
    - 1
)

if expected_channel_end != original_channel_end:

    raise ValueError(
        "Inconsistent channel metadata:\n"
        f"original_channel_start = "
        f"{original_channel_start}\n"
        f"original_channel_end = "
        f"{original_channel_end}\n"
        f"n_channels = {n_channels}\n"
        f"expected_channel_end = "
        f"{expected_channel_end}"
    )


# ------------------------------------------------------------
# Distance
# ------------------------------------------------------------

distance = (
    distance_start
    + array_indices * dx
)


duration_original = (
    n_time / fs
)


print("\n=== PROCESSED DATA ===")

print(
    f"Shape:               {data.shape}"
)

print(
    f"Channels in HDF5:    {n_channels}"
)

print(
    f"Original channels:   "
    f"{original_channel_start} - "
    f"{original_channel_end}"
)

print(
    f"Time samples:        {n_time}"
)

print(
    f"Sampling frequency:  {fs:.3f} Hz"
)

print(
    f"Duration:            "
    f"{duration_original:.2f} s"
)

print(
    f"Distance:            "
    f"{distance[0] / 1e3:.2f} - "
    f"{distance[-1] / 1e3:.2f} km"
)

print(
    "\nChannel mapping:"
)

print(
    f"data[0, :] -> "
    f"channel {channel_indices[0]}"
)

print(
    f"data[{n_channels - 1}, :] -> "
    f"channel {channel_indices[-1]}"
)


# %%
# ============================================================
# PLOT PROCESSED HDF5
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 6)
)

vmin, vmax = -0.5, 1.0

im = ax.imshow(
    data,
    aspect="auto",
    cmap="jet",
    origin="lower",
    vmin=vmin,
    vmax=vmax,
    extent=[
        time[0],
        time[-1],
        distance[0] / 1e3,
        distance[-1] / 1e3,
    ],
)

ax.set_xlabel("Time [s]")
ax.set_ylabel("Distance [km]")

ax.set_title(
    datetime.fromtimestamp(
        unix_time_start,
        tz=timezone.utc,
    ).strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    )[:22] + "Z"
)

cbar = fig.colorbar(
    im,
    ax=ax,
)

cbar.set_label(
    "Strain"
)

plt.tight_layout()
plt.show()
plt.close(fig)


# %%
# ============================================================
# KVP CONFIGURATION
# ============================================================

fmin_filt = 19.0
fmax_filt = 23.0
nfilt = 6

freqmax = 20.0
octaves = 3
voices = 4
cf_cycles = 90.0
jmp_cycles = 4.0
jump = 2.0
mingap = 3.0
nbands = 4

pad_seconds = 60.0
noise_seed = 12345


kvp_picker = KVP(
    freqmax=freqmax,
    octaves=octaves,
    voices=voices,
    cf_cycles=cf_cycles,
    jmp_cycles=jmp_cycles,
    jump=jump,
    mingap=mingap,
    nbands=nbands,
)


# %%
# ============================================================
# TEST SINGLE CHANNEL
# ============================================================

channel = 750

array_index = (
    channel - original_channel_start
)

signal = data[
    array_index,
    :
].astype(
    np.float64
)


# %%
# FILTER SINGLE CHANNEL
# ============================================================

signal = SigFilt_HP(
    signal,
    fs,
    nfilt,
    fmin_filt,
)

signal = SigFilt_LP(
    signal,
    fs,
    nfilt,
    fmax_filt,
)


# %%
# PLOT FILTERED CHANNEL
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 4)
)

ax.plot(
    time,
    signal,
    linewidth=0.8,
)

ax.set_xlabel(
    "Time [s]"
)

ax.set_ylabel(
    "Strain"
)

ax.set_title(
    f"DAS channel {channel} — "
    f"{distance[array_index] / 1e3:.3f} km"
)

ax.grid(
    True,
    alpha=0.3,
)

plt.tight_layout()
plt.show()
plt.close(fig)


# %%
# ============================================================
# ADD NOISE CONTEXT
# ============================================================

def add_noise_context(
    signal,
    fs,
    pad_seconds=60.0,
    seed=None,
):

    signal = np.asarray(
        signal,
        dtype=np.float64,
    )

    pad_samples = int(
        pad_seconds * fs
    )

    # Robust noise estimate
    median_signal = np.median(
        signal
    )

    mad = np.median(
        np.abs(
            signal - median_signal
        )
    )

    sigma_noise = (
        1.4826 * mad
    )

    rng = np.random.default_rng(
        seed
    )

    noise_before = rng.normal(
        loc=0.0,
        scale=sigma_noise,
        size=pad_samples,
    )

    noise_after = rng.normal(
        loc=0.0,
        scale=sigma_noise,
        size=pad_samples,
    )

    signal_extended = np.concatenate([
        noise_before,
        signal,
        noise_after,
    ])

    return (
        signal_extended,
        sigma_noise,
    )


# %%
# ============================================================
# RUN KVP ON ONE SIGNAL
# ============================================================

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

    return kvp_picker.obspy(
        tr
    )


# %%
# ============================================================
# RUN KVP OVER ALL DAS CHANNELS
# ============================================================

def run_kvp_das(
    data,
    fs,
    distance,
    unix_time_start,
    array_indices,
    channel_indices,
    kvp_picker,
    fmin_filt,
    fmax_filt,
    nfilt,
    pad_seconds=60.0,
    noise_seed=12345,
):

    rows = []

    duration_original = (
        data.shape[1] / fs
    )

    n_channels_to_process = (
        len(array_indices)
    )

    for i, (
        array_index,
        channel,
    ) in enumerate(
        zip(
            array_indices,
            channel_indices,
        ),
        start=1,
    ):

        # ====================================================
        # Original signal
        # ====================================================

        signal = data[
            array_index,
            :
        ].astype(
            np.float64
        )

        # ====================================================
        # Bandpass filtering
        # ====================================================

        signal = SigFilt_HP(
            signal,
            fs,
            nfilt,
            fmin_filt,
        )

        signal = SigFilt_LP(
            signal,
            fs,
            nfilt,
            fmax_filt,
        )

        # ====================================================
        # Synthetic temporal context
        # ====================================================

        channel_seed = (
            noise_seed
            + int(channel)
        )

        signal_extended, sigma_noise = (
            add_noise_context(
                signal=signal,
                fs=fs,
                pad_seconds=pad_seconds,
                seed=channel_seed,
            )
        )

        # ====================================================
        # KVP
        # ====================================================

        try:

            output = run_kvp_signal(
                signal=signal_extended,
                fs=fs,
                starttime=(
                    unix_time_start
                    - pad_seconds
                ),
                channel=channel,
                kvp_picker=kvp_picker,
            )

        except Exception as e:

            print(
                f"KVP failed on channel "
                f"{channel}: {e}"
            )

            continue

        if output is None:
            continue

        if len(output) == 0:
            continue

        # ====================================================
        # Extract picks
        # ====================================================

        for pick in output:

            if not np.any(
                pick.singlepicks[
                    "idx_ref"
                ] >= 0
            ):
                continue

            time_extended = pick.onset(
                posix=False
            )

            time_rel = (
                time_extended
                - pad_seconds
            )

            # ------------------------------------------------
            # Original H5 interval
            # ------------------------------------------------

            if not (
                0 <= time_rel
                < duration_original
            ):
                continue

            # ------------------------------------------------
            # Distance
            # ------------------------------------------------

            distance_km = (
                distance[array_index]
                / 1e3
            )

            rows.append({

                "Channel": int(
                    channel
                ),

                "Distance_km": float(
                    distance_km
                ),

                "Number_bands": int(
                    pick.nb
                ),

                "Lowest_band": float(
                    pick.centralfreqs[0]
                ),

                "Highest_band": float(
                    pick.centralfreqs[-1]
                ),

                "time_rel": float(
                    time_rel
                ),

                "unix_time": float(
                    unix_time_start
                    + time_rel
                ),
            })

        # ====================================================
        # Progress
        # ====================================================

        if (
            i % 250 == 0
            or i == n_channels_to_process
        ):

            print(
                f"Processed "
                f"{i}/"
                f"{n_channels_to_process} "
                f"channels "
                f"(original {channel})"
            )

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


# %%
# ============================================================
# RUN KVP
# ============================================================

df_picks = run_kvp_das(
    data=data,
    fs=fs,
    distance=distance,
    unix_time_start=unix_time_start,
    array_indices=array_indices,
    channel_indices=channel_indices,
    kvp_picker=kvp_picker,
    fmin_filt=fmin_filt,
    fmax_filt=fmax_filt,
    nfilt=nfilt,
    pad_seconds=pad_seconds,
    noise_seed=noise_seed,
)

print(
    "\n=== KVP DETECTIONS ==="
)

print(
    f"Total detections: "
    f"{len(df_picks)}"
)


# %%
# ============================================================
# PLOT KVP PICKS
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 6)
)

im = ax.imshow(
    data,
    aspect="auto",
    origin="lower",
    cmap="gray",
    vmin=-0.5,
    vmax=1.0,
    extent=[
        time[0],
        time[-1],
        distance[0] / 1e3,
        distance[-1] / 1e3,
    ],
)

ax.scatter(
    df_picks["time_rel"],
    df_picks["Distance_km"],
    s=12,
    c="red",
    edgecolors="none",
    label="KVP picks",
    zorder=3,
)

ax.set_xlabel(
    "Time [s]"
)

ax.set_ylabel(
    "Distance [km]"
)

ax.set_title(
    datetime.fromtimestamp(
        unix_time_start,
        tz=timezone.utc,
    ).strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    )[:22] + "Z"
)

cbar = fig.colorbar(
    im,
    ax=ax,
)

cbar.set_label(
    "Strain"
)

ax.legend()

plt.tight_layout()
plt.show()
plt.close(fig)


# %%
# ============================================================
# REMOVE FINAL 2 SECONDS
# ============================================================

edge_time = 2.0

analysis_end = (
    duration_original
    - edge_time
)

df_picks_clean = (
    df_picks[
        df_picks["time_rel"]
        < analysis_end
    ]
    .copy()
)


print(
    "\n=== FINAL CLEAN RESULTS ==="
)

print(
    f"Original duration: "
    f"{duration_original:.3f} s"
)

print(
    f"Edge removed:      "
    f"{edge_time:.3f} s"
)

print(
    f"Analysis end:      "
    f"{analysis_end:.3f} s"
)

print(
    f"Picks before:      "
    f"{len(df_picks)}"
)

print(
    f"Picks after:       "
    f"{len(df_picks_clean)}"
)

print(
    f"Picks removed:     "
    f"{len(df_picks) - len(df_picks_clean)}"
)


# %%
# ============================================================
# FINAL KVP PICKS
# ============================================================

df_picks = df_picks_clean

print(
    "\n=== FINAL KVP PICKS ==="
)

print(
    df_picks
)

print(
    "\nColumns:"
)

print(
    df_picks.columns
)


# %%
# ============================================================
# SAVE CSV IN THE SAME DIRECTORY AS THE HDF5
# ============================================================

input_file = os.path.abspath(
    file_path
)

input_stem = os.path.splitext(
    os.path.basename(input_file)
)[0]


# ------------------------------------------------------------
# Output filename
# ------------------------------------------------------------

output_filename = (
    input_stem
    + "_rawKVPpicks.csv"
)


# ------------------------------------------------------------
# Save next to the input HDF5
# ------------------------------------------------------------

output_file = os.path.join(
    os.path.dirname(input_file),
    output_filename,
)


print(
    f"\nSaving KVP picks to:\n"
    f"{output_file}"
)


# %%
# ============================================================
# SAVE CSV WITH METADATA
# ============================================================

creation_date = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S"
)

with open(
    output_file,
    "w",
    encoding="utf-8",
) as f:

    # ========================================================
    # File information
    # ========================================================

    f.write(
        "# KVP PROCESSING METADATA\n"
    )

    f.write(
        f"# created by {SCRIPT_NAME} ({creation_date})\n"
    )

    f.write(
        f"# input_file: "
        f"{os.path.basename(input_file)}\n"
    )

    f.write(
        f"# output_file: "
        f"{output_filename}\n"
    )

    f.write(
        "#\n"
    )

    # ========================================================
    # Acquisition metadata
    # ========================================================

    f.write(
        "# HDF5 ACQUISITION PARAMETERS\n"
    )

    f.write(
        f"# sampling_frequency_Hz = "
        f"{fs:.12g}\n"
    )

    f.write(
        f"# spatial_sampling_m_per_channel = "
        f"{dx:.12g}\n"
    )

    f.write(
        f"# distance_start_m = "
        f"{distance_start:.12g}\n"
    )

    f.write(
        f"# number_of_channels_in_hdf5 = "
        f"{n_channels}\n"
    )

    f.write(
        f"# original_number_of_channels = "
        f"{original_number_of_channels}\n"
    )

    f.write(
        f"# number_of_time_samples = "
        f"{n_time}\n"
    )

    f.write(
        f"# original_duration_s = "
        f"{duration_original:.12g}\n"
    )

    f.write(
        f"# unix_time_start = "
        f"{unix_time_start:.12g}\n"
    )

    f.write(
        "#\n"
    )

    # ========================================================
    # Spatial selection
    # ========================================================

    f.write(
        "# SPATIAL CHANNEL SELECTION\n"
    )

    f.write(
        "# Channel values in the CSV are ORIGINAL "
        "HDF5/DAS channel indices.\n"
    )

    f.write(
        f"# original_channel_start = "
        f"{original_channel_start}\n"
    )

    f.write(
        f"# original_channel_end = "
        f"{original_channel_end}\n"
    )

    f.write(
        f"# number_of_processed_channels = "
        f"{n_channels}\n"
    )

    f.write(
        "#\n"
    )

    # ========================================================
    # Temporal selection
    # ========================================================

    f.write(
        "# TEMPORAL ANALYSIS WINDOW\n"
    )

    f.write(
        "# analysis_start_s = 0\n"
    )

    f.write(
        f"# edge_time_removed_from_end_s = "
        f"{edge_time:.12g}\n"
    )

    f.write(
        f"# analysis_end_s = "
        f"{analysis_end:.12g}\n"
    )

    f.write(
        "#\n"
    )

    # ========================================================
    # KVP parameters
    # ========================================================

    f.write(
        "# KVP PICKER PARAMETERS\n"
    )

    f.write(
        f"# fmin_filt_Hz = "
        f"{fmin_filt:.12g}\n"
    )

    f.write(
        f"# fmax_filt_Hz = "
        f"{fmax_filt:.12g}\n"
    )

    f.write(
        f"# filter_order = "
        f"{nfilt}\n"
    )

    f.write(
        f"# freqmax = "
        f"{freqmax:.12g}\n"
    )

    f.write(
        f"# octaves = "
        f"{octaves}\n"
    )

    f.write(
        f"# voices = "
        f"{voices}\n"
    )

    f.write(
        f"# cf_cycles = "
        f"{cf_cycles:.12g}\n"
    )

    f.write(
        f"# jmp_cycles = "
        f"{jmp_cycles:.12g}\n"
    )

    f.write(
        f"# jump = "
        f"{jump:.12g}\n"
    )

    f.write(
        f"# mingap = "
        f"{mingap:.12g}\n"
    )

    f.write(
        f"# nbands = "
        f"{nbands}\n"
    )

    f.write(
        "#\n"
    )

    # ========================================================
    # Noise context
    # ========================================================

    f.write(
        "# TEMPORAL NOISE CONTEXT\n"
    )

    f.write(
        f"# pad_seconds = "
        f"{pad_seconds:.12g}\n"
    )

    f.write(
        f"# noise_seed = "
        f"{noise_seed}\n"
    )

    f.write(
        "#\n"
    )

    # ========================================================
    # DATA
    # ========================================================

    f.write(
        "# DATA\n"
    )

    df_picks.to_csv(
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
    f"{len(df_picks)}"
)


# %%
# ============================================================
# OUTPUT
# ============================================================

print(
    "\n========================================"
)

print(
    " CSV SAVED"
)

print(
    "========================================"
)

print(
    f"\nCSV saved to:\n"
    f"{output_file}"
)

print(
    f"\nRows written: "
    f"{len(df_picks)}"
)
