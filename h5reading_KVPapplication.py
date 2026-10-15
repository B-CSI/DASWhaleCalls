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
# LOAD AND CONCATENATE TWO HDF5 FILES
# ============================================================
file_1 = os.path.join(
    ".",
    "data_example",
    "2024_01_13_07h57m34s_HDAS_SAFE_DASWhaleCalls_example.h5",
)
file_2 = os.path.join(
    ".",
    "data_example",
    "2024_01_13_07h58m34s_HDAS_SAFE_DASWhaleCalls_example.h5",
)


def read_h5(path):

    with h5py.File(path, "r") as f:

        result = {
            "data": f["data"][:],
            "fs": float(f.attrs["sampling_frequency"]),
            "dx": float(f.attrs["spatial_sampling"]),
            "unix_time_start": float(f.attrs["unix_time_start"]),
            "distance_start": float(f.attrs["distance_start"]),
            "channel_start": int(f.attrs["original_channel_start"]),
            "channel_end": int(f.attrs["original_channel_end"]),
            "original_number_of_channels": (
                int(f.attrs["original_number_of_channels"])
                if "original_number_of_channels" in f.attrs
                else None
            ),
        }

    return result


h5_1 = read_h5(file_1)
h5_2 = read_h5(file_2)

# ============================================================
# VALIDATE METADATA
# ============================================================

for key in (
    "fs",
    "dx",
    "distance_start",
    "channel_start",
    "channel_end",
):
    if not np.isclose(h5_1[key], h5_2[key]):
        raise ValueError(
            f"Incompatible HDF5 metadata: {key}\n"
            f"File 1: {h5_1[key]}\n"
            f"File 2: {h5_2[key]}"
        )

if h5_1["data"].shape[0] != h5_2["data"].shape[0]:
    raise ValueError("The H5 files have different channel counts.")

if h5_1["data"].shape[0] != (
    h5_1["channel_end"] - h5_1["channel_start"] + 1
):
    raise ValueError("File 1 channel metadata is inconsistent.")

if h5_2["data"].shape[0] != (
    h5_2["channel_end"] - h5_2["channel_start"] + 1
):
    raise ValueError("File 2 channel metadata is inconsistent.")

# ============================================================
# VALIDATE TEMPORAL CONTINUITY
# ============================================================

fs = h5_1["fs"]

n_samples_1 = h5_1["data"].shape[1]
n_samples_2 = h5_2["data"].shape[1]

expected_start_2 = (
    h5_1["unix_time_start"] + n_samples_1 / fs
)

time_error = (
    h5_2["unix_time_start"] - expected_start_2
)

tolerance = 0.5 / fs  # half a sample

if abs(time_error) > tolerance:
    raise ValueError(
        "The H5 files are not temporally contiguous.\n"
        f"Expected second start: {expected_start_2:.9f}\n"
        f"Actual second start:   {h5_2['unix_time_start']:.9f}\n"
        f"Difference:            {time_error:.9f} s"
    )

# ============================================================
# CONCATENATE ALONG TIME AXIS
# ============================================================

data = np.concatenate(
    [h5_1["data"], h5_2["data"]],
    axis=1,
)

# Keep the first file's acquisition reference
unix_time_start = h5_1["unix_time_start"]
dx = h5_1["dx"]
distance_start = h5_1["distance_start"]

original_channel_start = h5_1["channel_start"]
original_channel_end = h5_1["channel_end"]
original_number_of_channels = (
    h5_1["original_number_of_channels"]
)

n_channels, n_time = data.shape

dt = 1.0 / fs
time = np.arange(n_time) * dt

array_indices = np.arange(n_channels, dtype=int)

channel_indices = (
    original_channel_start + array_indices
)

distance = (
    distance_start + array_indices * dx
)

duration_original = n_time / fs

print("\n=== CONCATENATED HDF5 ===")
print(f"File 1: {os.path.basename(file_1)}")
print(f"File 2: {os.path.basename(file_2)}")
print(f"Combined shape: {data.shape}")
print(f"Sampling frequency: {fs:.6f} Hz")
print(f"Duration: {duration_original:.3f} s")
print(f"Temporal offset error: {time_error:.9f} s")
print(
    "Start UTC:",
    datetime.fromtimestamp(
        unix_time_start, tz=timezone.utc
    ).isoformat(),
)
print(
    "End UTC:",
    datetime.fromtimestamp(
        unix_time_start + duration_original,
        tz=timezone.utc,
    ).isoformat(),
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
):

    rows = []

    duration_original = data.shape[1] / fs
    n_channels_to_process = len(array_indices)

    for i, (array_index, channel) in enumerate(
        zip(array_indices, channel_indices),
        start=1,
    ):

        signal = data[array_index, :].astype(np.float64)

        # Filter the complete concatenated signal
        signal = SigFilt_HP(
            signal, fs, nfilt, fmin_filt
        )

        signal = SigFilt_LP(
            signal, fs, nfilt, fmax_filt
        )

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
        if i % 250 == 0 or i == n_channels_to_process:
            print(
                f"Processed {i}/{n_channels_to_process} "
                f"channels (original {channel})"
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
)

print("\n=== KVP DEBUG ===")
print("Concatenated HDF5 shape:", data.shape)
print("Total duration:", duration_original)
print("Number of picks:", len(df_picks))

if not df_picks.empty:
    print(df_picks.head(10))
    print(
        "Time range:",
        df_picks["time_rel"].min(),
        df_picks["time_rel"].max(),
    )
    print(
        "Spatial range:",
        df_picks["Distance_km"].min(),
        df_picks["Distance_km"].max(),
    )
else:
    print("KVP DID NOT RETURN ANY PICKS.")


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

input_file = os.path.abspath(file_1)

# Output filename
output_filename = (
    os.path.splitext(os.path.basename(file_1))[0]
    + ".csv"
)

output_file = os.path.join(
    os.path.dirname(os.path.abspath(file_1)),
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

    f.write("# TEMPORAL CONTEXT\n")
    f.write("# context_source = adjacent real HDF5 recording\n")
    f.write(f"# concatenated_file_1 = {os.path.basename(file_1)}\n")
    f.write(f"# concatenated_file_2 = {os.path.basename(file_2)}\n")
    f.write("# synthetic_noise_padding = disabled\n")
    f.write("#\n")

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
# %%
