# %% 
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
from datetime import datetime, timezone
from obspy import Trace, UTCDateTime
import pandas as pd
import re
from src.signal_functions import *
# %% 
file_path = "data_test/2024_01_13_07h57m34s_HDAS_SAFE_DASWhaleCalls_example.h5"
with h5py.File(file_path, "r") as f:
    data = f["data"][:] # data = [distance, time]
    fs = float(f.attrs["sampling_frequency"])
    dx = float(f.attrs["spatial_sampling"])
    unix_time_start = float(f.attrs["unix_time_start"])
    distance_start = float(f.attrs["distance_start"])

n_channels, n_time = data.shape
dt = 1.0 / fs
time = np.arange(n_time) * dt
distance = (distance_start + np.arange(n_channels) * dx)

print("\n=== PROCESSED DATA ===")
print(f"Shape:               {data.shape}")
print(f"Channels:            {n_channels}")
print(f"Time samples:        {n_time}")
print(f"Duration:            {time[-1]:.2f} s")
print(
    f"Distance:            "
    f"{distance[0] / 1e3:.2f} - "
    f"{distance[-1] / 1e3:.2f} km"
)

# %% Plot processed HDF5
FigName = "DAS_time_distance_example"
fig, ax = plt.subplots(figsize=(12, 6))
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
        distance[-1] / 1e3
    ]
)

ax.set_xlabel("Time [s]")
ax.set_ylabel("Distance [km]")
ax.set_title(
    datetime.fromtimestamp(
        unix_time_start,
        tz=timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S.%f")[:22] + "Z"
)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Strain")

plt.show()
plt.close(fig)
# %%
# ============================================================
# KVP SETUP (macOS / Apple Silicon)
#
# KVP requires the native OpenMP library libgomp.1.dylib.
#
# 1. Install GCC with Homebrew:
#    $ brew install gcc
#
# 2. Locate libgomp:
#    $ find /opt/homebrew -name "libgomp.1.dylib" 2>/dev/null
#
# 3. Add the GCC library path to KVP's native library:
#    $ install_name_tool -add_rpath \
#      /opt/homebrew/Cellar/gcc/16.2.0/lib/gcc/16 \
#      venv/lib/python3.11/site-packages/kvp/lib/libkvp-maco.so
#
# 4. Test the installation:
#    $ python -c "from kvp import KVP; print('KVP OK')"
#
# If "KVP OK" is printed, KVP is correctly installed.
# ============================================================
# %% Testing KVP in a single DAS channel
channel = 750  # test channel
signal = data[channel, :]

# Create ObsPy Trace from signal
# tr = Trace(data=signal.astype(np.float64))
# tr.stats.sampling_rate = fs
# tr.stats.starttime = UTCDateTime(unix_time_start)

fmin_filt = 19.0
fmax_filt = 23.0
nfilt = 6

signal = SigFilt_HP(
    signal,
    fs,
    nfilt,
    fmin_filt
)

# Low-pass
signal = SigFilt_LP(
    signal,
    fs,
    nfilt,
    fmax_filt
)

# Create ObsPy Trace from filtered signal
tr = Trace(data=signal)
tr.stats.sampling_rate = fs
tr.stats.starttime = UTCDateTime(unix_time_start)

print("\n=== KVP INPUT ===")
print(f"Channel:       {channel}")
print(f"Distance:      {distance[channel] / 1e3:.3f} km")
print(f"Samples:       {tr.stats.npts}")
print(f"Sampling rate: {tr.stats.sampling_rate:.2f} Hz")
print(f"Start time:    {tr.stats.starttime}")

# %% Plot single DAS channel
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(tr.times(),signal,linewidth=0.8)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Strain")
ax.set_title(f"DAS channel {channel} — {distance[channel] / 1e3:.3f} km")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
plt.close(fig)

Nfft = 128
overlap = 0.7
tspect, fspect, psd, info_spect = spectrogram_analysis_nfft(signal, fs, Nfft, overlap, plotter=False)
_,tbin,fbin,fvalid,_ = info_spect
tspect 

flim_min, flim_max = 10,30
fig = plt.figure(figsize=(10, 6))
orig_map=plt.colormaps.get_cmap('jet')
PSDmin = np.nanmin(psd[np.where((fspect >= fvalid))[0],:]) #dB
PSDmax = np.nanmax(psd[np.where((fspect >= fvalid))[0],:]) #dB
if max(fspect)<=5e3:
    plt.pcolormesh(tspect, fspect, psd, cmap=orig_map, vmin=PSDmin, vmax=PSDmax)
    # plt.axhline(fvalid,color='black',linestyle='--',linewidth=4)
    plt.ylabel('Frequency [Hz]')
    plt.ylim(flim_min, flim_max)
else:
    plt.pcolormesh(tspect, fspect*1e-3, psd, cmap=orig_map, vmin=PSDmin, vmax=PSDmax)
    # plt.axhline(fvalid*1e-3,color='black',linestyle='--',linewidth=4)
    plt.ylabel('Frequency [kHz]')
    plt.ylim(flim_min*1e-3, flim_max*1e-3)
plt.title(f'Channel {channel} | Distance = {distance[channel]/1e3:.1f} km\nNFFT={Nfft}, Overlap={int(overlap*100)}%')
cbar = plt.colorbar()
cbar.set_label('PSD [dB re 1A$^2$/Hz]', rotation=270, verticalalignment='baseline')
plt.xlabel('Time [s]')
plt.tight_layout()
plt.show()

# %% Run KVP on one DAS channel
from kvp import KVP
freqmax=20.0
octaves=3
voices=4
cf_cycles=90.0
jmp_cycles=4.0
jump=2.0
mingap=3.0
nbands=4
kvp_picker = KVP(
    freqmax=freqmax,
    octaves=octaves,
    voices=voices,
    cf_cycles=cf_cycles,
    jmp_cycles=jmp_cycles,
    jump=jump,
    mingap=mingap,
    nbands=nbands
)
picks = kvp_picker.obspy(tr)
print("\n=== KVP PICKS ===")
print(picks)

# %%
# Edge padding for KVP
#
# KVP computes kurtosis characteristic functions over finite time
# windows whose duration increases towards the lowest-frequency bands.
# For the frequency scales used in this workflow, these windows can
# extend over several tens of seconds. Since the input DAS files
# contain 60-s segments, events occurring close to the beginning
# or end of a segment may be affected by boundary effects.
#
# To provide sufficient temporal context, each filtered trace is
# extended by 60 s before and after the original H5 segment.
#
# The additional samples are synthetic zero-mean Gaussian noise.
# Its standard deviation is estimated independently for each channel
# from the filtered signal using a robust MAD-based estimator.
#
# KVP is then applied to the extended trace. After picking, only
# detections whose onset falls within the original H5 time interval
# are retained. Their times are referred back to the beginning of
# the original H5 segment.
#
# The synthetic padding is used only to provide temporal context to
# KVP and is never included in the final DAS visualization.
# %% Función para añadir el ruido
def add_noise_context(
    signal,
    fs,
    pad_seconds=60.0,
    seed=None,
):
    """
    Extend a 1-D signal with synthetic Gaussian noise before and after it.

    The noise has zero mean and a channel-specific standard deviation
    estimated from the signal using a robust MAD-based estimator.

    Parameters
    ----------
    signal : np.ndarray
        1-D filtered signal.
    fs : float
        Sampling frequency [Hz].
    pad_seconds : float
        Duration of synthetic noise added before and after the signal [s].
    seed : int or None
        Random seed for reproducible noise.

    Returns
    -------
    signal_extended : np.ndarray
        Signal with synthetic noise before and after.
    sigma_noise : float
        Estimated standard deviation used for the synthetic noise.
    """

    signal = np.asarray(signal, dtype=np.float64)

    pad_samples = int(pad_seconds * fs)

    # ----------------------------------------------------------
    # Robust estimate of the background noise level
    # ----------------------------------------------------------

    median_signal = np.median(signal)

    mad = np.median(
        np.abs(signal - median_signal)
    )

    sigma_noise = 1.4826 * mad

    # ----------------------------------------------------------
    # Synthetic zero-mean Gaussian noise
    # ----------------------------------------------------------

    rng = np.random.default_rng(seed)

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

    # ----------------------------------------------------------
    # Extended signal
    # ----------------------------------------------------------

    signal_extended = np.concatenate([
        noise_before,
        signal,
        noise_after,
    ])

    return signal_extended, sigma_noise

# %% Función KVP para una sola señal
def run_kvp_signal(
    signal,
    fs,
    starttime,
    channel,
    kvp_picker,
):
    """
    Run KVP on a single signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    fs : float
        Sampling frequency [Hz].
    starttime : float
        POSIX start time of the trace.
    channel : int
        DAS channel number.
    kvp_picker : KVP
        Configured KVP picker.

    Returns
    -------
    output : KVPOutput
        KVP output for the input signal.
    """

    tr = Trace(
        data=np.asarray(signal, dtype=np.float64)
    )

    tr.stats.sampling_rate = fs
    tr.stats.starttime = UTCDateTime(starttime)
    tr.stats.channel = f"CH{channel}"

    output = kvp_picker.obspy(tr)

    return output

# %%
def run_kvp_das(
    data,
    fs,
    distance,
    unix_time_start,
    channel_indices,
    kvp_picker,
    fmin_filt,
    fmax_filt,
    nfilt,
    pad_seconds=60.0,
    noise_seed=12345,
):
    """
    Run the complete KVP workflow over multiple DAS channels.

    For each channel:

    1. Extract the original DAS signal.
    2. Apply the 19-23 Hz bandpass filter.
    3. Add synthetic noise before and after the original segment.
    4. Run KVP on the extended signal.
    5. Convert KVP onset times back to the original H5 time axis.
    6. Discard picks outside the original H5 interval.

    The returned picks therefore always refer to the original
    unpadded DAS segment.
    """

    rows = []

    duration_original = data.shape[1] / fs

    n_channels_to_process = len(channel_indices)

    for i, channel in enumerate(channel_indices, start=1):

        # ======================================================
        # Original signal
        # ======================================================

        signal = data[channel, :].astype(np.float64)

        # ======================================================
        # Bandpass filtering
        # ======================================================

        # Filter used for the detection of whale signals
        # around ~20 Hz.
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

        # ======================================================
        # Add synthetic temporal context
        # ======================================================

        # Use a channel-dependent seed so that the noise is
        # reproducible and independent between channels.
        channel_seed = noise_seed + int(channel)

        signal_extended, sigma_noise = add_noise_context(
            signal=signal,
            fs=fs,
            pad_seconds=pad_seconds,
            seed=channel_seed,
        )

        # ======================================================
        # Run KVP
        # ======================================================

        try:

            output = run_kvp_signal(
                signal=signal_extended,
                fs=fs,
                starttime=unix_time_start - pad_seconds,
                channel=channel,
                kvp_picker=kvp_picker,
            )

        except Exception as e:

            print(
                f"KVP failed on channel {channel}: {e}"
            )

            continue

        if output is None or len(output) == 0:
            continue

        # ======================================================
        # Extract KVP picks
        # ======================================================

        for pick in output:

            # --------------------------------------------------
            # Ignore completely invalid KVP picks
            # --------------------------------------------------

            if not np.any(
                pick.singlepicks["idx_ref"] >= 0
            ):
                continue

            # --------------------------------------------------
            # Onset relative to the EXTENDED trace
            # --------------------------------------------------

            time_extended = pick.onset(
                posix=False
            )

            # --------------------------------------------------
            # Refer onset to the ORIGINAL H5 segment
            #
            # Extended trace:
            #
            #   60 s noise | 60 s original | 60 s noise
            #              ^
            #            t = 0
            #
            # --------------------------------------------------

            time_rel = time_extended - pad_seconds

            # --------------------------------------------------
            # Keep only picks inside original H5
            # --------------------------------------------------

            if not (
                0 <= time_rel < duration_original
            ):
                continue

            # --------------------------------------------------
            # Store detection
            # --------------------------------------------------

            rows.append({
                "Channel": channel,
                "Distance_km": distance[channel] / 1e3,
                "Number_bands": pick.nb,
                "Lowest_band": pick.centralfreqs[0],
                "Highest_band": pick.centralfreqs[-1],
                "time_rel": time_rel,
                "unix_time": unix_time_start + time_rel,
            })

        # ======================================================
        # Progress
        # ======================================================

        if (
            i % 250 == 0
            or i == n_channels_to_process
        ):
            print(
                f"Processed "
                f"{i}/{n_channels_to_process} channels"
            )

    return pd.DataFrame(rows)

# %%
channel_indices = np.arange(n_channels)

df_picks = run_kvp_das(
    data=data,
    fs=fs,
    distance=distance,
    unix_time_start=unix_time_start,
    channel_indices=channel_indices,
    kvp_picker=kvp_picker,
    fmin_filt=fmin_filt,
    fmax_filt=fmax_filt,
    nfilt=nfilt,
    pad_seconds=60.0,
    noise_seed=12345,
)

print("\n=== KVP DETECTIONS ===")
print(df_picks)

print(
    f"\nTotal detections: {len(df_picks)}"
)

# %%
fig, ax = plt.subplots(figsize=(12, 6))

vmin, vmax = -0.5, 1.0

im = ax.imshow(
    data,
    aspect="auto",
    origin="lower",
    cmap="gray",
    vmin=vmin,
    vmax=vmax,
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

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Strain")

ax.legend()

plt.tight_layout()
plt.show()
plt.close(fig)
# %%
# Remove edge-affected picks at the end of the H5 segment
#
# The strong concentration of KVP picks during the final seconds
# is interpreted as a boundary effect. Although temporal padding
# is used to reduce edge effects and improve the detection of events
# close to the beginning of the segment, detections very close to
# the end of the original H5 interval remain affected by the finite
# observation window.
#
# To avoid carrying these edge-related detections into the subsequent
# analysis, we exclude the final 2 s of the original H5 segment.
#
# The DAS data itself is not modified; only the corresponding KVP
# detections are removed.

edge_time = 2.0

# Duration of the original H5 segment
duration_original = data.shape[1] / fs
analysis_end = duration_original - edge_time

df_picks_clean = df_picks[
    df_picks["time_rel"] < analysis_end
].copy()

print("\n=== KVP PICKS AFTER EDGE REMOVAL ===")
print(
    f"Removed final {edge_time:.1f} s "
    f"({analysis_end:.1f}–{duration_original:.1f} s)"
)
print(
    f"Picks before filtering: {len(df_picks)}"
)
print(
    f"Picks after filtering:  {len(df_picks_clean)}"
)
print(
    f"Picks removed:          "
    f"{len(df_picks) - len(df_picks_clean)}"
)
# %%
df_picks = df_picks_clean
fig, ax = plt.subplots(figsize=(12, 6))

vmin, vmax = -0.5, 1.0

im = ax.imshow(
    data,
    aspect="auto",
    origin="lower",
    cmap="gray",
    vmin=vmin,
    vmax=vmax,
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

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Strain")

ax.legend()

plt.tight_layout()
plt.show()
plt.close(fig)
# %%
