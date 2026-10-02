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

# %% KVP over multiple DAS channels
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
):
    rows = []

    for channel in channel_indices:

        signal = data[channel, :].astype(np.float64)

        signal = SigFilt_HP(signal, fs, nfilt, fmin_filt)
        signal = SigFilt_LP(signal, fs, nfilt, fmax_filt)

        tr = Trace(data=signal)
        tr.stats.sampling_rate = fs
        tr.stats.starttime = UTCDateTime(unix_time_start)
        tr.stats.channel = f"CH{channel}"

        try:
            output = kvp_picker.obspy(tr)
        except Exception as e:
            print(f"KVP failed on channel {channel}: {e}")
            continue

        if output is None or not hasattr(output, "picks"):
            continue

        for pick in output.picks:

            # Array interno de picks por banda
            pick_array = pick.__dict__["_picks_"]

            # Nos quedamos solamente con picks válidos
            valid = pick_array["posix_ons"] > 0

            if not np.any(valid):
                continue

            # Para este KVPick usamos el primer onset válido
            idx = np.flatnonzero(valid)[0]

            rows.append({
                "Channel": channel,
                "Distance_km": distance[channel] / 1e3,
                "Number_bands": pick.nb,
                "Lowest_band": pick.centralfreqs[0],
                "Highest_band": pick.centralfreqs[-1],
                "time_rel": pick_array["t_ons"][idx],
                "unix_time": pick_array["posix_ons"][idx],
            })

    return pd.DataFrame(rows)

channel_indices = np.arange(n_channels)
df_picks = run_kvp_das(
    data=data,
    fs=fs,
    distance=distance,
    unix_time_start=unix_time_start,
    channel_indices=np.arange(n_channels),
    kvp_picker=kvp_picker,
    fmin_filt=fmin_filt,
    fmax_filt=fmax_filt,
    nfilt=nfilt,
)

print("\n=== KVP DETECTIONS ===")
print(df_picks)

print(f"\nTotal detections: {len(df_picks)}")
# %% Plot DAS with KVP picks

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

# ----------------------------------------------------------
# KVP picks
# ----------------------------------------------------------

ax.scatter(
    df_picks["time_rel"],
    df_picks["Distance_km"],
    s=12,
    c="red",
    edgecolors="none",
    label="KVP picks",
    zorder=3,
)

# ----------------------------------------------------------
# Labels
# ----------------------------------------------------------

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