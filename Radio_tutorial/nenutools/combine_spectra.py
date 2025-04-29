import os
import numpy as np
import h5py
from nenupy.io.tf import Spectra
import astropy.units as u


def combine_spectra(
    date,                    # Observation date, e.g. "20250329"
    output_dir,              # Where to write the combined HDF5
    stokes="I",              # Stokes parameter (I, Q, U, V…)
    rebin_dt=0.5 * u.s,      # Time resolution to rebin
    rebin_df=100 * u.kHz,    # Frequency resolution to rebin
    tmin=None,               # ISO start time (string) or None
    tmax=None,               # ISO end time or None
    force_reload=False,      # If True, always recompute & overwrite
    normalization=True,      # Whether to apply instrument‐response norm
    normalization_type='quiet',  # 'quiet', 'median', or 'mean'
    exclude_freq_range=None  # List of (min,max) freq bands to drop
):
    """
    1) Locate the low‐(0) and high‐(1) .spectra on the TF server.
    2) Call Spectra.get(..., file_name=...) to write each to its own HDF5.
    3) Read both HDF5s back, align in time, concatenate in frequency.
    4) Optionally exclude bands & normalize.
    5) Convert time from JD to UNIX seconds, write a single combined HDF5 and return its path.
    """
    # build intermediate & final filenames
    base = f"{date}_{stokes}"
    low_h5 = os.path.join(output_dir, f"{base}_0.hdf5")
    high_h5 = os.path.join(output_dir, f"{base}_1.hdf5")
    out_h5  = os.path.join(output_dir, f"{date}_{stokes}_COM.hdf5")

    # skip if already done (and not forced)
    if not force_reload and os.path.exists(out_h5):
        print(f"✅ Already combined → {out_h5}")
        return out_h5

    # find the SUN_TRACKING folder on the server
    year, month = date[:4], date[4:6]
    base_path = f"/databf/nenufar-tf/LT11/{year}/{month}/"
    sun_folder = None
    for d in os.listdir(base_path):
        if date in d and "_SUN_TRACKING" in d:
            sun_folder = os.path.join(base_path, d)
            break
    if sun_folder is None:
        raise FileNotFoundError(f"No _SUN_TRACKING for {date} in {base_path}")

    # locate the two .spectra files
    low_spec = high_spec = None
    for fn in os.listdir(sun_folder):
        if fn.endswith("_0.spectra"):
            low_spec = os.path.join(sun_folder, fn)
        elif fn.endswith("_1.spectra"):
            high_spec = os.path.join(sun_folder, fn)
    if low_spec is None or high_spec is None:
        raise FileNotFoundError("Missing *_0.spectra or *_1.spectra")

    # prepare arguments for Spectra.get()
    get_kwargs = {"stokes": stokes, "rebin_dt": rebin_dt, "rebin_df": rebin_df}
    if tmin: get_kwargs["tmin"] = tmin
    if tmax: get_kwargs["tmax"] = tmax

    # 1) export each band to its own HDF5
    for spec, h5path in [(low_spec, low_h5), (high_spec, high_h5)]:
        print(f"📥 Writing {spec} → {h5path}")
        sp = Spectra(spec, check_missing_data=False)
        sp.pipeline.parameters["remove_channels"] = [0,1,-1]
        sp.get(file_name=h5path, **get_kwargs)
        del sp  # free memory

    # helper to read our two HDF5 files
    def read_h5(path):
        with h5py.File(path, "r") as f:
            grp = f["SUB_ARRAY_POINTING_000/BEAM_000/COORDINATES"]
            freq = grp["frequency"][:]    # (nchan,)
            time = grp["time"][:]         # (ntime,) JD
            data = f["SUB_ARRAY_POINTING_000/BEAM_000/I"][:]  # (ntime,nchan)
        return data[..., np.newaxis], time, freq

    # 2) load both bands
    low_data, low_time, low_freq   = read_h5(low_h5)
    high_data, high_time, high_freq = read_h5(high_h5)

    # 3) align in time (truncate the longer one)
    nt_low, nt_high = len(low_time), len(high_time)
    if nt_low > nt_high:
        print(f"⚠️ Truncating low by {nt_low-nt_high} samples")
        low_data, low_time = low_data[:nt_high], low_time[:nt_high]
    elif nt_high > nt_low:
        print(f"⚠️ Truncating high by {nt_high-nt_low} samples")
        high_data, high_time = high_data[:nt_low], high_time[:nt_low]
    assert len(low_time)==len(high_time), "Time axes still mismatch!"

    # 4) concatenate in frequency
    combined_data = np.concatenate([low_data, high_data], axis=1)
    raw_jd       = low_time                     # JD axis
    combined_freq = np.concatenate([low_freq, high_freq])

    # 5) exclude any unwanted freq bands
    if exclude_freq_range:
        mask = np.ones_like(combined_freq, dtype=bool)
        for mn, mx in exclude_freq_range:
            if mn is None:
                mask &= combined_freq > mx
            elif mx is None:
                mask &= combined_freq < mn
            else:
                mask &= ~((combined_freq>=mn) & (combined_freq<=mx))
        combined_data = combined_data[:,mask,:]
        combined_freq = combined_freq[mask]

    # 6) apply normalization if requested
    if normalization:
        print(f"🔍 Normalizing by '{normalization_type}' method")
        if normalization_type=="quiet":
            idx = np.argmin(combined_data.sum(axis=(1,2)))
            resp = np.median(combined_data[idx,:,:], axis=-1)
        elif normalization_type=="median":
            resp = np.median(combined_data, axis=(0,2))
        elif normalization_type=="mean":
            resp = np.mean(combined_data, axis=(0,2))
        else:
            raise ValueError(f"Unknown normalization_type: {normalization_type}")
        combined_data /= resp[None,:,None]

    # 7) convert JD → UNIX seconds (1970-01-01 epoch)
    # unix_time = (raw_jd - 2440587.5) * 86400.0

    # 8) write out the single combined HDF5
    print(f"💾 Saving combined HDF5 → {out_h5}")
    with h5py.File(out_h5, "w") as f:
        coords = f.create_group("SUB_ARRAY_POINTING_000/BEAM_000/COORDINATES")
        coords.create_dataset("frequency", data=combined_freq, dtype="f8")
        coords.create_dataset("time",      data=raw_jd,    dtype="f8")
        f.create_dataset("SUB_ARRAY_POINTING_000/BEAM_000/I",
                         data=combined_data.squeeze(), dtype="f8")
    print("✅ Done.")

    return out_h5


import h5py
import numpy as np
from astropy.time import Time
from matplotlib.dates import date2num
from datetime import datetime

def load_combined_hdf5(hdf5_path):
    """
    Read a combined NenuFAR HDF5 (.hdf5) and return:
      - data3d:       (ntime, nchan, 1) numpy array of Stokes-I
      - time_jd:      (ntime,) numpy array of Julian Dates
      - time_unix:    (ntime,) numpy array of UNIX timestamps (float seconds)
      - time_dt:      list of datetime.datetime (UTC) objects
      - time_mpl:     (ntime,) numpy array of Matplotlib datenums
      - freq:         (nchan,) numpy array of frequencies in MHz

    This matches the behaviour of your old load_spectra_and_bursts routine.
    """
    with h5py.File(hdf5_path, 'r') as f:
        grp    = f['SUB_ARRAY_POINTING_000/BEAM_000/COORDINATES']
        time_jd= grp['time'][:]          # (ntime,) Julian Dates
        freq   = grp['frequency'][:]     # (nchan,)
        data2d = f['SUB_ARRAY_POINTING_000/BEAM_000/I'][:]  # (ntime,nchan)

    # 1) convert JD -> astropy Time
    t_jd      = Time(time_jd, format='jd', scale='utc')
    # 2) UNIX timestamps (float seconds since 1970-01-01)
    time_unix = t_jd.unix
    # 3) Python datetime objects (UTC)
    time_dt   = t_jd.to_datetime()
    # 4) Matplotlib datenums (days since 0001-01-01)
    time_mpl  = date2num(time_dt)

    # 5) expand to (ntime,nchan,1)
    data3d = data2d[..., np.newaxis]

    # diagnostics (optional)
    print(f"Loaded HDF5: {hdf5_path}")
    print(f"  data shape : {data3d.shape}  (ntime, nchan, 1)")
    print(f"  freq range : {freq[0]:.2f} → {freq[-1]:.2f} MHz")
    print(f"  time range : {datetime.utcfromtimestamp(time_unix[0])} → "
          f"{datetime.utcfromtimestamp(time_unix[-1])}")

    return data3d, time_jd, time_unix, time_dt, time_mpl, freq