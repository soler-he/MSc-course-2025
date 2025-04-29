import os
import numpy as np
import pickle
import h5py
from astropy.time import Time
from matplotlib.dates import date2num
from datetime import datetime


def load_spectra_and_bursts(
    results_folder: str,   # e.g. "/home/jzhang/SOLER tools/lab_results"
    date: str,             # e.g. "20250329"
    stokes: str,           # e.g. "I"
    burst_type: str,       # "original" or "smoothed"
    burst_numbers: list    # e.g. [1,7,8,9]
) -> dict:
    """
    Load the combined dynamic spectrum from HDF5:
      YYYYMMDD_<STOKES>_COM.hdf5
    and extract ROIs for the requested burst numbers from the
    corresponding pickle.

    Returns a dict:
      { burst_number: {
          full_data, full_time_jd, full_time_unix,
          full_time_mpl, full_freq_values,
          roi_data, roi_time_jd, roi_time_unix,
          roi_time_mpl, roi_freq_values,
          burst_meta
        }
      }
    """

    # 1) build the HDF5 name and check it exists
    combined_name = f"{date}_{stokes}_COM.hdf5"
    combined_path = os.path.join(results_folder, combined_name)
    if not os.path.exists(combined_path):
        raise FileNotFoundError(f"Combined HDF5 not found: {combined_path}")
    print(f"\n🔹 Spectra Loaded (HDF5): {combined_path}")

    # 2) open and drill into the NenuFAR group structure
    with h5py.File(combined_path, "r") as hf:
        coords       = hf["SUB_ARRAY_POINTING_000/BEAM_000/COORDINATES"]
        freq_values  = coords["frequency"][:]    # (nfreq,)
        time_jd      = coords["time"][:]         # (ntime,) — Julian Dates
        intensity2d  = hf["SUB_ARRAY_POINTING_000/BEAM_000/I"][:]  # (ntime, nfreq)

    # 3) stack a singleton polarization axis → shape (ntime, nfreq, 1)
    combined_data = intensity2d[:, :, np.newaxis]

    # 4) convert JD → astropy Time, then to UNIX seconds, datetimes, and mpl dates
    t_jd      = Time(time_jd, format="jd", scale="utc")
    time_unix = t_jd.unix
    time_dt   = t_jd.to_datetime()
    time_mpl  = date2num(time_dt)

    # 5) diagnostics print
    print(f"   ▶ full spectrum shape: {intensity2d.shape}  (time × freq)")
    print(f"   ▶ full time range  : "
          f"{datetime.utcfromtimestamp(time_unix[0])} → "
          f"{datetime.utcfromtimestamp(time_unix[-1])}")
    print(f"   ▶ full freq range  : "
          f"{freq_values[0]:.2f} MHz → {freq_values[-1]:.2f} MHz")

    # 6) load the burst‐list pickle that you generated previously
    #    (use the passed-in date to build the folder name)
    date_folder = f"{date[:4]}_{date[4:6]}_{date[6:]}"
    burst_pkl = os.path.join(
        results_folder,
        f"{date_folder}_output_NenuFAR_T3bursts_detection",
        f"burst_list_{burst_type}.pkl"
    )
    if not os.path.exists(burst_pkl):
        raise FileNotFoundError(f"Burst list not found: {burst_pkl}")
    with open(burst_pkl, "rb") as f:
        burst_list = pickle.load(f)
    print(f"\n✅ Loaded {len(burst_list)} bursts from {burst_pkl}")

    # 7) for each requested burst number, slice out the ROI
    results = {}
    for burst in burst_list:
        num = burst["number"]
        if num not in burst_numbers:
            continue

        print(f"\n🔹 Burst {num} Slicing:")
        print(f"   • requested time window : {burst['start_time']} → {burst['end_time']}")
        print(f"   • requested freq window : {burst['end_freq']:.2f} → {burst['start_freq']:.2f} MHz")

        # convert the burst start/end into JD to match time_jd
        t0_jd = Time(burst["start_time"].timestamp(), format="unix", scale="utc").jd
        t1_jd = Time(burst["end_time"].timestamp(),   format="unix", scale="utc").jd

        # find the indices
        time_idx = np.where((time_jd >= t0_jd) & (time_jd <= t1_jd))[0]
        freq_idx = np.where((freq_values >= burst["end_freq"]) &
                            (freq_values <= burst["start_freq"]))[0]

        if time_idx.size == 0 or freq_idx.size == 0:
            print(f"⚠️  No ROI indices for burst {num}, skipping.")
            continue

        # slice out the ROI, keeping the pol-axis
        roi_data        = combined_data[np.ix_(time_idx, freq_idx, [0])]
        roi_time_jd     = time_jd[time_idx]
        roi_time_unix   = time_unix[time_idx]
        roi_time_dt     = time_dt[time_idx]
        roi_time_mpl    = time_mpl[time_idx]
        roi_freq_values = freq_values[freq_idx]

        print(f"   ▶ ROI shape     : {roi_data.shape}  (time × freq × pol)")
        print(f"   ▶ ROI time span : {datetime.utcfromtimestamp(roi_time_unix[0])} → "
              f"{datetime.utcfromtimestamp(roi_time_unix[-1])}")
        print(f"   ▶ ROI freq span : {roi_freq_values[0]:.2f} → {roi_freq_values[-1]:.2f} MHz")

        results[num] = {
            "full_data":        combined_data,
            "full_time_jd":     time_jd,
            "full_time_unix":   time_unix,
            "full_time_mpl":    time_mpl,
            "full_freq_values": freq_values,
            "roi_data":         roi_data,
            "roi_time_jd":      roi_time_jd,
            "roi_time_unix":    roi_time_unix,
            "roi_time_dt":      roi_time_dt,
            "roi_time_mpl":     roi_time_mpl,
            "roi_freq_values":  roi_freq_values,
            "burst":            burst
        }

    return results