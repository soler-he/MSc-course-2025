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