# window_detection.py

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.dates import DateFormatter, num2date

def detect_window(
    data3d,
    time_mpl,
    freq_values,
    t_start_mpl,
    t_end_mpl,
    f_min,
    f_max,
    show=True,
    outlier_sigma=None    # <-- new: drop points with residual > n×σ
):
    """
    Slice out a user‐defined time/frequency window and return a dict
    compatible with analyze_burst_velocity (i.e. with keys 'burst',
    'full_data','full_time_mpl','roi_data','roi_time_mpl','roi_freq_values', etc.)
    """

    # 1) 2D spec and masks
    spec2d  = data3d.squeeze()
    tmask   = (time_mpl >= t_start_mpl) & (time_mpl <= t_end_mpl)
    fmask   = (freq_values >= f_min)   & (freq_values <= f_max)
    sub     = spec2d[tmask][:, fmask]
    t_sub   = time_mpl[tmask]
    f_sub   = freq_values[fmask]

    # 2) find peak in each channel
    peak_times, peak_freqs = [], []
    for i, f in enumerate(f_sub):
        idx = np.nanargmax(sub[:, i])
        peak_times.append(t_sub[idx])
        peak_freqs.append(f)
    peak_times = np.array(peak_times)
    peak_freqs = np.array(peak_freqs)

    # 3) first linear drift fit
    tsec = (peak_times - t_sub[0]) * 24*3600
    if len(tsec) > 1:
        slope, intercept, *_ = stats.linregress(tsec, peak_freqs)
        drift_rate = slope
    else:
        slope = intercept = drift_rate = np.nan

    # 4) optional outlier removal
    if outlier_sigma is not None and len(tsec)>1:
        pred      = slope*tsec + intercept
        resid     = peak_freqs - pred
        sigma_res = np.nanstd(resid)
        good      = np.abs(resid) <= outlier_sigma * sigma_res
        # keep only the “good” ones
        tsec, peak_freqs, peak_times = tsec[good], peak_freqs[good], peak_times[good]
        # re‐fit
        if len(tsec)>1:
            slope, intercept, *_ = stats.linregress(tsec, peak_freqs)
            drift_rate = slope

    # 5) assemble a minimal “burst” dict to satisfy analyze_burst_velocity
    burst_meta = {
        "number":         1,
        "start_time":     num2date(t_start_mpl),
        "peak_time":      num2date(peak_times[peak_freqs.argmax()]),
        "end_time":       num2date(t_end_mpl),
        "start_freq":     f_max,
        "end_freq":       f_min,
        "start_time_mpl": t_start_mpl,
        "peak_time_mpl":  peak_times.mean(),
        "end_time_mpl":   t_end_mpl
    }

    # 6) Plot if asked
    if show:
        fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,8))
        # dynamic spectrum
        ax1.imshow(sub.T, origin="lower", aspect="auto",
                   extent=[t_sub[0],t_sub[-1],f_sub[0],f_sub[-1]],
                   cmap="viridis")
        ax1.scatter(peak_times, peak_freqs, c="r", marker="x", label="Cleaned Peaks")
        ax1.set_title("Windowed Dyn. Spec"); ax1.legend()
        fig.colorbar(ax1.images[0], ax=ax1, label="Amplitude")
        ax1.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))

        # drift plot
        ax2.scatter(tsec, peak_freqs, c="b", label="Peak Times")
        if not np.isnan(drift_rate):
            ax2.plot(tsec, slope*tsec+intercept, "r--",
                     label=f"df/dt = {drift_rate:.3f} MHz/s")
        ax2.set_xlabel("Seconds since window start")
        ax2.set_ylabel("Frequency (MHz)")
        ax2.set_title("Frequency Drift")
        ax2.legend()

        plt.tight_layout()
        plt.show()

    # 7) return exactly what analyze_burst_velocity expects
    return {
        "burst":           burst_meta,
        "full_data":       data3d,
        "full_time_mpl":   time_mpl,
        "roi_data":        sub[:,:,np.newaxis],
        "roi_time_mpl":    t_sub,
        "roi_freq_values": f_sub,
        # extras that you might find handy:
        "peak_times":      peak_times,
        "peak_freqs":      peak_freqs,
        "drift_rate":      drift_rate,
        "drift_intercept": intercept
    }