import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.dates import DateFormatter
from datetime import datetime
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import pickle
import matplotlib.patches as patches

#------------------------------------------------------------------------------
# T3 (Type III) Burst Detection Module
#------------------------------------------------------------------------------
# This module provides a single function, T3_detection, which:
#  1) Accepts raw dynamic spectrum arrays (no file I/O inside).
#  2) Splits the time series into overlapping chunks for robust peak finding.
#  3) Optionally cleans 'polluted' frequency channels (RFI mitigation).
#  4) Detects bursts in both raw and smoothed power time series.
#  5) Calculates start/end times & frequencies for each burst.
#  6) Generates visualizations and saves results.
#------------------------------------------------------------------------------

# Global counters for burst numbering
burst_id_counter_original = 1  # ✅ reset at each T3_detection call
burst_id_counter_smoothed = 1

def T3_detection(
    combined_data,                # (ntime, nchan, 1) numpy array
    combined_time_unix,           # (ntime,) UNIX timestamps (float seconds)
    combined_time_dt,             # list of datetime.datetime objects (UTC)
    combined_time_mpl,            # (ntime,) Matplotlib datenums
    combined_freq_values,         # (nchan,) frequencies in MHz
    output_dir,                   # Directory for saving outputs
    save_figures=True,            # Save chunk-by-chunk plots
    save_bursts=True,             # Save detected burst lists (pickles)
    show=True,                    # Display plots in-line
    chunk_size=200,               # # time samples per chunk
    pollution_threshold_factor=2, # RFI removal: std > factor*median flagged
    time_threshold_factor=0.3,    # Fraction of peak defining start/end
    smoothing_sigma=1,            # Gaussian smoothing sigma on normalized power
    overlap=20,                   # Overlap between adjacent chunks
    remove_pollution=True,        # Zero-out polluted freq channels if True
    show_pollute=False,           # Show raw vs cleaned spectrum
    smooth_peak_threshold=0.05    # Minimum height in smoothed series
):
    """
    Detect type III bursts in a dynamic spectrum.

    Parameters:
      combined_data: numpy array
        3D dynamic spectrum (ntime, nchan, 1).
      combined_time_unix: numpy array
        UNIX timestamps (sec).
      combined_time_dt: list of datetime
        Python datetime objects (UTC) matching time axis.
      combined_time_mpl: numpy array
        Matplotlib datenums for plotting.
      combined_freq_values: numpy array
        Frequency axis (MHz).
      output_dir: str
        Folder where figures and burst lists will be saved.

    Returns:
      burst_list_original, burst_list_smoothed: lists of dicts
        Metadata for each detected burst.
    """
    # --- Validate inputs ---
    assert combined_data is not None,       "combined_data required"
    assert combined_time_unix is not None,  "combined_time_unix required"
    assert combined_time_dt is not None,    "combined_time_dt required"
    assert combined_time_mpl is not None,   "combined_time_mpl required"
    assert combined_freq_values is not None, "combined_freq_values required"
    assert output_dir is not None,          "output_dir must be specified"

    # Reset burst counters
    global burst_id_counter_original, burst_id_counter_smoothed
    burst_id_counter_original = 1
    burst_id_counter_smoothed = 1

    # Derive observation date string from first datetime
    obs_date_str = combined_time_dt[0].strftime("%Y_%m_%d")

    # Prepare output folder
    out_folder = os.path.join(output_dir, f"{obs_date_str}_output_NenuFAR_T3bursts_detection")
    os.makedirs(out_folder, exist_ok=True)

    burst_list_original = []
    burst_list_smoothed = []

    n_steps = combined_data.shape[0]
    # Generate overlapping chunk start indices
    starts = range(0, n_steps, chunk_size - overlap)

    for chunk_idx, s0 in enumerate(starts):
        s1 = min(s0 + chunk_size, n_steps)
        sl = slice(s0, s1)
        data_chunk = combined_data[sl, :, :]
        t_unix     = combined_time_unix[sl]
        t_mpl      = combined_time_mpl[sl]

        # --- Pollution removal: zero-out channels with high std dev ---
        if remove_pollution:
            stds = data_chunk.std(axis=(0,2))
            thr  = pollution_threshold_factor * np.median(stds)
            mask = stds <= thr
            clean = data_chunk.copy()
            clean[:, ~mask, :] = 0
        else:
            clean = data_chunk

        # Total power & normalization
        total_power = clean.sum(axis=(1,2)).ravel()
        norm_power  = (total_power - np.median(total_power)) / np.median(total_power)
        smooth_pow  = gaussian_filter1d(norm_power, sigma=smoothing_sigma)

        # Peak finding in raw & smoothed series
        peaks_raw, _      = find_peaks(total_power, height=np.mean(total_power), distance=10)
        all_smooth, _     = find_peaks(smooth_pow, height=np.mean(norm_power), distance=10)
        peaks_smooth = [i for i in all_smooth if smooth_pow[i] >= smooth_peak_threshold]
        peaks_smooth = np.array(peaks_smooth, dtype=int)

        # Helper to calculate burst metadata
        def calculate_burst_times(sig, peaks, t_unix, t_mpl, burst_type):
            bursts = []
            for pk in peaks:
                pv = sig[pk]
                thr_t = time_threshold_factor * pv
                # Find rising edge (start)
                i0 = pk
                while i0 > 0 and sig[i0] >= thr_t and np.gradient(sig[i0-1:i0+1])[0] > 0:
                    i0 -= 1
                # Find falling edge (end)
                i1 = pk
                while i1 < len(sig)-1 and sig[i1] >= thr_t and np.gradient(sig[i1:i1+2])[0] < 0:
                    i1 += 1
                # Mean spectrum around burst → freq range
                spec = clean[i0:i1+1].mean(axis=0).sum(axis=-1)
                freq_thr = 0.3 * np.max(spec)
                f_i0 = np.max(np.where(spec >= freq_thr)[0])
                f_i1 = np.min(np.where(spec >= freq_thr)[0])
                f0 = combined_freq_values[f_i0]
                f1 = combined_freq_values[f_i1]
                # Assign unique ID
                global burst_id_counter_original, burst_id_counter_smoothed
                if burst_type == 'original':
                    num = burst_id_counter_original; burst_id_counter_original += 1
                else:
                    num = burst_id_counter_smoothed; burst_id_counter_smoothed += 1
                name = f"{obs_date_str}_T3_{num:03d}"
                bursts.append({
                    'number': num,
                    'name':   name,
                    'type':   burst_type,
                    'start_time':  datetime.utcfromtimestamp(t_unix[i0]),
                    'peak_time':   datetime.utcfromtimestamp(t_unix[pk]),
                    'end_time':    datetime.utcfromtimestamp(t_unix[i1]),
                    'start_freq':  f0,
                    'end_freq':    f1,
                    'start_time_mpl': t_mpl[i0],
                    'peak_time_mpl':  t_mpl[pk],
                    'end_time_mpl':   t_mpl[i1]
                })
            return bursts

        # Build burst lists
        bo = calculate_burst_times(total_power, peaks_raw, t_unix, t_mpl, 'original')
        bs = calculate_burst_times(smooth_pow, peaks_smooth, t_unix, t_mpl, 'smoothed')
        burst_list_original.extend(bo)
        burst_list_smoothed.extend(bs)

        # --- Visualization per chunk ---
        if save_figures or show:
            fig = plt.figure(figsize=(15,12))
            gs  = GridSpec(3,2, width_ratios=[50,1], height_ratios=[3,1,1], hspace=0.1)
            ax1 = fig.add_subplot(gs[0,0])
            data_plot = data_chunk if show_pollute else clean
            spec2d = data_plot.mean(axis=-1).T
            vmin,vmax = np.percentile(spec2d,[5,95])
            ext = [t_mpl[0], t_mpl[-1], combined_freq_values[0], combined_freq_values[-1]]
            im  = ax1.imshow(spec2d, aspect='auto', origin='lower',
                             extent=ext, cmap='viridis', vmin=vmin, vmax=vmax)
            ax1.set_ylabel('Frequency (MHz)')
            ax1.set_title(f'Chunk {chunk_idx+1}')
            cax = fig.add_subplot(gs[0,1]); plt.colorbar(im, cax=cax, label='Amplitude')
            # Draw smoothed bursts
            for b in bs:
                rect = patches.Rectangle((b['start_time_mpl'], b['end_freq']),
                                         b['end_time_mpl']-b['start_time_mpl'],
                                         b['start_freq']-b['end_freq'],
                                         edgecolor='red', facecolor='none', linewidth=2)
                ax1.add_patch(rect)
                ax1.text(
                    b['start_time_mpl'] + (b['end_time_mpl']-b['start_time_mpl'])/2,
                    b['start_freq'] + 0.5,
                    str(b['number']), color='red', ha='center', va='bottom', fontweight='bold'
                )
            # Plot original & smoothed power
            for ax, pw, bl, title in zip(
                [fig.add_subplot(gs[1,0], sharex=ax1), fig.add_subplot(gs[2,0], sharex=ax1)],
                [total_power, smooth_pow], [bo, bs], ['Original', 'Smoothed']
            ):
                ax.plot(t_mpl, pw, label=title)
                for b in bl:
                    idx0 = list(t_mpl).index(b['start_time_mpl'])
                    ax.scatter(b['start_time_mpl'], pw[idx0], color='green', marker='^')
                    idxp = list(t_mpl).index(b['peak_time_mpl'])
                    ax.scatter(b['peak_time_mpl'], pw[idxp], color='red', marker='o')
                    idx1 = list(t_mpl).index(b['end_time_mpl'])
                    ax.scatter(b['end_time_mpl'], pw[idx1], color='purple', marker='v')
                ax.set_title(title)
                ax.grid(True)
            ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M:%S'))
            fig.autofmt_xdate()
            plt.savefig(os.path.join(out_folder, f'chunk_{chunk_idx+1}.png'), dpi=300)
            if show: plt.show()
            plt.close(fig)

    # --- Save burst pickles ---
    if save_bursts:
        with open(os.path.join(out_folder,'burst_list_original.pkl'),'wb') as f:
            pickle.dump(burst_list_original, f)
        with open(os.path.join(out_folder,'burst_list_smoothed.pkl'),'wb') as f:
            pickle.dump(burst_list_smoothed, f)

    return burst_list_original, burst_list_smoothed
