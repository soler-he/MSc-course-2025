import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from scipy import stats, optimize
from scipy.interpolate import splev, splrep
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter, NullFormatter, LogLocator


def gaussian(x, A, x0, sigma):
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def mirror_lobe(times, flux, peak_time):
    """
    Mirror the data (times and flux) around the specified peak_time.
    times, flux: 1D arrays (assumed sorted in ascending order)
    peak_time: scalar (same units as times)
    
    Returns:
       combined_times, combined_flux: arrays containing the original data
       concatenated with the mirrored data, then sorted by time.
    """
    # Mirror times: for each t, mirrored time = 2*peak_time - t
    mirrored_times = 2 * peak_time - times[::-1]
    mirrored_flux = flux[::-1]
    # Combine original and mirrored data
    combined_times = np.concatenate([times, mirrored_times])
    combined_flux = np.concatenate([flux, mirrored_flux])
    # Sort by time
    sort_idx = np.argsort(combined_times)
    return combined_times[sort_idx], combined_flux[sort_idx]

def analyze_burst_velocity(
    burst_data,
    density_model="saito",
    emission_mechanism="F",# "F"=fundamental, "H"=second harmonic
    freq_range=None,
    fit_method="FWHM", #"FWHM" or "1/e"
    y_scale="none",
    fit_mode="none",  # ✅ "none", "single", "split"
    debug=False,        # ✅ Enable/Disable debug plots
    debug_freq_ranges=None,  #None for every channel, and can be multiple ranges, such as [(42, 42.5), (45,46)]
    show_beam_width=False,       # <<< NEW FLAG
    show_density_models=False
):
    """
    Analyze and visualize the electron beam velocity from a given burst's dynamic spectrum.

    Parameters:
        burst_data (dict): Contains extracted dynamic spectrum, time, frequency, and burst metadata.
        density_model (str): The coronal density model used for distance calculation.
        freq_range (tuple, optional): Custom frequency range (min_freq, max_freq).
        fit_method (str): "FWHM" or "1/e"
        y_scale (str): "log" for logscale or "inverse" for inverse frequency (1/f scale)
        fit_mode (str): "none", "single", "split"
        debug (bool): If True, plot the flux curve and fitted Gaussians.

    Returns:
        None
    """

    burst = burst_data["burst"]
    full_time_mpl = burst_data["full_time_mpl"]
    full_data = burst_data["full_data"]
    roi_data = burst_data["roi_data"]
    roi_time_mpl = burst_data["roi_time_mpl"]
    roi_freq_values = burst_data["roi_freq_values"]

    # --- NEW: if there's a singleton pol‐axis, drop it so roi_data is (ntime, nfreq) ---
    if roi_data.ndim == 3 and roi_data.shape[2] == 1:
        roi_data = roi_data[:, :, 0]

    print(f"\n🔹 Processing Burst {burst['number']}...")
    print(f"   Time Range: {burst['start_time']} to {burst['end_time']}")
    print(f"   Frequency Range: {burst['start_freq']:.2f} MHz to {burst['end_freq']:.2f} MHz")
    print(f"   Extracted Data Shape: {roi_data.shape}")

    # ✅ Apply custom frequency range if provided
    if freq_range is not None:
        min_freq, max_freq = freq_range
        valid_indices = np.where((roi_freq_values >= min_freq) & (roi_freq_values <= max_freq))[0]

        if len(valid_indices) == 0:
            print(f"⚠️ Warning: No valid frequency data in the range {min_freq} - {max_freq} MHz")
            return

        roi_data = roi_data[:, valid_indices]
        roi_freq_values = roi_freq_values[valid_indices]

    # ✅ Convert time to seconds relative to burst start time
    burst_start_time = roi_time_mpl[0]
    burst_end_time = roi_time_mpl[-1]
    peak_times = []
    peak_freqs = []
    start_times = []
    end_times = []

    for i, freq in enumerate(roi_freq_values):
        flux_values = roi_data[:, i]
        peak_idx = np.argmax(flux_values)
        peak_flux = flux_values[peak_idx]

        # ✅ Record peak time and frequency
        peak_times.append(roi_time_mpl[peak_idx])
        peak_freqs.append(freq)

        # ✅ Set threshold for FWHM or 1/e
        if fit_method == "FWHM":
            threshold_flux = peak_flux / 2
        elif fit_method == "1/e":
            threshold_flux = peak_flux / np.e
        else:
            raise ValueError("Invalid fit_method. Use 'FWHM' or '1/e'.")

        if fit_mode == "none":
            try:
                # --- Baseline subtraction and normalization ---
                baseline = np.min(flux_values)
                flux_corrected = flux_values - baseline
                peak_flux_corrected = flux_values[peak_idx] - baseline
                if peak_flux_corrected <= 0:
                    raise ValueError("Peak flux after baseline subtraction is non-positive.")
                flux_norm = flux_corrected / peak_flux_corrected

                # --- Define the normalized threshold ---
                if fit_method == "FWHM":
                    threshold_norm = 0.5
                elif fit_method == "1/e":
                    threshold_norm = 1/np.e
                else:
                    raise ValueError("Invalid fit_method. Use 'FWHM' or '1/e'.")

                # --- Determine start index: last index before peak where normalized flux falls below threshold ---
                start_idx = peak_idx
                while start_idx > 0:
                    if flux_norm[start_idx] < threshold_norm:
                        break
                    start_idx -= 1

                # --- Determine end index: first index after peak where normalized flux falls below threshold ---
                end_idx = peak_idx
                while end_idx < len(flux_norm) - 1:
                    if flux_norm[end_idx] < threshold_norm:
                        break
                    end_idx += 1

                start_time_val = roi_time_mpl[start_idx]
                end_time_val = roi_time_mpl[end_idx]

            except Exception as e:
                print(f"⚠️ None branch failed: {e}")
                start_time_val = np.nan
                end_time_val = np.nan

            # --- Debug Plots ---
            if debug:
                if debug_freq_ranges is not None:
                    in_range = any((freq >= fmin and freq <= fmax) for (fmin, fmax) in debug_freq_ranges)
                    if in_range:
                        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharex=True)
                        # Left: Original flux with threshold line
                        threshold_orig = threshold_norm * peak_flux_corrected + baseline
                        axes[0].plot(np.arange(len(flux_values)), flux_values, 'bo-', label="Original Flux")
                        axes[0].axhline(threshold_orig, color='red', linestyle='--', label=f"Threshold ({fit_method})")
                        axes[0].axvline(start_idx, color='green', linestyle='--', label="Start")
                        axes[0].axvline(end_idx, color='purple', linestyle='--', label="End")
                        axes[0].set_title(f"Original Flux (Channel: {freq:.2f} MHz)")
                        axes[0].set_xlabel("Time Index")
                        axes[0].set_ylabel("Flux")
                        axes[0].legend()
                        # Right: Normalized flux with threshold line
                        axes[1].plot(np.arange(len(flux_norm)), flux_norm, 'bo-', label="Normalized Flux")
                        axes[1].axhline(threshold_norm, color='red', linestyle='--', label=f"Threshold ({fit_method})")
                        axes[1].axvline(start_idx, color='green', linestyle='--', label="Start")
                        axes[1].axvline(end_idx, color='purple', linestyle='--', label="End")
                        axes[1].set_title(f"Normalized Flux (Channel: {freq:.2f} MHz)")
                        axes[1].set_xlabel("Time Index")
                        axes[1].set_ylabel("Normalized Flux")
                        axes[1].legend()
                        plt.tight_layout()
                        plt.show()
                else:
                    # Plot for every channel
                    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharex=True)
                    threshold_orig = threshold_norm * peak_flux_corrected + baseline
                    axes[0].plot(np.arange(len(flux_values)), flux_values, 'bo-', label="Original Flux")
                    axes[0].axhline(threshold_orig, color='red', linestyle='--', label=f"Threshold ({fit_method})")
                    axes[0].axvline(start_idx, color='green', linestyle='--', label="Start")
                    axes[0].axvline(end_idx, color='purple', linestyle='--', label="End")
                    axes[0].set_title(f"Original Flux (Channel: {freq:.2f} MHz)")
                    axes[0].set_xlabel("Time Index")
                    axes[0].set_ylabel("Flux")
                    axes[0].legend()
                    axes[1].plot(np.arange(len(flux_norm)), flux_norm, 'bo-', label="Normalized Flux")
                    axes[1].axhline(threshold_norm, color='red', linestyle='--', label=f"Threshold ({fit_method})")
                    axes[1].axvline(start_idx, color='green', linestyle='--', label="Start")
                    axes[1].axvline(end_idx, color='purple', linestyle='--', label="End")
                    axes[1].set_title(f"Normalized Flux (Channel: {freq:.2f} MHz)")
                    axes[1].set_xlabel("Time Index")
                    axes[1].set_ylabel("Normalized Flux")
                    axes[1].legend()
                    plt.tight_layout()
                    plt.show()

            # Append results for this channel (even if nan) so that the arrays remain the same size.
            start_times.append(start_time_val)
            end_times.append(end_time_val)

        elif fit_mode == "single":
            # Convert time to seconds relative to burst start time
            time_seconds = (roi_time_mpl - burst_start_time) * 24 * 3600

            if len(time_seconds) > 3 and len(flux_values) > 3:
                try:
                    # --- Baseline subtraction and normalization ---
                    baseline       = np.min(flux_values)
                    flux_corr      = flux_values - baseline
                    peak_corr      = flux_corr[peak_idx]
                    if peak_corr <= 0:
                        raise ValueError("Peak flux after baseline subtraction is non-positive.")
                    flux_norm      = flux_corr / peak_corr

                    # --- Initial guesses for Gaussian (A, x0, sigma) ---
                    A0    = 1.0
                    x0_0  = time_seconds[peak_idx]
                    sigma0= (time_seconds[-1] - time_seconds[0]) / 4

                    popt, _ = optimize.curve_fit(
                        gaussian,
                        time_seconds,
                        flux_norm,
                        p0=[A0, x0_0, sigma0],
                        maxfev=5000
                    )
                    A_fit, x0_fit, sigma_fit = popt

                    # --- Compute start & end times from width ---
                    if fit_method == "FWHM":
                        half_width = 1.177 * sigma_fit    # FWHM = 2*√(2 ln2)*σ → half = 1.177σ
                    else:  # "1/e" width = σ
                        half_width = sigma_fit

                    start_time = burst_start_time + (x0_fit - half_width) / (24*3600)
                    end_time   = burst_start_time + (x0_fit + half_width) / (24*3600)

                    # --- DEBUG PLOTTING (unchanged) ---
                    if debug:
                        do_plot = True
                        if debug_freq_ranges:
                            do_plot = any(fmin <= freq <= fmax
                                          for (fmin, fmax) in debug_freq_ranges)
                        if do_plot:
                            fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,4), sharex=True)
                            # raw flux
                            ax1.plot(time_seconds, flux_values, 'bo-', label="Raw Flux")
                            ax1.axvline(x0_fit-half_width, color='green', label="Start")
                            ax1.axvline(x0_fit+half_width, color='purple', label="End")
                            ax1.set_title(f"{freq:.2f} MHz Raw")
                            ax1.legend()

                            # normalized + fit
                            ax2.plot(time_seconds, flux_norm, 'bo-', label="Normalized")
                            tt = np.linspace(time_seconds.min(), time_seconds.max(), 200)
                            ax2.plot(tt, gaussian(tt, *popt), 'r--', label="Gaussian Fit")
                            ax2.axvline(x0_fit-half_width, color='green', label="Start")
                            ax2.axvline(x0_fit+half_width, color='purple', label="End")
                            ax2.set_title(f"{freq:.2f} MHz Fit")
                            ax2.legend()

                            plt.tight_layout()
                            plt.show()

                except Exception as e:
                    print(f"⚠️ Gaussian fit failed at {freq:.2f} MHz: {e}")
                    start_time = np.nan
                    end_time   = np.nan

                # Always append (even if NaN)
                start_times.append(start_time)
                end_times.append(end_time)

                
        elif fit_mode == "split":
            # 1) Identify the global maximum for this channel
            global_peak_idx = np.argmax(flux_values)  # index of the highest point
            global_peak_time = roi_time_mpl[global_peak_idx]
            global_peak_time_sec = (global_peak_time - burst_start_time) * 24*3600

            # 2) Split the data at the global peak
            left_times = roi_time_mpl[:global_peak_idx+1]   # +1 to include the peak in the left side
            left_flux = flux_values[:global_peak_idx+1]
            right_times = roi_time_mpl[global_peak_idx:]    # from the peak onward
            right_flux = flux_values[global_peak_idx:]

            # 3) Convert times to seconds from burst start
            left_times_sec  = (left_times  - burst_start_time) * 24*3600
            right_times_sec = (right_times - burst_start_time) * 24*3600

            # 4) Clean up data: remove NaNs and infinities
            valid_left  = ~np.isnan(left_times_sec) & ~np.isnan(left_flux) & ~np.isinf(left_times_sec) & ~np.isinf(left_flux)
            valid_right = ~np.isnan(right_times_sec) & ~np.isnan(right_flux) & ~np.isinf(right_times_sec) & ~np.isinf(right_flux)
            left_times_sec  = left_times_sec[valid_left]
            left_flux       = left_flux[valid_left]
            right_times_sec = right_times_sec[valid_right]
            right_flux      = right_flux[valid_right]

            # Initialize fallback
            start_time = burst_start_time
            end_time   = burst_start_time

            # ------------------------------
            # (A) Fit the RISE (Left Side)
            # ------------------------------
            try:
                # Baseline & normalization for left side
                baseline_left = np.min(left_flux)
                left_flux_corr = left_flux - baseline_left
                left_peak_val  = np.max(left_flux_corr)  # the global maximum for this side is the actual peak
                if left_peak_val <= 0:
                    raise ValueError("Left side peak flux is non-positive after baseline subtraction.")
                left_flux_norm = left_flux_corr / left_peak_val

                # Mirror around the global peak time for the left side
                #   (peak_time_sec is the time of the global max)
                mirrored_left_times, mirrored_left_flux = mirror_lobe(left_times_sec, left_flux_norm, global_peak_time_sec)

                # Fit Gaussian
                init_guess_left = [1.0, global_peak_time_sec, 1.0]  # amplitude ~1, center ~peak_time_sec, sigma ~1
                popt_left, _ = optimize.curve_fit(
                    gaussian, 
                    mirrored_left_times, 
                    mirrored_left_flux,
                    p0=init_guess_left,
                    maxfev=5000
                )
                A_left, x0_left, sigma_left = popt_left

                if fit_method == "FWHM":
                    width_left = 2.355 * sigma_left
                elif fit_method == "1/e":
                    width_left = sigma_left * np.sqrt(2*np.log(1/np.e))
                else:
                    raise ValueError("Invalid fit_method. Use 'FWHM' or '1/e'.")

                # The left start time is (x0_left - width_left/2)
                start_time_sec = x0_left - width_left/2
                start_time = burst_start_time + start_time_sec/(24*3600)

            except Exception as e:
                print(f"⚠️ Left fit failed: {e}")
                start_time = burst_start_time

            # ------------------------------
            # (B) Fit the DECAY (Right Side)
            # ------------------------------
            try:
                baseline_right = np.min(right_flux)
                right_flux_corr = right_flux - baseline_right
                right_peak_val  = np.max(right_flux_corr)  # same global peak for the right side
                if right_peak_val <= 0:
                    raise ValueError("Right side peak flux is non-positive after baseline subtraction.")
                right_flux_norm = right_flux_corr / right_peak_val

                mirrored_right_times, mirrored_right_flux = mirror_lobe(right_times_sec, right_flux_norm, global_peak_time_sec)

                init_guess_right = [1.0, global_peak_time_sec, 1.0]
                popt_right, _ = optimize.curve_fit(
                    gaussian, 
                    mirrored_right_times, 
                    mirrored_right_flux,
                    p0=init_guess_right,
                    maxfev=5000
                )
                A_right, x0_right, sigma_right = popt_right

                if fit_method == "FWHM":
                    width_right = 2.355 * sigma_right
                elif fit_method == "1/e":
                    width_right = sigma_right * np.sqrt(2*np.log(1/np.e))

                # The right end time is (x0_right + width_right/2)
                end_time_sec = x0_right + width_right/2
                end_time = burst_start_time + end_time_sec/(24*3600)

            except Exception as e:
                print(f"⚠️ Right fit failed: {e}")
                end_time = burst_start_time

            # ------------------------------
            # (C) Single Debug Plot
            # ------------------------------
            if debug:
                do_plot = True
                if debug_freq_ranges is not None:
                    do_plot = any((freq >= fmin and freq <= fmax) for (fmin, fmax) in debug_freq_ranges)
                if do_plot:
                    plt.figure(figsize=(10,6))
                    # Combine left & right true flux (after baseline sub & normalization)
                    left_flux_final  = (left_flux - baseline_left)   / left_peak_val
                    right_flux_final = (right_flux - baseline_right) / right_peak_val

                    # Combine times & flux for display
                    all_times_sec = np.concatenate([left_times_sec, right_times_sec])
                    all_flux_norm = np.concatenate([left_flux_final, right_flux_final])

                    # Plot the combined true light curve
                    plt.plot(all_times_sec, all_flux_norm, 'ko-', label="True Light Curve")

                    # Evaluate left & right fits on their original times
                    left_fit = gaussian(left_times_sec, *popt_left) if 'popt_left' in locals() else None
                    right_fit = gaussian(right_times_sec, *popt_right) if 'popt_right' in locals() else None
                    if left_fit is not None:
                        plt.plot(left_times_sec, left_fit, 'r--', label="Left Gaussian Fit")
                    if right_fit is not None:
                        plt.plot(right_times_sec, right_fit, 'm--', label="Right Gaussian Fit")

                    plt.axvline(start_time_sec, color='green',  linestyle='--', label="Start Time")
                    plt.axvline(end_time_sec,   color='purple', linestyle='--', label="End Time")

                    plt.title(f"Split Mode Fit at {freq:.2f} MHz")
                    plt.xlabel("Time (s) from burst start")
                    plt.ylabel("Normalized Flux")
                    plt.legend()
                    plt.grid(True)
                    plt.show()

            # Append final times
            start_times.append(start_time)
            end_times.append(end_time)

    peak_times = np.array(peak_times)
    peak_freqs = np.array(peak_freqs)
    start_times = np.array(start_times)
    end_times = np.array(end_times)

    peak_times_seconds = (peak_times - burst_start_time) * 24 * 3600
    start_times_seconds = (start_times - burst_start_time) * 24 * 3600
    end_times_seconds = (end_times - burst_start_time) * 24 * 3600

    # ✅ Perform linear fit to obtain drift rate
    if len(peak_times_seconds) > 1:
        slope, intercept, _, _, _ = stats.linregress(peak_times_seconds, peak_freqs)
        drift_rate = slope  # MHz/s
    else:
        drift_rate = np.nan

    print(f"   Drift Rate (df/dt): {drift_rate:.4f} MHz/s")

    # ✅ Convert frequency to radial distance using density model
    e = 4.8e-10
    m_e = 9.1e-28
    pi = np.pi

    # pick plasma frequency: fundamental f_pe=f, or harmonic f_pe=f/2
    if emission_mechanism.upper() == "F":
        f_pe_Hz = peak_freqs * 1e6
    elif emission_mechanism.upper() == "H":
        f_pe_Hz = (peak_freqs/2) * 1e6
    else:
        raise ValueError(f"Unknown emission_mechanism: {emission_mechanism}")

    density = (pi * f_pe_Hz)**2 * m_e / (4 * pi * e**2)

    if density_model == "saito":
        r_solar = np.linspace(1.1, 10, 10000)
        density_saito = 1.36e6 * r_solar ** (-2.14) + 1.68e8 * r_solar ** (-6.13)
        tck = splrep(density_saito[::-1], r_solar[::-1], s=0)
        r_data = splev(density, tck)

    elif density_model == "leblanc98":
        r_solar = np.linspace(1.1, 10, 10000)
        density_leblanc98 = 3.3e5 * r_solar**(-2.) + 4.1e6 * r_solar**(-4.) + 8.0e7 * r_solar**(-6.)
        tck = splrep(density_leblanc98[::-1], r_solar[::-1], s=0)
        r_data = splev(density, tck)

    elif density_model == "parkerfit":
        r_solar = np.linspace(1.1, 10, 10000)
        h0 = 144.0 / 6.96e5
        h1 = 20.0 / 960.
        nc = 3e11 * np.exp(-(r_solar - 1.0) / h1)
        density_parkerfit = 4.8e9 / r_solar**14. + 3e8 / r_solar**6. + 1.39e6 / r_solar**2.3 + nc
        tck = splrep(density_parkerfit[::-1], r_solar[::-1], s=0)
        r_data = splev(density, tck)

    elif density_model == "dndr_leblanc98":
        r_solar = np.linspace(1.1, 10, 10000)
        density_dndr_leblanc98 = -2. * 3.3e5 * r_solar**(-3.) - 4. * 4.1e6 * r_solar**(-5.) - 6. * 8.0e7 * r_solar**(-7.)
        tck = splrep(density_dndr_leblanc98[::-1], r_solar[::-1], s=0)
        r_data = splev(density, tck)

    else:
        raise ValueError("Unsupported density model.")

    # ✅ Perform linear fit to obtain velocity for PEAK flux
    if len(peak_times_seconds) > 1:
        vel_slope, vel_intercept, _, _, _ = stats.linregress(peak_times_seconds, r_data)
        velocity_peak = vel_slope * 7e10 / 3e10  # Convert cm/s to fraction of c
    else:
        velocity_peak = np.nan

    # ✅ Perform linear fit to obtain velocity for START flux
    if len(start_times_seconds) > 1:
        vel_slope_start, vel_intercept_start, _, _, _ = stats.linregress(start_times_seconds, r_data)
        velocity_start = vel_slope_start * 7e10 / 3e10
    else:
        velocity_start = np.nan

    # ✅ Perform linear fit to obtain velocity for END flux
    if len(end_times_seconds) > 1:
        vel_slope_end, vel_intercept_end, _, _, _ = stats.linregress(end_times_seconds, r_data)
        velocity_end = vel_slope_end * 7e10 / 3e10
    else:
        velocity_end = np.nan

    print(f"   Electron Beam Velocity (Peak): {velocity_peak:.4f} c")
    print(f"   Electron Beam Velocity (Start): {velocity_start:.4f} c")
    print(f"   Electron Beam Velocity (End): {velocity_end:.4f} c")

        # --------------------------------------------------------------------
    # Optionally show all density‐model curves + horizontal density ranges
    # --------------------------------------------------------------------
    if show_density_models:
        # 1) Radius grid
        r_solar = np.linspace(1.1, 10.0, 5000)

        # 2) Build each model
        models = {}
        models["saito"] = 1.36e6 * r_solar**(-2.14) + 1.68e8 * r_solar**(-6.13)
        models["leblanc98"] = (
            3.3e5 * r_solar**(-2.0)
            + 4.1e6 * r_solar**(-4.0)
            + 8.0e7 * r_solar**(-6.0)
        )
        models["parkerfit"] = (
            4.8e9  / r_solar**14
            + 3e8   / r_solar**6
            + 1.39e6* r_solar**(-2.3)
            + 3e11  * np.exp(-(r_solar - 1.0) / (20.0/960.0))
        )

        # 3) Compute burst‐range densities
        #    fundamental:
        f0 = burst["start_freq"]  # MHz
        f1 = burst["end_freq"]    # MHz
        # convert MHz→Hz
        f0_Hz = f0 * 1e6
        f1_Hz = f1 * 1e6
        # plasma frequency → density
        dens0_fund = (pi * f0_Hz)**2 * m_e / (4*pi*e**2)
        dens1_fund = (pi * f1_Hz)**2 * m_e / (4*pi*e**2)
        # harmonic: f/2
        dens0_harm = (pi * (f0_Hz/2))**2 * m_e / (4*pi*e**2)
        dens1_harm = (pi * (f1_Hz/2))**2 * m_e / (4*pi*e**2)

        # 4) Plot them
        fig, ax = plt.subplots(figsize=(8,6))
        for name, dens_profile in models.items():
            ax.plot(
                r_solar, dens_profile,
                label=name, linewidth=1.5
            )

        # 5) Overlay horizontal bands
        ax.fill_between(
            [r_solar.min(), r_solar.max()],
            dens0_fund, dens1_fund,
            color='gray', alpha=0.3,
            label='Fundamental range'
        )
        ax.fill_between(
            [r_solar.min(), r_solar.max()],
            dens0_harm, dens1_harm,
            color='orange', alpha=0.3,
            label='Harmonic range'
        )

        # 6) Finalize
        ax.set_yscale('log')
        ax.set_xlabel('Radial Distance [$R_\\odot$]')
        ax.set_ylabel('Electron Density [$\\mathrm{cm}^{-3}$]')
        ax.set_title('Coronal Density Models & Burst Density Ranges')
        ax.grid(True, which='both', ls=':')
        ax.legend(loc='best', fontsize='small')
        plt.tight_layout()
        plt.show()

    # ✅ Compute beam width (Δf) for each time interval
    beam_widths = []
    beam_times = []
    normalized_beam_widths = []

    # Define time intervals for calculating Δf / f_max (e.g., 1 second steps)
    time_intervals = np.arange(peak_times_seconds.min(), peak_times_seconds.max(), 1.0)

    for t_start in time_intervals:
        t_end = t_start + 1.0

        # ✅ Find points within this time interval
        in_interval = (peak_times_seconds >= t_start) & (peak_times_seconds < t_end)
        
        if np.sum(in_interval) >= 2:
            f_start = np.min(peak_freqs[in_interval])
            f_end = np.max(peak_freqs[in_interval])
            f_max = np.mean(peak_freqs[in_interval])  # Average if multiple peaks
            
            Δf = f_end - f_start
            beam_widths.append(Δf)
            beam_times.append(t_start)

            if f_max > 0:
                normalized_beam_widths.append(Δf / f_max)
            else:
                normalized_beam_widths.append(np.nan)
                

    # ✅ Plot Results: Add 4th panel for beam width evolution
    nrows = 4 if show_beam_width else 3
    heights = [1,1,1.2,1] if show_beam_width else [1,1,1.2]
    fig, axs = plt.subplots(
        nrows, 1, figsize=(12, 4*nrows),
        gridspec_kw={'height_ratios': heights},
        squeeze=False
    )
    axs = axs.ravel()

    # ✅ Expand time range by 30 seconds before and after burst
    expanded_time_start = roi_time_mpl[0] - 30 / (24 * 3600)  # 30 seconds in fraction of a day
    expanded_time_end = roi_time_mpl[-1] + 30 / (24 * 3600)

    # ✅ Find indices that correspond to expanded range
    start_idx = np.searchsorted(roi_time_mpl, expanded_time_start)
    end_idx = np.searchsorted(roi_time_mpl, expanded_time_end)

    # ✅ Expand roi_data and roi_time_mpl to cover this range
    roi_time_mpl = roi_time_mpl[start_idx:end_idx]
    roi_data = roi_data[start_idx:end_idx, :]

    # use nanpercentile for robustness, and your roi_data is now 2-D
    vmin, vmax = np.nanpercentile(roi_data, [5, 95])

    if y_scale == "inverse":
        # use nanpercentile for robust color limits
        vmin, vmax = np.nanpercentile(roi_data, [5, 95])

        # build an inverse‐frequency axis (1 / MHz)
        inv_freq = 1.0 / roi_freq_values  # shape (nfreq,)

        # prepare meshgrid for pcolormesh: Time along x, inv_freq along y
        T, F = np.meshgrid(roi_time_mpl, inv_freq)

        # plot with pcolormesh so the non‐uniform y‐spacing is handled
        im = axs[0].pcolormesh(
            T, F, roi_data.T,
            shading="auto",
            cmap="viridis",
            vmin=vmin, vmax=vmax
        )
        axs[0].invert_yaxis()

        # overlay the inverse‐frequency scatter points (peak, start, end)
        axs[0].scatter(
            peak_times,
            1.0 / peak_freqs,
            color="black", marker="+",
            label="Peak Flux", zorder=5
        )
        axs[0].scatter(
            start_times,
            1.0 / peak_freqs,
            color="red", marker="+",
            label="Start Rise", zorder=5
        )
        axs[0].scatter(
            end_times,
            1.0 / peak_freqs,
            color="blue", marker="+",
            label="End Decay", zorder=5
        )

        axs[0].set_ylabel("1 / Frequency (1/MHz)")
        axs[0].set_title(f"Dynamic Spectrum for Burst {burst['number']}")

        # add a right‐hand axis in actual MHz
        import matplotlib.ticker as ticker
        secax = axs[0].secondary_yaxis(
            'right',
            functions=(lambda inv: 1.0/inv, lambda f: 1.0/f)
        )
        secax.set_ylabel("Frequency (MHz)")
        secax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False))
        secax.yaxis.set_minor_formatter(ticker.NullFormatter())

        # colorbar
        fig.colorbar(im, ax=axs[0], label="Amplitude")

    elif y_scale == "log":
        extent = [roi_time_mpl[0], roi_time_mpl[-1], roi_freq_values[0], roi_freq_values[-1]]
        
        im = axs[0].imshow(roi_data.T, aspect="auto", origin="lower", extent=extent,
                        cmap="viridis", vmin=vmin, vmax=vmax)
        
        axs[0].set_yscale("log")
        
        axs[0].scatter(peak_times, peak_freqs, color="black", marker="+", label="Peak Flux", zorder=5)
        axs[0].scatter(start_times, peak_freqs, color="red", marker="+", label="Start Rise", zorder=5)
        axs[0].scatter(end_times, peak_freqs, color="blue", marker="+", label="End Decay", zorder=5)

        axs[0].set_ylabel("Frequency (MHz)")

        # (optional) mirror linear MHz on the right
        sec = axs[0].secondary_yaxis(
            'right',
            functions=(lambda f: f, lambda f: f)
        )
        sec.set_ylabel("Frequency (MHz)")

        # force the secondary axis to show more ticks and no “×10ⁿ” offset:
        sec.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        sec.yaxis.set_major_formatter(ScalarFormatter())
        sec.yaxis.set_minor_formatter(NullFormatter())
        # hide the offset text (the “×10¹”):
        sec.yaxis.get_offset_text().set_visible(False)

    else:
        extent = [roi_time_mpl[0], roi_time_mpl[-1],
                  roi_freq_values[0], roi_freq_values[-1]]
        im = axs[0].imshow(
            roi_data.T, aspect="auto", origin="lower",
            extent=extent, cmap="viridis",
            vmin=vmin, vmax=vmax
        )
        
        axs[0].scatter(peak_times, peak_freqs, color="black", marker="+", label="Peak Flux", zorder=5)
        axs[0].scatter(start_times, peak_freqs, color="red", marker="+", label="Start Rise", zorder=5)
        axs[0].scatter(end_times, peak_freqs, color="blue", marker="+", label="End Decay", zorder=5)

        axs[0].set_ylabel("Frequency (MHz)")

    # ✅ Set x-axis format to UTC and label
    axs[0].xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    axs[0].set_xlim(expanded_time_start, expanded_time_end)
    axs[0].set_xlabel("Time (UT)")
    axs[0].set_title(f"Dynamic Spectrum for Burst {burst['number']}")
    axs[0].legend()


    # ✅ Frequency Drift Plot
    axs[1].scatter(peak_times_seconds, peak_freqs, color="blue", label="Peak Flux Points")
    if len(peak_times_seconds) > 1:
        axs[1].plot(peak_times_seconds, peak_times_seconds * slope + intercept, "r--", label="Linear Fit (Peak)")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Frequency (MHz)")
    axs[1].set_title("Frequency Drift Over Time")
    axs[1].legend()

    # ✅ Radial Distance vs Time Plot
    axs[2].scatter(peak_times_seconds, r_data, color="green", label="Radial Distance")
    if len(start_times_seconds) > 1:
        axs[2].plot(start_times_seconds, start_times_seconds * vel_slope_start + vel_intercept_start,
                    "red", linestyle="dashed", label=f"Start Velocity: {velocity_start:.4f} c")
    if len(peak_times_seconds) > 1:
        axs[2].plot(peak_times_seconds, peak_times_seconds * vel_slope + vel_intercept,
                    "orange", linestyle="dashed", label=f"Peak Velocity: {velocity_peak:.4f} c")
    if len(end_times_seconds) > 1:
        axs[2].plot(end_times_seconds, end_times_seconds * vel_slope_end + vel_intercept_end,
                    "blue", linestyle="dashed", label=f"End Velocity: {velocity_end:.4f} c")

    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Radial Distance (R☉)")
    axs[2].set_title("Electron Beam Velocity Over Time")
    axs[2].legend()

    if show_beam_width:
        # ✅ Beam Width Plot (NEW)
        axs[3].plot(beam_times, normalized_beam_widths, marker="o", linestyle="-", color="purple", label="Beam Width / Peak Frequency")
        axs[3].set_xlabel("Time (s)")
        axs[3].set_ylabel("Δf / f_max")
        axs[3].set_title("Beam Width Evolution Over Time")
        axs[3].legend()

    plt.tight_layout()
    plt.show()