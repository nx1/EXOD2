import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from exod.processing.data_cube import DataCubeXMM
from exod.xmm.event_list import EventList


def create_fake_burst(data_cube, x_pos, y_pos, time_peak_fraction, width_time, amplitude):
    """
    Create a fake burst in the data_cube at position x_pos, y_pos, at time_peak_fraction of the time axis.

    Parameters:
        data_cube (DataCube): DataCube() Object.
        x_pos (int): x position of the burst.
        y_pos (int): y position of the burst.
        time_peak_fraction (float): 0.0 - 1.0.
        width_time (float): how long the burst lasts in seconds.
        amplitude (int): Height of burst.

    Returns:
        peak_data (np.array): Same size as data_cube with burst data.
    """

    time       = np.arange(0, data_cube.shape[2])
    peak_data  = np.zeros(data_cube.shape, dtype=int)
    time_peak  = time_peak_fraction * len(time)
    width_bins = width_time / data_cube.time_interval

    #Poissonian PSF
    # 6.6" FWHM, 4.1" per pixel, so (6.6/4.1) pixel FWHM, 2.355 to convert FWHM in sigma but seems too much
    sigma_2d = (6.6 / data_cube.size_arcsec) # * 2.355

    # Pixels to iterate over
    r = 10 # Size of box to calculate values
    x_lo, x_hi = max(0,int(x_pos-r*sigma_2d)), min(int(x_pos+r*sigma_2d), peak_data.shape[0]-1)
    y_lo, y_hi = max(0,int(y_pos-r*sigma_2d)), min(int(y_pos+r*sigma_2d), peak_data.shape[1]-1)

    for x in range(x_lo, x_hi):
        for y in range(y_lo, y_hi):
            dist_sq    = (x - x_pos)**2 + (y - y_pos)**2 # Distance^2 from burst center
            psf        = 1 / (2*np.pi*(sigma_2d**2)) * np.exp(-(dist_sq) / (2*(sigma_2d**2)))
            bin_means  = psf * amplitude * data_cube.time_interval * np.exp(-(time-time_peak)**2 / (2*(width_bins**2)))
            bin_values = np.random.poisson(lam=bin_means)
            # print(x, y, dist_sq, psf)
            # print(bin_means, bin_values)
            peak_data[x,y] += bin_values
    #peak_data = convolve(peak_data, np.ones((3,3,1), dtype=np.int64),mode='constant', cval=0.0)
    return peak_data

def create_fake_onebin_burst(data_cube, x_pos, y_pos, time_peak_fraction, amplitude):
    """
    Create a fake burst in the data_cube at position x_pos, y_pos, at time_peak_fraction of the time axis.
 
    Parameters:
        data_cube (DataCube): DataCube() Object.
        x_pos (int): x position of the burst.
        y_pos (int): y position of the burst.
        time_peak_fraction (float): 0.0 - 1.0.
        amplitude (int): Height of burst.

    Returns:
        peak_data (np.array): Same size as data_cube with burst data.
    """
    time        = np.arange(0, data_cube.shape[2])
    peak_data   = np.zeros(data_cube.shape, dtype=int)
    time_index  = int(time_peak_fraction * len(time))

    # Poissonian PSF
    # 6.6" FWHM, 4.1" per pixel, so (6.6/4.1) pixel FWHM, 2.355 to convert FWHM in sigma but seems too much
    sigma_2d = (6.6 / data_cube.size_arcsec) # * 2.355

    # Pixels to iterate over
    r = 10 # Size of box to calculate values
    x_lo, x_hi = max(0, int(x_pos-r*sigma_2d)), min(int(x_pos+r*sigma_2d), peak_data.shape[0]-1)
    y_lo, y_hi = max(0, int(y_pos-r*sigma_2d)), min(int(y_pos+r*sigma_2d), peak_data.shape[1]-1)


    for x in range(x_lo, x_hi):
        for y in range(y_lo, y_hi):
            dist_sq    = (x - x_pos)**2 + (y - y_pos)**2 # Distance^2 from burst center
            psf        = (1 / (2*np.pi*(sigma_2d**2))) * np.exp(-(dist_sq) / (2*(sigma_2d**2)))
            bin_means  = psf * amplitude
            bin_values = np.random.poisson(lam=bin_means)
            # print(x, y, dist_sq, psf)
            # print(bin_means, bin_values)
            peak_data[x,y,time_index] += bin_values
    #peak_data = convolve(peak_data, np.ones((3,3,1), dtype=np.int64),mode='constant', cval=0.0)
    return peak_data


def create_fake_Nbins_burst(data_cube, x_pos, y_pos, time_peak_fractions, amplitude):
    """
    Create Many fake bursts in the data cube

    Parameters:
        data_cube (DataCube): DataCube() Object.
        x_pos (int): x position of the burst.
        y_pos (int): y position of the burst.
        time_peak_fractions (list): list of time fractions where the bursts should be placed.
        amplitude (int): Height of burst.

    Returns:
        peak_data (np.array): Same size as data_cube with burst data.
    """
    peaks_data = np.zeros(data_cube.shape, dtype=int)
    for time_fraction in time_peak_fractions:
        peaks_data += create_fake_onebin_burst(data_cube, x_pos, y_pos, time_fraction, amplitude)
    return peaks_data


def create_fake_eclipse(data_cube, x_pos, y_pos, time_peak_fraction, width_time, amplitude, constant_level):
    """
    Create a fake eclipse in the data_cube at position x_pos, y_pos, at time_peak_fraction of the time axis.

    Parameters:
        data_cube (DataCube): DataCube() Object.
        x_pos (int): x position of the burst.
        y_pos (int): y position of the burst.
        time_peak_fraction (float): 0.0 - 1.0.
        width_time (float): how long the Eclispe lasts in seconds.
        amplitude (int): Amplitude of Eclipse

    Returns:
        peak_data (np.array): Same size as data_cube with Eclipse data.
    """

    time         = np.arange(0, data_cube.shape[2])
    eclipse_data = np.zeros(data_cube.shape, dtype=int)
    time_peak    = time_peak_fraction * len(time)
    width_bins   = width_time / data_cube.time_interval

    #Poissonian PSF
    # 6.6" FWHM, 4.1" per pixel, so (6.6/4.1) pixel FWHM, 2.355 to convert FWHM in sigma but seems too much
    sigma_2d = (6.6 / data_cube.size_arcsec) * 2.355

    # Pixels to iterate over
    r = 10 # Size of box to calculate values
    x_lo, x_hi = max(0, int(x_pos-r*sigma_2d)), min(int(x_pos+r*sigma_2d), eclipse_data.shape[0]-1)
    y_lo, y_hi = max(0, int(y_pos-r*sigma_2d)), min(int(y_pos+r*sigma_2d), eclipse_data.shape[1]-1)

    for x in range(x_lo, x_hi):
        for y in range(y_lo, y_hi):
            dist_sq    = (x - x_pos)**2 + (y - y_pos)**2 # Distance^2 from burst center
            psf        = 1 / (2*np.pi*np.sqrt(sigma_2d)) * np.exp(-(dist_sq) / (2*(sigma_2d**2)))
            bin_means  = psf * data_cube.time_interval * (constant_level - amplitude * np.exp(-(time-time_peak)**2 / (2*(width_bins**2))))
            bin_values = np.random.poisson(lam=bin_means)
            # print(x, y, dist_sq, psf)
            # print(bin_means, bin_values)
            eclipse_data[x,y] += bin_values
    #peak_data = convolve(peak_data, np.ones((3,3,1), dtype=np.int64),mode='constant', cval=0.0)
    return eclipse_data




def create_multiple_fake_eclipses(data_cube, tab_x_pos, tab_y_pos, tab_time_peak_fraction, tab_width_time, tab_amplitude, tab_constant_level):
    """
    Create Many fake eclipses in the data cube

    Parameters:
        data_cube (DataCube): DataCube() Object.
        tab_x_pos (list): list of x positions of the Eclipses.
        tab_y_pos (list): list of y positions of the Eclipses.
        tab_time_peak_fraction (list): list of time fractions where the Eclipses should be placed.
        tab_width_time (list): list of how long the Eclipses lasts in seconds.
        tab_amplitude (list): list of amplitudes of Eclipses

    Returns:
        peak_data (np.array): Same size as data_cube with Eclipse data.
    """

    time          = np.arange(0, data_cube.shape[2])
    eclipse_data  = np.zeros(data_cube.shape, dtype=int)

    for time_peak_fraction, width_time, x_pos, y_pos, constant_level, amplitude in\
        zip(tab_time_peak_fraction,tab_width_time,tab_x_pos,tab_y_pos,tab_constant_level,tab_amplitude):
        time_peak  = time_peak_fraction * (len(time)-1)
        width_bins = width_time / data_cube.time_interval

        #Poissonian PSF
        # 6.6" FWHM, 4.1" per pixel, so (6.6/4.1) pixel FWHM, 2.355 to convert FWHM in sigma but seems too much
        sigma_2d = (6.6 / data_cube.size_arcsec) * 1.4 # * 2.355

        # Pixels to iterate over
        r = 10 # Size of box to calculate values
        x_lo, x_hi = max(0, int(x_pos - r * sigma_2d)), min(int(x_pos + r * sigma_2d), eclipse_data.shape[0] - 1)
        y_lo, y_hi = max(0, int(y_pos - r * sigma_2d)), min(int(y_pos + r * sigma_2d), eclipse_data.shape[1] - 1)
        if np.nansum(data_cube.data[x_pos,y_pos]) > 0:
            for x in range(x_lo, x_hi):
                for y in range(y_lo, y_hi):
                    dist_sq    = (x - x_pos)**2 + (y - y_pos)**2 # Distance^2 from burst center
                    psf        = 1 / (2*np.pi*np.sqrt(sigma_2d)) * np.exp(-(dist_sq) / (2*(sigma_2d**2)))
                    bin_means  = psf * data_cube.time_interval * (constant_level - amplitude * np.exp(-(time-time_peak)**2 / (2*(width_bins**2))))
                    bin_values = np.random.poisson(lam=bin_means)
                    # print(x, y, dist_sq, psf)
                    # print(bin_means, bin_values)
                    eclipse_data[x,y] += bin_values
    #peak_data = convolve(peak_data, np.ones((3,3,1), dtype=np.int64),mode='constant', cval=0.0)
    return eclipse_data


def check_synthetic_peak_counts_diff(obsid):
    """
    Evaluate differences in peak counts for synthetic bursts added to XMM-Newton data cubes.

    Parameters:
        obsid (str): XMM Observation ID
    """
    # Load observation and event data and WCS
    observation = Observation(obsid)
    observation.get_files()
    observation.get_events_overlapping_subsets()
    event_list = EventList.from_event_lists(observation.events_overlapping_subsets[0])
    event_list.read()

    # Parameters for the synthetic test
    size_arcsec = 20
    amplitudes = [0.5, 2, 5, 10]
    time_bins = [500, 1000, 2000]
    num_trials = 40

    all_res = []
    for time_bin in tqdm(time_bins):
        data_cube = DataCubeXMM(event_list=event_list, size_arcsec=size_arcsec, time_interval=time_bin)

        for amplitude in tqdm(amplitudes):
            count_differences = []

            for i in range(num_trials):
                # Random position and time fraction for the synthetic burst
                x_pos = np.random.randint(low=5, high=data_cube.data.shape[0] - 5)
                y_pos = np.random.randint(low=5, high=data_cube.data.shape[1] - 5)
                time_fraction = np.random.random()

                # Add synthetic burst to the data cube
                data_cube_with_burst_only = create_fake_onebin_burst(
                    data_cube=data_cube,
                    x_pos=x_pos,
                    y_pos=y_pos,
                    time_peak_fraction=time_fraction,
                    amplitude=amplitude * time_bin
                )

                data_cube_with_burst_added = data_cube.data + data_cube_with_burst_only
                count_difference = np.sum(data_cube_with_burst_added) - np.sum(data_cube.data)
                count_difference = int(count_difference)

                res = {'amplitude'        : amplitude,
                       'time_bin'         : time_bin,
                       'trial_num'        : i,
                       'count difference' : count_difference}
                all_res.append(res)
                count_differences.append(count_difference)

            plt.figure()
            plt.title(f"Burst Amplitude={amplitude} Time Bin={time_bin}\ntrials={num_trials}")
            plt.hist(count_differences, bins=20, label=r"$C_{data+burst} - C_{data}$")
            plt.axvline(x=amplitude * time_bin, color='red', linestyle='--', label="Amplitude*time_bin (Expected)")
            plt.axvline(x=amplitude * time_bin * 1.5, color='green', linestyle='--', label="1.5x Amplitude*time_bin")
            plt.xlabel(r"Difference in Counts")
            plt.ylabel("Frequency")
            plt.legend()
            plt.show()
    df_res = pd.DataFrame(all_res)
    return df_res




if __name__ == "__main__":
    from exod.xmm.observation import Observation
    from exod.utils.plotting import use_scienceplots
    use_scienceplots()

    obsid = '0831790701'
    observation = Observation(obsid)
    observation.get_files()
    savedir = observation.path_processed
    event_list = observation.events_processed_pn[0]
    event_list.read()
    data_cube = DataCubeXMM(event_list, size_arcsec=20, time_interval=100)

    create_fake_burst(data_cube, 10, 10, 0.5, 100, 1000)
    create_fake_onebin_burst(data_cube, 10, 10, 0.5, 1000)
    create_fake_Nbins_burst(data_cube,  10, 10, [0.5, 0.5], 1000)
    create_fake_eclipse(data_cube, 10, 10, 0.5, 100, 1000, 10000)
    create_multiple_fake_eclipses(data_cube, [10, 10], [10, 10], [0.5, 0.5], [100, 100], [1000, 1000], [10000, 10000])
    df_res = check_synthetic_peak_counts_diff(obsid)
    print(df_res)
