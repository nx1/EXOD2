import numpy as np


def create_fake_burst(data_cube, x_pos, y_pos, time_peak_fraction, width_time, amplitude):
    """
    data_cube    : DataCube() Object
    x_pos, y_pos : position of the burst
    time_peak_fraction : 0.0 - 1.0
    width_time : how long the burst lasts in seconds
    amplitude  : Height of burst

    Returns
    -------
    peak_data : same size as data_cube with burst data.
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
    x_lo, x_hi = int(x_pos-r*sigma_2d), int(x_pos+r*sigma_2d)
    y_lo, y_hi = int(y_pos-r*sigma_2d), int(y_pos+r*sigma_2d)

    for x in range(x_lo, x_hi):
        for y in range(y_lo, y_hi):
            dist_sq    = (x - x_pos)**2 + (y - y_pos)**2 # Distance^2 from burst center
            psf        = 1 / (2*np.pi*np.sqrt(sigma_2d)) * np.exp(-(dist_sq) / (2*(sigma_2d**2)))
            bin_means  = psf * amplitude * data_cube.time_interval * np.exp(-(time-time_peak)**2 / (2*(width_bins**2)))
            bin_values = np.random.poisson(lam=bin_means)
            # print(x, y, dist_sq, psf)
            # print(bin_means, bin_values)
            peak_data[x,y] += bin_values
    #peak_data = convolve(peak_data, np.ones((3,3,1), dtype=np.int64),mode='constant', cval=0.0)
    return peak_data

def create_fake_eclipse(data_cube, x_pos, y_pos, time_peak_fraction, width_time, amplitude, constant_level):
    """
    data_cube    : DataCube() Object
    x_pos, y_pos : position of the burst
    time_peak_fraction : 0.0 - 1.0
    width_time : how long the eclipse lasts in seconds
    amplitude  : Height of burst
    constant_level : level of the constant emission before eclipse. Should be larger than amplitude

    Returns
    -------
    peak_data : same size as data_cube with eclipse data.
    """

    time       = np.arange(0, data_cube.shape[2])
    eclipse_data  = np.zeros(data_cube.shape, dtype=int)
    time_peak  = time_peak_fraction * len(time)
    width_bins = width_time / data_cube.time_interval

    #Poissonian PSF
    # 6.6" FWHM, 4.1" per pixel, so (6.6/4.1) pixel FWHM, 2.355 to convert FWHM in sigma but seems too much
    sigma_2d = (6.6 / data_cube.size_arcsec) # * 2.355

    # Pixels to iterate over
    r = 10 # Size of box to calculate values
    x_lo, x_hi = int(x_pos-r*sigma_2d), int(x_pos+r*sigma_2d)
    y_lo, y_hi = int(y_pos-r*sigma_2d), int(y_pos+r*sigma_2d)

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
