from astropy.table import Table, vstack
from scipy.stats import binned_statistic_dd
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os
from exod.pre_processing.read_events_files import read_EPIC_events_file
from exod.utils.path import data_processed
from cv2 import inpaint, INPAINT_NS
from scipy.stats import poisson
from tqdm import tqdm

import logging
logging.getLogger('matplotlib.font_manager').disabled = True

def create_fake_burst(cubeshape, time_interval, time_peak_fraction, position, width_time, amplitude, size_arcsec):
    time = np.arange(0, cubeshape[-1])
    peak_data = np.zeros(cubeshape, dtype=int)
    time_peak = time_peak_fraction*len(time)
    width_bins = width_time / time_interval

    #Poissonian PSF
    sigma_2d = (6.6 / size_arcsec)# * 2.355  # 6.6" FWHM, 4.1" per pixel, so (6.6/4.1) pixel FWHM, 2.355 to convert FWHM in sigma
    for x in range(int(position[0]-10*sigma_2d), int(position[0]+10*sigma_2d)):
        for y in range(int(position[1]-10*sigma_2d), int(position[1]+10*sigma_2d)):
            sqdist = (x-position[0])**2+(y-position[1])**2
            psf = (1/(2*np.pi*np.sqrt(sigma_2d)))*np.exp(-(sqdist)/(2*(sigma_2d**2)))
            peak_data[x,y]+=np.random.poisson(psf*(amplitude*time_interval)*np.exp(-(time-time_peak)**2/(2*(width_bins**2))))
    #peak_data = convolve(peak_data, np.ones((3,3,1), dtype=np.int64),mode='constant', cval=0.0)
    return peak_data

size_arcsec=20
time_interval=100
cube, coordinates_XY = read_EPIC_events_file('0831790701', size_arcsec, time_interval, gti_only=False)

def run_computation(cube):
    image = np.sum(cube, axis=2)
    threshold=np.nanpercentile(image.flatten(), 99)
    condition = np.where(image>threshold)
    mask = np.zeros(image.shape)
    mask[condition]=1
    mask=np.uint8(mask[:,:,np.newaxis])
    no_source_image = inpaint(image.astype(np.float32)[:,:,np.newaxis], mask, 5, flags=INPAINT_NS)
    source_only_image = image-no_source_image

    maxi = max(np.nanmax(image), np.nanmax(no_source_image))
    mini = min(np.nanmin(image), np.nanmin(no_source_image))
    mini=max(mini,1)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    m1 = ax1.imshow(image.T, origin='lower', interpolation='none', norm=LogNorm(mini, maxi))
    plt.colorbar(mappable=m1, ax=ax1)
    m2 = ax2.imshow(no_source_image.T, origin='lower', interpolation='none', norm=LogNorm(mini, maxi))
    plt.colorbar(mappable=m2, ax=ax2)
    plt.savefig(os.path.join(data_processed, '0831790701', "background_test.png"))

    normalized_background = no_source_image/np.nansum(no_source_image)
    print(np.where(image<threshold, cube[:,:,3],0).shape)
    lightcurve_no_source_image = [np.sum(np.where(image<threshold, cube[:,:,i],0), axis=(0,1)) for i in range(cube.shape[2])]
    print(lightcurve_no_source_image)
    plt.figure()
    plt.plot(lightcurve_no_source_image)
    plt.savefig(os.path.join(data_processed, '0831790701', "lightcurve_background_test.png"))
    extrapolated_images = [normalized_background*image_value + source_only_image/(cube.shape[2])
                           for image_value in lightcurve_no_source_image]

    for ind, extrapolated_image in zip(range(len(extrapolated_images)), extrapolated_images):
        true_image = cube[:,:,ind]
        maxi = max(np.nanmax(true_image), np.nanmax(extrapolated_image))
        mini = min(np.nanmin(true_image), np.nanmin(extrapolated_image))
        mini = max(mini, 1)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        m1 = ax1.imshow(true_image.T, origin='lower', interpolation='none', norm=LogNorm(mini, maxi))
        plt.colorbar(mappable=m1, ax=ax1,fraction=0.046, pad=0.04)
        ax1.set_title("True image")
        ax1.axis('off')
        m2 = ax2.imshow(extrapolated_image.T, origin='lower', interpolation='none', norm=LogNorm(mini, maxi))
        plt.colorbar(mappable=m2, ax=ax2,fraction=0.046, pad=0.04)
        ax2.set_title("Extrapolated image")
        ax2.axis('off')
        m3= ax3.imshow(-poisson.logpmf(true_image, extrapolated_image).T, origin='lower', interpolation='none', norm=LogNorm())
        plt.colorbar(mappable=m3, ax=ax3,fraction=0.046, pad=0.04)
        ax3.set_title("Poisson likelihood")
        ax3.axis('off')
        fig.tight_layout()
        plt.savefig(os.path.join(data_processed, '0831790701', f"Test/background_test_{ind}.png"))


def calibrate_result_amplitude(tab_amplitude, cube_raw, time_interval,time_peak_fraction,peak_width, position):
    tab_likelihoods_peak=[]
    tab_percentiles = []
    tab_counts = []
    percentiles = (50,84,95,99,99.99)
    plt.figure()
    for amplitude in tqdm(tab_amplitude):
        cube_end = cube_raw + create_fake_burst(cube_raw.shape, time_interval, time_peak_fraction=time_peak_fraction,
                                       position=position,
                                       width_time=peak_width, amplitude=amplitude, size_arcsec=size_arcsec)
        (x,y)=position
        tab_counts.append(cube_end[x,y,int(time_peak_fraction*cube.shape[2])])
        image = np.sum(cube_end, axis=2)
        threshold = np.nanpercentile(image.flatten(), 99)
        condition = np.where(image > threshold)
        mask = np.zeros(image.shape)
        mask[condition] = 1
        mask = np.uint8(mask[:, :, np.newaxis])
        no_source_image = inpaint(image.astype(np.float32)[:, :, np.newaxis], mask, 5, flags=INPAINT_NS)
        source_only_image = image - no_source_image

        normalized_background = no_source_image / np.nansum(no_source_image)
        lightcurve_no_source_image = [np.sum(np.where(image < threshold, cube[:, :, i], 0), axis=(0, 1)) for i in
                                      range(cube.shape[2])]
        extrapolated_images = np.array([normalized_background * image_value + source_only_image / (cube.shape[2])
                               for image_value in lightcurve_no_source_image]).transpose(1, 2, 0)
        log_likelihoods=-poisson.logpmf(cube_end[x,y], extrapolated_images[x,y])
        log_likelihoods_all_image=-poisson.logpmf(cube_end, extrapolated_images)
        tab_percentiles.append(np.nanpercentile(log_likelihoods_all_image[:,:,int(time_peak_fraction*cube.shape[2])],
                                                percentiles, axis=(0,1)))
        tab_likelihoods_peak.append(log_likelihoods[int(time_peak_fraction*cube.shape[2])])
        plt.plot(log_likelihoods, label=amplitude)
    tab_percentiles=np.array(tab_percentiles)
    plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(data_processed, '0831790701', f"Calibration_peak_test.png"))

    fig, (ax1,ax2) = plt.subplots(2,1)
    ax1.plot(tab_amplitude, tab_likelihoods_peak)
    for i in range(len(percentiles)):
        ax1.plot(tab_amplitude, tab_percentiles[:,i], label=percentiles[i])
    ax1.loglog()
    ax1.set_xlabel('Peak amplitude')
    ax1.set_ylabel('Peak likelihood')
    ax1.legend()
    ax2.plot(tab_amplitude, tab_counts)
    ax2.loglog()
    plt.savefig(os.path.join(data_processed, '0831790701', f"Calibration_peak.png"))


calibrate_result_amplitude(np.geomspace(1e-2,1e1,25),
                           cube, time_interval,0.5,500, (25,25))


