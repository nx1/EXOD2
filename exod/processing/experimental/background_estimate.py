import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os
from exod.pre_processing.read_events_files import read_EPIC_events_file
from exod.utils.path import data_processed
from exod.utils.synthetic_data import create_fake_burst
from cv2 import inpaint, INPAINT_NS, filter2D
from scipy.stats import poisson
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import cmasher as cmr
from photutils.detection import DAOStarFinder

import logging
logging.getLogger('matplotlib.font_manager').disabled = True

def compute_background(cube):
    image = np.sum(cube, axis=2)
    threshold=np.nanpercentile(image.flatten(), 99)
    condition = np.where(image>threshold)
    mask = np.zeros(image.shape)
    mask[condition]=1
    mask=np.uint8(mask[:,:,np.newaxis])
    no_source_image = inpaint(image.astype(np.float32)[:,:,np.newaxis], mask, 5, flags=INPAINT_NS)
    source_only_image = image-no_source_image

    normalized_background = no_source_image/np.nansum(no_source_image)

    lightcurve_no_source_image = [np.sum(np.where(image<threshold, cube[:,:,i],0), axis=(0,1)) for i in range(cube.shape[2])]
    background_images = [normalized_background*frame_value
                           for frame_value in lightcurve_no_source_image]
    background_images=np.array(background_images).transpose(1, 2, 0)
    background_withsource = [normalized_background*frame_value + source_only_image/(cube.shape[2])
                           for frame_value in lightcurve_no_source_image]
    background_withsource=np.array(background_withsource).transpose(1, 2, 0)
    return background_images, background_withsource

def compute_background_two_templates(cube, lc_HE, time_binning):
    image_total = np.sum(cube, axis=2)
    threshold=np.nanpercentile(image_total.flatten(), 95)
    condition = np.where(image_total>threshold)
    mask = np.zeros(image_total.shape)
    mask[condition]=1
    mask=np.uint8(mask[:,:,np.newaxis])

    plt.figure()
    timebins=np.arange(cube.shape[2])
    lc = np.nansum(cube,axis=(0,1))
    plt.plot(timebins, np.where(lc_HE > 1*time_binning, lc, np.nan), c='r')
    plt.plot(timebins, np.where(lc_HE < 1*time_binning, lc, np.nan), c='b')
    plt.savefig(os.path.join(data_processed, '0831790701', f"BackgroundTest/Full_image_lightcurve.png"))

    no_source_image_total = inpaint(image_total.astype(np.float32)[:,:,np.newaxis], mask, 5, flags=INPAINT_NS)
    source_only_image = image_total-no_source_image_total
    normalized_total_background = no_source_image_total/np.nansum(no_source_image_total)

    image_GTI = np.sum(cube[:,:,np.where(lc_HE<(1*time_binning))[0]], axis=2)
    no_source_image_GTI = inpaint(image_GTI.astype(np.float32)[:,:,np.newaxis], mask, 5, flags=INPAINT_NS)
    source_only_image_GTI = image_GTI-no_source_image_GTI
    normalized_GTI_background = no_source_image_GTI/np.nansum(no_source_image_GTI)

    image_BTI = np.sum(cube[:,:,np.where(lc_HE>(1*time_binning))[0]], axis=2)
    no_source_image_BTI = inpaint(image_BTI.astype(np.float32)[:,:,np.newaxis], mask, 5, flags=INPAINT_NS)
    source_only_image_BTI = image_BTI-no_source_image_BTI
    normalized_BTI_background = no_source_image_BTI/np.nansum(no_source_image_BTI)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_figwidth(10)
    fig.set_figheight(5)
    m1 = ax1.imshow(normalized_total_background, origin='lower', interpolation='none', norm=LogNorm())
    ax1.set_title('Total')
    ax1.axis("off")
    plt.colorbar(mappable=m1, ax=ax1, fraction=0.046, pad=0.04)
    m2 = ax2.imshow(normalized_GTI_background, origin='lower', interpolation='none', norm=LogNorm())
    ax2.set_title('GTI')
    ax2.axis("off")
    plt.colorbar(mappable=m2, ax=ax2, fraction=0.046, pad=0.04)
    m3 = ax3.imshow(normalized_BTI_background, origin='lower', interpolation='none', norm=LogNorm())  # vmin=-5, vmax=5, cmap='cmr.guppy')
    ax3.set_title('BTI')
    ax3.axis("off")
    plt.colorbar(mappable=m3, ax=ax3, fraction=0.046, pad=0.04)
    plt.savefig(os.path.join(data_processed, '0831790701', f"BackgroundTest/test_GTI_image.png"))


    lightcurve_no_source_image = [np.sum(np.where(image_total<threshold, cube[:,:,i], 0),
                                         axis=(0,1)) for i in range(cube.shape[2])]
    background_images = np.where(lc_HE<1*time_binning,
                                 normalized_GTI_background[:,:,np.newaxis]*lightcurve_no_source_image,
                                 normalized_BTI_background[:,:,np.newaxis]*lightcurve_no_source_image)
    #background_images = np.array(background_images).transpose(1, 2, 0)
    background_withsource = np.where(lc_HE < 1 * time_binning,
                         normalized_GTI_background[:,:,np.newaxis] * lightcurve_no_source_image
                                     +source_only_image_GTI[:,:,np.newaxis]/(cube.shape[2]),
                         normalized_BTI_background[:,:,np.newaxis] * lightcurve_no_source_image
                                     +source_only_image_BTI[:,:,np.newaxis]/(cube.shape[2]))
    #background_withsource = np.array(background_withsource).transpose(1, 2, 0)
    return background_images, background_withsource

def check_GTIvsBTI_image_structure(size_arcsec,time_interval):
    #TODO: Extract images from GTI and BTI and check for non-linearities, that might justify using two template responses
    GTI_threshold = {"pn":0.5,"M1":0.2,"M2":0.2}
    for instruments in ("pn","M1","M2"):
        HE_cube, HE_coordinates_XY = read_EPIC_events_file('0831790701', size_arcsec, time_interval, gti_only=False,
                                                           emin=10, emax=12, instr=[instruments])
        lc_HE = np.sum(HE_cube, axis=(0,1))/time_interval
        cube, coordinates_XY = read_EPIC_events_file('0831790701', size_arcsec, time_interval, gti_only=False,
                                                           emin=0.2, emax=12, instr=[instruments])
        image = np.sum(cube, axis=2)
        threshold=np.nanpercentile(image.flatten(), 99)
        cube = np.array([np.where(image<threshold, frame, np.nan) for frame in cube.transpose(2,0,1)]).transpose(1,2,0)
        plt.figure()
        plt.imshow(np.sum(cube, axis=2), origin='lower', interpolation='none', norm=LogNorm())
        plt.savefig(os.path.join(data_processed, '0831790701', f"BackgroundTest/test_image_nosource_{instruments}.png"))
        lc = np.nansum(cube, axis=(0,1))/time_interval
        timebins=np.arange(cube.shape[2])
        plt.figure()
        plt.plot(timebins, np.where(lc_HE>GTI_threshold[instruments], lc, np.nan), c='r')
        plt.plot(timebins, np.where(lc_HE<GTI_threshold[instruments], lc, np.nan), c='b')
        plt.savefig(os.path.join(data_processed, '0831790701', f"BackgroundTest/test_GTI_{instruments}.png"))

        fig, (ax1,ax2,ax3) = plt.subplots(1,3)
        fig.set_figwidth(15)
        fig.set_figheight(5)
        GTI_image = np.nansum(cube[:,:,lc_HE<GTI_threshold[instruments]], axis=(2))
        BTI_image = np.nansum(cube[:,:,lc_HE>GTI_threshold[instruments]], axis=(2))
        m1 = ax1.imshow(GTI_image, origin='lower', interpolation='none', norm=LogNorm())
        ax1.set_title('Good Time Interval')
        plt.colorbar(mappable=m1, ax=ax1,fraction=0.046, pad=0.04)
        m2 = ax2.imshow(BTI_image, origin='lower', interpolation='none', norm=LogNorm())
        ax2.set_title('Bad Time Interval')
        plt.colorbar(mappable=m2, ax=ax2,fraction=0.046, pad=0.04)
        m3 = ax3.imshow((BTI_image/np.nansum(BTI_image))/(GTI_image/np.nansum(GTI_image))
                        , origin='lower', interpolation='none',norm=LogNorm())# vmin=-5, vmax=5, cmap='cmr.guppy')
        ax3.set_title('BTI/GTI')
        plt.colorbar(mappable=m3, ax=ax3,fraction=0.046, pad=0.04)
        plt.savefig(os.path.join(data_processed, '0831790701', f"BackgroundTest/test_GTI_image_{instruments}.png"))
#check_GTIvsBTI_image_structure(10,100)

def run_computation(cube_raw, with_peak=False, size_arcsec=15):
    if with_peak:
        cube = cube_raw+create_fake_burst(cube_raw.shape, 1000, time_peak_fraction=0.05,
                                       position=(0.41*cube_raw.shape[0],0.36*cube_raw.shape[1]),
                                       width_time=50, amplitude=1e1, size_arcsec=size_arcsec)
    else:
        cube = cube_raw
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
    maxi_value = np.max(np.sum(cube, axis=(0,1)))/(cube.shape[0]*cube.shape[1])


    fig, (ax1, ax2) = plt.subplots(1, 2)
    m1 = ax1.imshow(image.T, origin='lower', interpolation='none', norm=LogNorm(mini, maxi))
    plt.colorbar(mappable=m1, ax=ax1)
    m2 = ax2.imshow(no_source_image.T, origin='lower', interpolation='none', norm=LogNorm(mini, maxi))
    plt.colorbar(mappable=m2, ax=ax2)
    plt.savefig(os.path.join(data_processed, '0831790701', "background_test.png"))

    normalized_background = no_source_image/np.nansum(no_source_image)
    lightcurve_no_source_image = [np.sum(np.where(image<threshold, cube[:,:,i],0), axis=(0,1)) for i in range(cube.shape[2])]
    plt.figure()
    plt.plot(lightcurve_no_source_image)
    plt.savefig(os.path.join(data_processed, '0831790701', "lightcurve_background_test.png"))
    extrapolated_images = [normalized_background*frame_value + source_only_image/(cube.shape[2])
                           for frame_value in lightcurve_no_source_image]

    for ind, extrapolated_image in zip(range(len(extrapolated_images)), extrapolated_images):
        true_image = cube[:,:,ind]
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.set_figheight(5)
        fig.set_figwidth(15)
        m1 = ax1.imshow(extrapolated_image.T, origin='lower', interpolation='none', norm=LogNorm(1, maxi_value))
        plt.colorbar(mappable=m1, ax=ax1,fraction=0.046, pad=0.04)
        ax1.set_title("Expected image")
        ax1.axis('off')
        m2 = ax2.imshow(true_image.T, origin='lower', interpolation='none', norm=LogNorm(1, maxi_value))
        plt.colorbar(mappable=m2, ax=ax2,fraction=0.046, pad=0.04)
        ax2.set_title("True image")
        ax2.axis('off')
        likelihood_map = -poisson.logpmf(true_image, extrapolated_image).T
        sigma= 1
        blurred_variability = gaussian_filter(likelihood_map,sigma)#, mode='constant',cval=0)
        m3= ax3.imshow(blurred_variability, origin='lower', interpolation='none', norm=LogNorm(1,10))
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
    tab_background_of_peak=[]
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
        tab_background_of_peak.append(extrapolated_images[x,y,int(time_peak_fraction*cube.shape[2])])
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
    ax2.plot(tab_amplitude, tab_background_of_peak, c='k',ls='--')
    ax2.loglog()
    plt.savefig(os.path.join(data_processed, '0831790701', f"Calibration_peak.png"))

if __name__=="__main__":
    size_arcsec=10
    time_interval=1000
    # cube, coordinates_XY = read_EPIC_events_file('0831790701', size_arcsec, time_interval, gti_only=False)
    # cube_peak = cube+create_fake_burst(cube.shape, 1000, time_peak_fraction=0.05,
    #                                        position=(0.21*cube.shape[0],0.26*cube.shape[1]),
    #                                        width_time=5000, amplitude=1e2, size_arcsec=20)
    # plt.figure(figsize=(15,15))
    # plt.imshow(np.nansum(cube_peak, axis=2), norm=LogNorm(), interpolation='none')
    # plt.savefig(os.path.join(data_processed, '0831790701', f"TestBurst.png"))
    # run_computation(cube, with_peak=True, size_arcsec=size_arcsec)

    # calibrate_result_amplitude(np.geomspace(1e-2,1e1,25),
    #                            cube, time_interval,0.5,500, (25,25))
    cubeHE, coordinates_XY = read_EPIC_events_file('0831790701', 30, time_interval,
                                                   gti_only=False, emin=10, emax=12)
    lc_HE = np.sum(cubeHE,axis=(0,1))
    cube, coordinates_XY = read_EPIC_events_file('0831790701', 10, time_interval,
                                                 gti_only=False, emin=0.2, emax=12)
    background_images_new, background_withsourc_new=compute_background_two_templates(cube, lc_HE, time_interval)
    background_images, background_withsource = compute_background(cube)
    maxi_value = np.max(np.sum(cube, axis=(0,1)))/(cube.shape[0]*cube.shape[1])
    for i in tqdm(range(background_images.shape[2])):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.set_figheight(5)
        fig.set_figwidth(15)
        m1 = ax1.imshow(background_withsource[:,:,i].T, origin='lower', interpolation='none', norm=LogNorm(1,maxi_value))
        plt.colorbar(mappable=m1, ax=ax1,fraction=0.046, pad=0.04)
        ax1.set_title("Expected image (old)")
        ax1.axis('off')
        m2 = ax2.imshow(background_withsourc_new[:,:,i].T, origin='lower', interpolation='none', norm=LogNorm(1, maxi_value))
        plt.colorbar(mappable=m2, ax=ax2,fraction=0.046, pad=0.04)
        ax2.set_title("Expected image (new)")
        ax2.axis('off')
        m3 = ax3.imshow(cube[:,:,i].T, origin='lower', interpolation='none', norm=LogNorm(1, maxi_value))
        plt.colorbar(mappable=m3, ax=ax3,fraction=0.046, pad=0.04)
        ax3.set_title("True image")
        ax3.axis('off')
        plt.savefig(os.path.join(data_processed, '0831790701', f"BackgroundTest/background_test_{i}.png"))

