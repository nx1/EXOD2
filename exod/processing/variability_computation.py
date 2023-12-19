import numpy as np
from scipy.ndimage import gaussian_filter
from astropy.convolution import convolve

def compute_pixel_variability(cube):
    image_max = np.nanmax(cube, axis=2)
    image_min = np.nanmin(cube, axis=2)
    image_median = np.nanmedian(cube, axis=2)
    V_mat = np.where(image_median > 0, np.nanmax((image_max - image_median, image_median - image_min)) / image_median,
                     image_max)
    return V_mat

def convolve_variability_box(V_mat, box_size=3):
    k = np.ones((box_size,box_size))/(box_size**2)
    convolved = convolve(V_mat, k)
    return convolved

def convolve_variability_GaussianBlur(V_mat, sigma=1):
    convolved = gaussian_filter(V_mat, sigma)
    convolved = np.where(V_mat>0, convolved, 0)
    return convolved

if __name__=='__main__':
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    from exod.pre_processing.read_events_files import read_EPIC_events_file
    from exod.utils.path import data_processed
    from matplotlib.colors import LogNorm
    import os

    # fig, (ax1,ax2) = plt.subplots(1,2)
    # cube,coordinates_XY = read_EPIC_events_file('0831790701', 15, 1000,3,
    #                                             gti_only=True, emin=0.2, emax=12)
    # m1=ax1.imshow(compute_pixel_variability(cube).T,origin='lower',interpolation='none', norm=LogNorm())
    # plt.colorbar(mappable=m1, ax=ax1,fraction=0.046, pad=0.04)
    # m2=ax2.imshow(convolve_variability_box(cube, box_size=3).T,origin='lower',interpolation='none', norm=LogNorm())
    # ax1.axis('off')
    # ax2.axis('off')
    # plt.colorbar(mappable=m2, ax=ax2,fraction=0.046, pad=0.04)
    # plt.savefig(os.path.join(data_processed,'0831790701', "plot_test.png"))
    #
    # fig, (ax1,ax2) = plt.subplots(1,2)
    # cube,coordinates_XY = read_EPIC_events_file('0831790701', 2, 1000,3, gti_only=True)
    # m1=ax1.imshow(compute_pixel_variability(cube).T,origin='lower',interpolation='none', norm=LogNorm())
    # plt.colorbar(mappable=m1, ax=ax1)
    # m2=ax2.imshow(convolve_variability_box(cube, box_size=3).T,origin='lower',interpolation='none', norm=LogNorm())
    # plt.colorbar(mappable=m2, ax=ax2)
    # plt.savefig(os.path.join(data_processed,'0831790701', "plot_test_gti.png"))

    fig, (ax1,ax2, ax3) = plt.subplots(1,3)
    cube,coordinates_XY,rejected = read_EPIC_events_file('0831790701', 15, 100,3,
                                                gti_only=True, emin=0.2, emax=12)
    image_max = np.nanmax(cube, axis=2)
    image_min = np.nanmin(cube, axis=2)
    image_mean = np.nanmean(cube, axis=2)
    image_median = np.nanmedian(cube, axis=2)
    ax1.imshow(image_median, norm=LogNorm(), interpolation='none')
    m2=ax2.imshow(np.where(image_median>0,(image_max - image_mean)/image_mean, image_max), norm=LogNorm(), interpolation='none')
    plt.colorbar(ax=ax2,mappable=m2)
    ax3.imshow((image_max), norm=LogNorm(), interpolation='none')
    plt.savefig(os.path.join(data_processed,'0831790701', "plot_test_median.png"))
    plt.close()

    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    cube,coordinates_XY,rejected = read_EPIC_events_file('0831790701', 10, 10,3,
                                                gti_only=True, emin=0.2, emax=12)
    V_raw = compute_pixel_variability(cube)
    convolved_old = convolve_variability_box(cube,3)
    convolved_new = convolve_variability_GaussianBlur(cube, 1)
    maxi, mini = np.nanmax((V_raw, convolved_old, convolved_new)), np.nanmin((V_raw, convolved_old, convolved_new))
    ax1.imshow(V_raw, norm=LogNorm(1,maxi), interpolation='none')
    ax2.imshow(convolved_old, norm=LogNorm(1,maxi), interpolation='none')
    ax3.imshow(convolved_new, norm=LogNorm(1,maxi), interpolation='none')
    plt.savefig(os.path.join(data_processed,'0831790701', "plot_test_variabilitymaps.png"))
    plt.close()