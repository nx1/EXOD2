import numpy as np
from scipy.ndimage import gaussian_filter
from astropy.convolution import convolve

def compute_pixel_variability(cube):
    image_max = np.max(cube, axis=2)
    image_min = np.min(cube, axis=2)
    image_median = np.median(cube, axis=2)

    V_mat = np.where(image_median > 0, np.max((image_max - image_median, image_median - image_min)) / image_median,
                     image_max)
    return V_mat

def convolve_variability(cube, box_size=3):
    V_mat = compute_pixel_variability(cube)
    #Old version
    # k = np.ones((box_size,box_size))/(box_size**2)
    # convolved = convolve(V_mat, k)

    #New version
    convolved = gaussian_filter(V_mat, 1)
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

    fig, (ax1,ax2) = plt.subplots(1,2)
    cube,coordinates_XY = read_EPIC_events_file('0831790701', 2, 1000,3,
                                                gti_only=False, emin=0.2, emax=2)
    m1=ax1.imshow(compute_pixel_variability(cube).T,origin='lower',interpolation='none', norm=LogNorm())
    plt.colorbar(mappable=m1, ax=ax1,fraction=0.046, pad=0.04)
    m2=ax2.imshow(convolve_variability(cube, box_size=3).T,origin='lower',interpolation='none', norm=LogNorm())
    ax1.axis('off')
    ax2.axis('off')
    plt.colorbar(mappable=m2, ax=ax2,fraction=0.046, pad=0.04)
    plt.savefig(os.path.join(data_processed,'0831790701', "plot_test.png"))

    # fig, (ax1,ax2) = plt.subplots(1,2)
    # cube,coordinates_XY = read_EPIC_events_file('0831790701', 2, 1000,3, gti_only=True)
    # m1=ax1.imshow(compute_pixel_variability(cube).T,origin='lower',interpolation='none', norm=LogNorm())
    # plt.colorbar(mappable=m1, ax=ax1)
    # m2=ax2.imshow(convolve_variability(cube, box_size=3).T,origin='lower',interpolation='none', norm=LogNorm())
    # plt.colorbar(mappable=m2, ax=ax2)
    # plt.savefig(os.path.join(data_processed,'0831790701', "plot_test_gti.png"))