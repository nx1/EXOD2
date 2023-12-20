import numpy as np
from astropy.convolution import convolve

from exod.utils.logger import logger

def calc_var_img(cube):
    logger.info('Computing Variability')
    image_max    = np.nanmax(cube, axis=2)
    image_min    = np.nanmin(cube, axis=2)
    image_median = np.median(cube, axis=2)

    condition = np.nanmax((image_max - image_median, image_median - image_min)) / image_median

    var_img = np.where(image_median > 0,
                       condition,
                       image_max)
    return var_img

def convolve_variability(var_img, box_size=3):
    logger.info('Convolving Variability')
    kernel = np.ones((box_size, box_size)) / box_size**2
    var_img_conv = convolve(var_img, kernel)

    # New version
    # convolved = gaussian_filter(var_img, 1)
    # convolved = np.where(var_img>0, convolved, 0)
    return var_img_conv


if __name__=='__main__':
    import os
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.colors import LogNorm
    import pandas as pd

    from exod.pre_processing.download_observations import read_observation_ids
    from exod.pre_processing.read_events_files import read_EPIC_events_file
    from exod.utils.path import data, data_processed


    obsids = read_observation_ids(data / 'observations.txt')

    for obsid in obsids[1:]:
        args = {'obsid':obsid,
                'size_arcsec':15,
                'time_interval':10000,
                'box_size':3,
                'gti_only':True,
                'min_energy':0.2,
                'max_energy':12}
    
        cube, coordinates_XY = read_EPIC_events_file(**args)
        V_mat  = calc_var_img(cube)
        V_conv = convolve_variability(V_mat, box_size=3)
    
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
        fig.suptitle(str(args))
        ax1.set_title('Variability Score')
        ax2.set_title('Convolved')
    
        m1 = ax1.imshow(V_mat, origin='lower', interpolation='none', norm=LogNorm())
        m2 = ax2.imshow(V_conv, origin='lower',interpolation='none', norm=LogNorm())
        plt.colorbar(mappable=m1, ax=ax1,fraction=0.046, pad=0.04)
        plt.colorbar(mappable=m2, ax=ax2,fraction=0.046, pad=0.04)
        plt.savefig(os.path.join(data_processed, obsid, "plot_test.png"))
        plt.show()
    
        args['size_arcsec'] = 2
        args['time_interval'] = 1000
    
        cube,coordinates_XY = read_EPIC_events_file(**args)
        V_mat  = calc_var_img(cube)
        V_conv = convolve_variability(V_mat, box_size=3)
    
        fig, (ax1,ax2) = plt.subplots(1,2)
        m1 = ax1.imshow(V_mat, origin='lower', interpolation='none', norm=LogNorm())
        m2 = ax2.imshow(V_conv, origin='lower', interpolation='none', norm=LogNorm())
        plt.colorbar(mappable=m1, ax=ax1)
        plt.colorbar(mappable=m2, ax=ax2)
        plt.savefig(os.path.join(data_processed, obsid, "plot_test_gti.png"))
        plt.show()
