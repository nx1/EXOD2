
from exod.pre_processing.read_events import get_image_files
from exod.utils.path import data_processed, data_results
from exod.utils.logger import logger

import os
import cmasher as cmr
import numpy as np
from astropy.convolution import convolve
from astropy.stats import sigma_clip
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from matplotlib.colors import LogNorm
from astropy.wcs import WCS
from astropy.io import fits
from scipy.stats import kstest, poisson
from skimage.measure import label, regionprops, regionprops_table


def calc_var_img(cube):
    """
    Calculate the variability image from a data cube.
    """
    logger.info('Computing Variability')
    image_max    = np.nanmax(cube, axis=2)
    image_min    = np.nanmin(cube, axis=2)
    image_median = np.median(cube, axis=2)

    condition = np.nanmax((image_max - image_median, image_median - image_min)) / image_median

    var_img = np.where(image_median > 0,
                       condition,
                       image_max)
    return var_img


def conv_var_img(var_img, box_size=3):
    """
    Convolve the variability image with a box kernel.
    """
    logger.info('Convolving Variability')
    kernel = np.ones((box_size, box_size)) / box_size**2
    var_img_conv = convolve(var_img, kernel)

    # New version
    # convolved = gaussian_filter(var_img, 1)
    # convolved = np.where(var_img>0, convolved, 0)
    return var_img_conv


def extract_var_regions(var_img):
    """
    Use skimage to obtain the contiguous regions in the variability image.

    The threshold is calculated using an interative sigma clip.

    We currently used the weighted centroid which calculates a centroid based on
    both the size of the bounding box and the values of the pixels themselves.

    Parameters
    ----------
    var_img : Variability image (2D)

    Returns
    -------
    df_regions : DataFrame of Detected Regions
    """
    logger.info('Extracting variable Regions')

    v_arr = var_img.flatten()

    v_filt, lo, hi = sigma_clip(
        v_arr,
        sigma=3,
        sigma_lower=None,
        sigma_upper=None,
        maxiters=5,
        cenfunc='median',
        stdfunc='std',
        axis=None,
        masked=True,
        return_bounds=True,
        copy=True,
        grow=False
    )

    threshold = hi
    logger.info(f'threshold: {threshold}')
    var_img_mask = var_img > threshold

    plt.figure(figsize=(10,3))
    plt.title('Thresholding')
    plt.plot(v_arr, label='Variability')
    plt.axhline(threshold, color='red', label=f'threshold={threshold:.2f}')
    plt.legend()
    plt.ylabel('V score')

    # Use the skimage label function to label connected components
    # The result is an array that has the number of the region
    # in the position of the variable regions
    # 0011020
    # 0010220
    # 0000000
    var_img_mask_labelled = label(var_img_mask)

    # Obtain the region properties for the detected regions.
    properties_ = ('label', 'bbox', 'weighted_centroid', 'intensity_mean', 'equivalent_diameter_area')
    region_dict = regionprops_table(label_image=var_img_mask_labelled,
                                    intensity_image=var_img,
                                    properties=properties_)
    df_regions = pd.DataFrame(region_dict)

    logger.info(f'region_table:\n{df_regions}')
    return df_regions


def filter_df_regions(df_regions):
    return df_regions

def plot_var_with_regions(var_img, df_regions, outfile):
    """
    Plot the variability image with the bounding boxes of the detected regions.

    Parameters
    ----------
    var_img    : np.ndarray : Variability Image  (or Likelihood Image)
    df_regions : pd.DataFrame : from extract_variability_regions
    outfile    : str : Path to save the figure to
    """
    logger.info('Plotting Variability map with source regions')

    fig, ax = plt.subplots(figsize=(8,8))
    cmap = plt.cm.hot
    cmap.set_bad('black')
    m1 = ax.imshow(var_img.T,
                   norm=LogNorm(),
                   interpolation='none',
                   origin='lower',
                   cmap=cmap)
    cbar = plt.colorbar(mappable=m1, ax=ax)
    cbar.set_label("Variability")
    ax.scatter(df_regions['weighted_centroid-0'], df_regions['weighted_centroid-1'], marker='+', s=10, color='white')
    for i, row in df_regions.iterrows():
        ind     = row['label']
        x_cen = row['weighted_centroid-0']
        y_cen = row['weighted_centroid-1']

        width  = row['bbox-2'] - row['bbox-0']
        height = row['bbox-3'] - row['bbox-1']

        x_pos = x_cen - width / 2
        y_pos = y_cen - height / 2

        rect = patches.Rectangle(xy=(x_pos, y_pos),
                                 width=width,
                                 height=height,
                                 linewidth=1,
                                 edgecolor='w',
                                 facecolor='none')

        plt.text(x_pos+width, y_pos+height, str(ind), c='w')
        ax.add_patch(rect)
    logger.info(f'Saving Variability image to: {outfile}')
    plt.savefig(outfile)
    # plt.show()


def get_regions_sky_position(obsid, coordinates_XY, df_regions):
    """
    Calculate the sky position of the detected regions.

    Erwan, gonna need you to run me through this one lol.

    One thing I see is that you use the WCS from one of three
    image files that is already pre-binned to 80 pixels, I feel
    uneasy about this...

    Parameters
    ----------
    obsid : str : Observation ID
    coordinates_XY :
    df_regions

    Returns
    -------

    """
    logger.info('Getting sky positions of regions')

    img_files = get_image_files(obsid)
    img_file = img_files[0] # Use the first image found.

    logger.info(f'Opening fits file: {img_file}')
    f = fits.open(img_file)

    logger.info('Creating WCS from header')
    header = f[0].header
    w = WCS(header)
    logger.info(w)

    # Watch out for this move: to know the EPIC X and Y coordinates of the variable sources, we use the final coordinates
    # in the variability map, which are not integers. To know to which X and Y correspond to this, we interpolate the
    # values of X and Y on the final coordinates. We divide by 80 because the WCS from the image is binned by x80
    # compared to X and Y values
    logger.info(f'Getting X and Y interpolation limits')
    logger.warning(f'This is assuming an 80 binning, this may not be the case for all cameras?')
    interpX = interp1d(range(len(coordinates_XY[0])), coordinates_XY[0]/80)
    interpY = interp1d(range(len(coordinates_XY[1])), coordinates_XY[1]/80)
    logger.info(f'interpX={interpX}, interpY={interpY}')

    all_res = []
    for i, row in df_regions.iterrows():
        X = interpX(row['weighted_centroid-0'])
        Y = interpY(row['weighted_centroid-1'])
        pos = w.pixel_to_world(X, Y)

        res = {'X'   : X,
               'Y'   : Y,
               'ra'  : pos.ra.value,
               'dec' : pos.dec.value}
        all_res.append(res)
            
    df_sky = pd.DataFrame(all_res)
    logger.info(f'df_sky:\n{df_sky}')
    return df_sky


def get_region_lightcurves(cube, df_regions):
    """
    Extract the lightcurves from the variable regions found.
    Returns
    -------
    lcs : List of Lightcurves
    """
    logger.info("Extracting lightcurves from data cube")
    lcs = []
    for i, row in df_regions.iterrows():
        xlo, xhi = row['bbox-0'], row['bbox-2']
        ylo, yhi = row['bbox-1'], row['bbox-3']
        data = cube[xlo:xhi, ylo:yhi]
        lc = np.sum(data, axis=(0,1), dtype=np.int32)
        lcs.append(lc)
    return lcs


def calc_KS_poission(lc):
    """
    Calculate the KS Probability assuming the lightcurve was
    created from a possion distribution with the mean of the lightcurve.
    """
    #logger.info("Calculating KS Probability, assuming a Poission Distribution")
    lc_mean = mean_of_poisson = np.nanmean(lc)
    N_data = len(lc)
    result = kstest(lc, [lc_mean] * N_data)
    expected_distribution = poisson(mean_of_poisson)
    ks_res = kstest(lc, expected_distribution.cdf)
    logger.debug(f'KS_prob: lc_mean = {lc_mean}, N_data = {N_data}\nks_res = {ks_res}')
    return ks_res


def plot_region_lightcurves(lcs, df_regions, obsid):
    N_poission_realisations = 5000
    color = cmr.take_cmap_colors(cmap='cmr.ocean', N=1, cmap_range=(0.3, 0.3))[0]
    logger.info(f'Plotting regions lightcurves, using {N_poission_realisations} Poission realisations for errors')

    for i, row in df_regions.iterrows():
        label = row['label']
        lc = lcs[i]
        lc_mean = np.nanmean(lc)
        lc_generated = np.random.poisson(lc, size=(N_poission_realisations, len(lc)))
        lc_percentiles = np.nanpercentile(lc_generated, (16,84), axis=0)

        plt.figure(figsize=(10, 3))
        plt.title(f'obsid={obsid} | label={label}')

        plt.step(range(len(lc)), lc, where='post')

        # Plot Error regions
        plt.fill_between(x=range(len(lc)),
                         y1=lc_percentiles[0],
                         y2=lc_percentiles[1],
                         alpha=0.4,
                         facecolor=color,
                         step="post",
                         label='16 and 84 percentiles')

        plt.xlabel('Window/Frame Number')
        plt.ylabel('Counts (N)')
        plt.legend()
        savepath = data_results / f'{obsid}' / f'lc_reg_{label}.png'
        logger.info(f'Saving lightcurve plot to: {savepath}')
        plt.savefig(savepath)


if __name__=='__main__':
    pass

