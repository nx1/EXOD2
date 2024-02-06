from exod.utils.path import data_processed, data_results
from exod.utils.logger import logger

import numpy as np
from astropy.convolution import convolve
from astropy.stats import sigma_clip
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
import pandas as pd
from matplotlib.colors import LogNorm
from scipy.stats import kstest, poisson
from skimage.measure import label, regionprops, regionprops_table


def plot_cube_statistics(cube):
    logger.info('Calculating and plotting data cube statistics...')
    image_max    = np.nanmax(cube, axis=2)
    image_min    = np.nanmin(cube, axis=2) # The Minimum and median are basically junk
    image_median = np.nanmedian(cube, axis=2)
    image_mean   = np.nanmean(cube, axis=2)
    image_std    = np.nanstd(cube, axis=2)
    image_sum    = np.nansum(cube, axis=2)

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    # Plotting images
    im_max    = ax[0, 0].imshow(image_max, interpolation='none')
    im_min    = ax[0, 1].imshow(image_min, interpolation='none')
    im_mean   = ax[1, 0].imshow(image_mean, interpolation='none')
    im_median = ax[1, 1].imshow(image_median, interpolation='none')
    im_std    = ax[1, 2].imshow(image_std, interpolation='none')
    im_sum    = ax[0, 2].imshow(image_sum, interpolation='none')

    # Adding colorbars
    cbar_max    = fig.colorbar(im_max, ax=ax[0, 0])
    cbar_min    = fig.colorbar(im_min, ax=ax[0, 1])
    cbar_mean   = fig.colorbar(im_mean, ax=ax[1, 0])
    cbar_median = fig.colorbar(im_median, ax=ax[1, 1])
    cbar_std    = fig.colorbar(im_std, ax=ax[1, 2])
    cbar_sum    = fig.colorbar(im_sum, ax=ax[0, 2])

    # Setting titles
    ax[0, 0].set_title('max')
    ax[0, 1].set_title('min')
    ax[1, 0].set_title('mean')
    ax[1, 1].set_title('median')
    ax[1, 2].set_title('std')
    ax[0, 2].set_title('sum')

    #plt.show()

def calc_var_img(cube):
    """
    Calculate the variability image from a data cube.
    """
    logger.info('Computing Variability')
    image_max    = np.nanmax(cube, axis=2)
    image_std    = np.nanstd(cube, axis=2)
    # image_sum    = np.nansum(cube, axis=2)

    # condition = np.nanmax((image_max - image_median, image_median - image_min)) / image_median
    #condition = np.nanmax((image_max - image_mean, image_mean - image_min)) / image_mean
    # var_img = np.where(image_mean > 0, condition, image_max)
    var_img = image_max * image_std
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


def extract_var_regions(var_img, sigma=5):
    """
    Use skimage to obtain the contiguous regions in the variability image.

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
    threshold = np.mean(var_img) + sigma * np.std(var_img)
    logger.info(f'threshold: {threshold} sigma: {sigma}')
    var_img_mask = var_img > threshold

    """
    # Iterative sigma clipping
    v_filt, lo, hi = sigma_clip(var_img_flat, sigma=sigma, sigma_lower=None, sigma_upper=None,
                                maxiters=5, cenfunc='median', stdfunc='std',
                                axis=None, masked=True, return_bounds=True,
                                copy=True, grow=False)
    threshold = hi
    """

    plot_threshold_level(var_img, sigma, threshold)

    # Use the skimage label function to label connected components
    # The result is an array that has the number of the region
    # in the position of the variable regions
    # 0011020
    # 0010220
    # 0000000
    var_img_mask_labelled = label(var_img_mask)

    # Obtain the region properties for the detected regions.
    properties_ = ('label', 'bbox', 'weighted_centroid', 'intensity_mean', 'equivalent_diameter_area', 'area_bbox')
    region_dict = regionprops_table(label_image=var_img_mask_labelled,
                                    intensity_image=var_img,
                                    properties=properties_)
    df_regions = pd.DataFrame(region_dict)

    # Sort by Most Variable and reset in the label column
    df_regions = df_regions.sort_values(by='intensity_mean', ascending=False).reset_index(drop=True)
    df_regions['label'] = df_regions.index
    return df_regions


def plot_threshold_level(var_img, sigma, threshold):
    plt.figure(figsize=(10, 3))
    plt.title('Thresholding')
    plt.plot(var_img.flatten(), label='Variability')
    plt.axhline(threshold, color='red', label=fr'threshold={threshold:.2f} ({sigma}$\sigma$)')
    plt.legend()
    plt.ylabel('V score')
    # plt.show()


def filter_df_regions(df_regions):
    logger.info('Removing regions with area_bbox > 12')
    df_regions = df_regions[df_regions['area_bbox'] <= 12] # Large regions
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

    src_color = 'blue'
    ax.scatter(df_regions['weighted_centroid-0'], df_regions['weighted_centroid-1'], marker='+', s=10, color=src_color)
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
                                 edgecolor=src_color,
                                 facecolor='none')

        plt.text(x_pos+width, y_pos+height, str(ind), c=src_color)
        ax.add_patch(rect)
    logger.info(f'Saving Variability image to: {outfile}')
    plt.savefig(outfile)
    # plt.show()


def get_regions_sky_position(df_regions, wcs, data_cube):
    """
    Calculate the sky position of the detected regions.
    """
    # To calculate the EPIC X and Y coordinates of the variable sources, we use the final coordinates
    # in the variability map, which are not integers. To know to which X and Y correspond to this, we interpolate the
    # values of X and Y on the final coordinates. We divide by 80 because the WCS from the image is binned by x80
    # compared to X and Y values
    logger.info(f'Getting X and Y interpolation limits')
    logger.warning(f'Assuming a binning of 80 in the image file')
    interpX = interp1d(range(len(data_cube.bin_x)), data_cube.bin_x / 80)
    interpY = interp1d(range(len(data_cube.bin_y)), data_cube.bin_y / 80)

    all_res = []
    for i, row in df_regions.iterrows():
        X = interpX(row['weighted_centroid-0'])
        Y = interpY(row['weighted_centroid-1'])
        pos = wcs.pixel_to_world(X, Y)

        res = {'X'   : X,
               'Y'   : Y,
               'ra'  : pos.ra.value,
               'dec' : pos.dec.value}
        all_res.append(res)
            
    df_sky = pd.DataFrame(all_res)
    df_regions = pd.concat([df_regions, df_sky], axis=1)
    return df_regions


def get_region_lightcurves(data_cube, df_regions):
    """
    Extract the lightcurves from the variable regions found.
    Returns
    -------
    df_lcs : DataFrame of Lightcurves
    """
    logger.info("Extracting lightcurves from data cube")
    lcs = [pd.DataFrame({'time': data_cube.bin_t[:-1]}),
           pd.DataFrame({'bti': data_cube.rejected_frame_bool[:-1]})]
    for i, row in df_regions.iterrows():
        xlo, xhi = row['bbox-0'], row['bbox-2']
        ylo, yhi = row['bbox-1'], row['bbox-3']
        data = data_cube.data[xlo:xhi, ylo:yhi]
        lc = np.nansum(data, axis=(0,1), dtype=np.int32)
        res = pd.DataFrame({f'src_{i}' : lc})
        lcs.append(res)

    df_lcs = pd.concat(lcs, axis=1)
    return df_lcs


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


def plot_region_lightcurves(df_lcs, df_regions, obsid):
    logger.info(f'Plotting regions lightcurves')
    for i, row in df_regions.iterrows():
        label = row['label']
        lc = df_lcs[f'src_{i}']
        #lc = lcs[i]

        """
        N_poission_realisations = 5000
        logger.info(f'Calculating Errors, using {N_poission_realisations} Poission realisations ')
        lc_mean = np.nanmean(lc)
        lc_generated = np.random.poisson(lc, size=(N_poission_realisations, len(lc)))
        lc_percentiles = np.nanpercentile(lc_generated, (16,84), axis=0)
        """

        fig, ax = plt.subplots(figsize=(10, 4))
        ax2 = ax.twiny()
        ax.set_title(f'obsid={obsid} | label={label}')
        ax.step(df_lcs['time'], df_lcs[f'src_{i}'], where='post', color='black', lw=1.0)
        ax2.step(range(len(lc)), lc, where='post', color='black', lw=1.0)

        """
        # Plot Error regions
        color = cmr.take_cmap_colors(cmap='cmr.ocean', N=1, cmap_range=(0.3, 0.3))[0]
        plt.fill_between(x=range(len(lc)),
                         y1=lc_percentiles[0],
                         y2=lc_percentiles[1],
                         alpha=0.4,
                         facecolor=color,
                         step="post",
                         label='16 and 84 percentiles')
        """

        ax.set_ylabel('Counts (N)')
        ax.set_xlabel('Time (s)')
        ax2.set_xlabel('Window/Frame Number')
        savepath = data_results / obsid / f'lc_reg_{label}.png'
        logger.info(f'Saving lightcurve plot to: {savepath}')
        plt.savefig(savepath)


if __name__=='__main__':
    pass

