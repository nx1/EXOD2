import os
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import cmasher as cmr
from matplotlib.colors import LogNorm
from astropy.wcs import WCS
from astropy.io import fits
from skimage.measure import label, regionprops

from exod.utils.path import data_processed, data_results
from exod.utils.logger import  logger


def extract_variability_regions(var_img, threshold):
    """
    Use skimage to obtain the contiguous regions in
    the variability image.

    the threshold used is given by:
        threshold * median

    then extracts the center of mass and bounding box of the corresponding pixel regions
    Parameters
    ----------
    var_img : Variability image (2D)
    threshold : Threshold over which to consider variable regions

    Returns
    -------
    df : DataFrame of Detected Regions
    """
    logger.info('Extracting variable Regions')
    threshold_value = threshold * np.median(var_img)
    logger.info(f'Threshold multiplier: {threshold} threshold_value: {threshold_value}')
    var_img_mask = var_img > threshold_value

    # Use the skimage label function to label connected components
    # The result is an array that has the number of the region
    # in the position of the variable regions
    # 0011020
    # 0010020
    # 0000000
    var_img_mask_labelled = label(var_img_mask)

    # Plot the Variable Regions
    plt.figure(figsize=(4,4))
    plt.title('Identified Variable Regions')
    plt.imshow(var_img_mask_labelled,
               cmap='tab20c',
               interpolation='none')
    plt.colorbar()

    regions = regionprops(label_image=var_img_mask_labelled,
                          intensity_image=var_img)
    all_res = []
    for i, r in enumerate(regions):
        # We can pull out a lot from each region here
        # See: https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
        res = {'region_number'     : i,
               'weighted_centroid' : r.weighted_centroid,
               'bbox'              : r.bbox,
               'intensity_mean'    : r.intensity_mean}
        all_res.append(res)

    df_regions = pd.DataFrame(all_res)
    logger.info(f'Detected Regions:\n{df_regions}')
    return df_regions

def plot_variability_with_regions(centers, bboxes, variability_map, outfile):
    logger.info('Plotting Variability map with source regions')

    fig, ax = plt.subplots()
    cmap = plt.cm.hot
    cmap.set_bad('black')
    m1 = ax.imshow(variability_map.T,
                   norm=LogNorm(),
                   interpolation='none',
                   origin='lower',
                   cmap=cmap)
    cbar = plt.colorbar(mappable=m1, ax=ax)
    cbar.set_label("Variability")

    for ind, (center, bbox) in enumerate(zip(centers, bboxes)):
        xpos = bbox[0] - 1.5
        ypos = bbox[1] - 1

        width  = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        rect = patches.Rectangle((xpos, ypos),
                                 width+2,
                                 height+2,
                                 linewidth=1,
                                 edgecolor='w',
                                 facecolor='none')

        plt.text(xpos+width, ypos+height, str(ind), c='w')
        ax.add_patch(rect)
    plt.savefig(outfile)
    plt.show()

def get_regions_sky_position(obsid, tab_centersofmass, coordinates_XY):
    logger.info('Getting sky positions of regions')
    path_processed_obs = data_processed / f'{obsid}'
    datapath = path_processed_obs / "PN_image.fits"
    datapath = path_processed_obs / "M1_image.fits"
    datapath = path_processed_obs / "M2_image.fits"

    logger.info(f'Opening fits file: {datapath}')
    f = fits.open(datapath)

    logger.info('Creating WCS from header')
    header = f[0].header
    w = WCS(header)
    logger.info(w)

    #Watch out for this move: to know the EPIC X and Y coordinates of the variable sources, we use the final coordinates
    #in the variability map, which are not integers. To know to which X and Y correspond to this, we interpolate the
    #values of X and Y on the final coordinates. We divide by 80 because the WCS from the image is binned by x80
    #compared to X and Y values
    interpX = interp1d(range(len(coordinates_XY[0])), coordinates_XY[0]/80)
    interpY = interp1d(range(len(coordinates_XY[1])), coordinates_XY[1]/80)

    all_res = []
    for (end_x ,end_y) in tab_centersofmass:
        logger.info(f'Interpolating X, and Y end_x={end_x} end_y={end_y}')
        X = interpX(end_x)
        Y = interpY(end_y)
        pos = w.pixel_to_world(X,Y)


        res = {'X'   : X,
               'Y'   : Y,
               'ra'  : pos.ra.value,
               'dec' : pos.dec.value}
        all_res.append(res)
            

    df = pd.DataFrame(all_res)
    logger.info(f'df:\n{df}')
    return df 
if __name__=='__main__':
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from exod.pre_processing.read_events_files import read_EPIC_events_file
    from exod.processing.variability_computation import compute_pixel_variability, convolve_variability
    from matplotlib.colors import LogNorm

    cube,coordinates_XY = read_EPIC_events_file('0831790701', 10, 100,3, gti_only=False)
    variability_map = compute_pixel_variability(cube)
    plot_variability_with_regions(variability_map, 8,
                                   os.path.join(data_results,'0831790701','Variability_Regions.png'))
    tab_centersofmass, bboxes = extract_variability_regions(variability_map, 8)
    tab_ra, tab_dec, tab_X, tab_Y=get_regions_sky_position('0831790701', tab_centersofmass, coordinates_XY)
