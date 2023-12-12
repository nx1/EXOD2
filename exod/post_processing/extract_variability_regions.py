from skimage.measure import label, regionprops
from astropy.wcs import WCS
from astropy.io import fits
from scipy.interpolate import interp1d
import numpy as np
import os
import cmasher as cmr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from exod.utils.path import data_processed, data_results


def extract_variability_regions(variability_map, threshold):
    """Labels the contiguous regions above the threshold times the median variability, then extracts the
    center of mass and bounding box of the corresponding pixel regions"""
    variable_pixels = (variability_map > threshold*np.median(variability_map)).astype(int)
    labeled_variability_map = label(variable_pixels)

    tab_centersofmass=[]
    tab_boundingboxes=[]
    for source in range(1,np.max(labeled_variability_map)):
        source_properties = regionprops(label_image=(labeled_variability_map==source).astype(int),
                                        intensity_image=variability_map)
        tab_centersofmass.append(source_properties[0].weighted_centroid)
        tab_boundingboxes.append(source_properties[0].bbox)
    return tab_centersofmass, tab_boundingboxes

def plot_variability_with_regions(variability_map, threshold, outfile):
    fig, ax = plt.subplots()
    m1=ax.imshow(variability_map.T, norm=LogNorm(), interpolation='none', origin='lower', cmap="cmr.ember")
    cbar=plt.colorbar(mappable=m1, ax=ax)
    cbar.set_label("Variability")
    centers, bboxes = extract_variability_regions(variability_map, threshold)
    for ind, center, bbox in zip(range(len(centers)),centers, bboxes):
        min_error = 10
        width= bbox[2]-bbox[0]
        height =  bbox[3]-bbox[1]
        shiftx=0
        shifty=0
        if width<min_error:
            shiftx = min_error-width
            width = min_error
        if height < min_error:
            shifty = min_error - height
            height = min_error
        rect = patches.Rectangle((bbox[0]-1-shifty/2,bbox[1]-1-shiftx/2), width, height, linewidth=1, edgecolor='w',
                                 facecolor='none')
        plt.text(bbox[0]-1-shifty/2+width+3,bbox[1]-1-shiftx/2+height+3, ind, c='w')
        ax.add_patch(rect)
    plt.axis('off')
    plt.savefig(outfile)

def get_regions_sky_position(obsid, tab_centersofmass, coordinates_XY):
    if "PN_pattern_clean.fits" in os.listdir(os.path.join(data_processed,obsid)):
        datapath = os.path.join(data_processed,obsid,"PN_image.fits")
    elif "M1_pattern_clean.fits" in os.listdir(os.path.join(data_processed,obsid)):
        datapath = os.path.join(data_processed, obsid, "M1_image.fits")
    elif "M2_pattern_clean.fits" in os.listdir(os.path.join(data_processed,obsid)):
        datapath = os.path.join(data_processed, obsid, "M2_image.fits")
    else:
        raise FileNotFoundError('No suitable EPIC processed data found !')
    f=fits.open(datapath)
    header = f[0].header
    w=WCS(header)
    #Watch out for this move: to know the EPIC X and Y coordinates of the variable sources, we use the final coordinates
    #in the variability map, which are not integers. To know to which X and Y correspond to this, we interpolate the
    #values of X and Y on the final coordinates. We divide by 80 because the WCS from the image is binned by x80
    #compared to X and Y values
    interpX = interp1d(range(len(coordinates_XY[0])), coordinates_XY[0]/80)
    interpY = interp1d(range(len(coordinates_XY[1])), coordinates_XY[1]/80)
    tab_ra = []
    tab_dec = []
    tab_X = []
    tab_Y = []
    for (end_x,end_y) in tab_centersofmass:
        X = interpX(end_x)
        Y = interpY(end_y)
        tab_X.append(X)
        tab_Y.append(Y)
        pos = w.pixel_to_world(X,Y)
        tab_ra.append(pos.ra.value)
        tab_dec.append(pos.dec.value)
    return tab_ra, tab_dec, tab_X, tab_Y

if __name__=='__main__':
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    from exod.pre_processing.read_events_files import read_EPIC_events_file
    from exod.processing.variability_computation import compute_pixel_variability, convolve_variability
    from matplotlib.colors import LogNorm

    cube,coordinates_XY = read_EPIC_events_file('0831790701', 10, 100,3, gti_only=False)
    variability_map = compute_pixel_variability(cube)
    plot_variability_with_regions(variability_map, 8,
                                   os.path.join(data_results,'0831790701','Variability_Regions.png'))
    tab_centersofmass, bboxes = extract_variability_regions(variability_map, 8)
    tab_ra, tab_dec, tab_X, tab_Y=get_regions_sky_position('0831790701', tab_centersofmass, coordinates_XY)
