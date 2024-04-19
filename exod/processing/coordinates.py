import numpy as np
import astropy.units as u
import pandas as pd
from scipy.interpolate import interp1d
from skimage.measure import regionprops_table, label

from exod.utils.logger import logger


def get_regions_sky_position(data_cube, df_regions, wcs):
    """
    Calculate the sky position of the detected regions.

    Test coords: observation : 0803990501
    1313 X-1     : 03 18 20.00 -66 29 10.9
    1313 X-2     : 03 18 22.00 -66 36 04.3
    SN 1978K     : 03 17 38.62 -66 33 03.4
    NGC1313 XMM4 : 03 18 18.46 -66 30 00.2 (lil guy next to x-1)

    To calculate the EPIC X and Y coordinates of the variable sources, we use the final coordinates
    in the variability map, which are not integers. To know to which X and Y correspond to this, we interpolate the
    values of X and Y on the final coordinates. We divide by 80 because the WCS from the image is binned by x80
    compared to X and Y values
    """
    img_bin_size = 80  # This is the binning of the wcs image when it was created using evselect.

    interpX = interp1d(range(len(data_cube.bin_x)), data_cube.bin_x)
    interpY = interp1d(range(len(data_cube.bin_y)), data_cube.bin_y)

    all_res = []
    for i, row in df_regions.iterrows():
        X = interpX(row['weighted_centroid-0'])
        Y = interpY(row['weighted_centroid-1'])

        x_img = X / img_bin_size
        y_img = Y / img_bin_size
        skycoord = wcs.pixel_to_world(x_img, y_img)
        skycoord = correct_sky_position(skycoord, data_cube)

        res = {'x_img'    : x_img,  # x_position in the binned fits image
               'y_img'    : y_img,  # y_position in the binned fits image
               'X'        : X,      # X in the event_list (sky coordinates)
               'Y'        : Y,      # Y in the event-list (sky coordinates)
               'ra'       : skycoord.ra.to_string(unit=u.hourangle, precision=2),
               'dec'      : skycoord.dec.to_string(unit=u.deg, precision=2),
               'ra_deg'   : skycoord.ra.value,
               'dec_deg'  : skycoord.dec.value}
        all_res.append(res)

    df_sky = pd.DataFrame(all_res)
    logger.info(f'df_sky:\n{df_sky}')
    return df_sky


def correct_sky_position(skycoord, data_cube):
    """
    Correct the systematic offset due to the gridding.

    The change we need needs to be positive in dec and negative in RA, which results in diagonal (UL/NW)
    position angle correction, corresponding to 360-45=315 degrees.

    The angular separation is the diagonal distance from the corner to the center of a pixel thus:
        (2*(size_arcsec/2)^2)^0.5
    """
    position_angle = 315 * u.deg
    separation = np.sqrt(2 * (data_cube.size_arcsec / 2) ** 2) * u.arcsec
    sc = skycoord.directional_offset_by(position_angle=position_angle, separation=separation)
    return sc


def calc_df_regions(image, image_mask):
    """Return the region dataframe for a given image and mask."""
    properties_ = ('label', 'bbox', 'centroid', 'weighted_centroid', 'intensity_mean', 'equivalent_diameter_area', 'area_bbox')
    region_dict = regionprops_table(label_image=label(image_mask), intensity_image=image, properties=properties_)
    df_region = pd.DataFrame(region_dict)
    return df_region
