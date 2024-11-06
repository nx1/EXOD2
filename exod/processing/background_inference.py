from exod.utils.logger import logger
from exod.utils.plotting import compare_images
from exod.xmm.observation import Observation

import numpy as np
import matplotlib.pyplot as plt
from cv2 import inpaint, INPAINT_NS, INPAINT_TELEA
from skimage.draw import disk
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
from astropy.convolution import convolve, Gaussian2DKernel
from scipy.interpolate import interp1d


def mask_known_sources(data_cube, wcs):
    """
    Mask known sources from the data cube.

    Parameters:
        data_cube (DataCube): DataCube object.
        wcs (astropy.wcs.WCS): wcs object.

    Returns:
        image_mask (np.ndarray): The mask of the sources.
    """
    def cropping_radius_counts(epic_total_rate, size_arcsec):
        """
        Get cropping radius for a given EPIC count rate.
        0.0 < RATE < 0.1 = 20
        0.1 < RATE < 1.0 = 40
        RATE > 1.0 = 80
        """
        if epic_total_rate > 1:
            return 80
        elif epic_total_rate > 0.1:
            return 40
        else:
            return size_arcsec

    observation = Observation(data_cube.event_list.obsid)
    observation.get_source_list()
    obsmli_file_path = observation.source_list
    logger.info(f'OBSMLI file: {obsmli_file_path}')
    tab_src = Table(fits.open(obsmli_file_path)[1].data)

    # We include bright (i.e. large DET_ML) point sources & extended sources
    tab_src_point    = tab_src[(tab_src['EP_DET_ML'] > 8) & (tab_src['EP_EXTENT'] == 0)]
    tab_src_extended = tab_src[(tab_src['EP_DET_ML'] > 8) & (tab_src['EP_EXTENT'] > 0)]

    radius_point_sources    = [cropping_radius_counts(counts, data_cube.size_arcsec) for counts in tab_src_point['EP_TOT']]
    radius_extended_sources = tab_src_extended['EP_EXTENT']

    logger.info(f'Total rows: {len(tab_src)} Point sources: {len(tab_src_point)} Extended Sources: {len(tab_src_extended)}')

    image_mask = np.full(data_cube.shape[:2], fill_value=False)

    for tab_data, tab_radius, color in zip((tab_src_point, tab_src_extended),
                                           (radius_point_sources, radius_extended_sources),
                                           ('r', 'y')):
        if len(tab_data) == 0:
            logger.info('No rows in tab_data, skipping...')
            continue

        # Create skycoord
        sc = SkyCoord(ra=tab_data['RA'], dec=tab_data['DEC'], unit='deg')
        x_img, y_img = wcs.world_to_pixel(sc)

        # Convert to Sky Coordinates
        X = x_img * 80
        Y = y_img * 80

        # Remove values outside the cube
        xcube_max, xcube_min = data_cube.bin_x[-1], data_cube.bin_x[0]
        ycube_max, ycube_min = data_cube.bin_y[-1], data_cube.bin_y[0]
        XY = np.array([[x, y] for x, y in zip(X, Y) if ((x < xcube_max) and
                                                        (y < ycube_max) and
                                                        (x > xcube_min) and
                                                        (y > ycube_min))]).T
        X = XY[0]
        Y = XY[1]

        # Interpolate to Cube coordinates
        interp_x_cube = interp1d(x=data_cube.bin_x, y=range(data_cube.shape[0]))
        interp_y_cube = interp1d(x=data_cube.bin_y, y=range(data_cube.shape[1]))
        x_cube = interp_x_cube(X)
        y_cube = interp_y_cube(Y)

        for x, y, rad in zip(x_cube, y_cube, tab_radius):
            radius_image = rad / data_cube.size_arcsec
            rr, cc = disk(center=(x, y), radius=radius_image)
            kept_pixels = (rr>0) & (rr<data_cube.shape[0]-1) & (cc>0) & (cc<data_cube.shape[1]-1) # Keep mask pixels only if in image
            rr, cc = rr[kept_pixels], cc[kept_pixels]
            image_mask[rr, cc] = True

    #         im[rr,cc] = np.nan
    #         circle = plt.Circle((x, y), radius_image, color=color, fill=False)
    #         plt.gca().add_patch(circle)

    # plt.imshow(im.T, origin='lower', norm=LogNorm(vmax=1e3), interpolation='none')
    # plt.show()
    # for x, y in zip(x_cube, y_cube):
    #     im[x - 1:x + 1, y - 1:y + 1] = 0

    # plt.figure(figsize=(10, 10))
    # plt.imshow(im.T, origin='lower', norm=LogNorm(), interpolation='none')
    return image_mask

def calc_background_template(image_sub, image_mask_source):
    """
    Calculate the background template for a given image_sub.

    This is done by the following:
        1. Remove the sources from the image.
        2. Calculate the number of counts in the background.
        3. Inpaint the holes where the sources were.
        4. Divide the image by the total counts in the background.

    We then blur the image and inpaint again, this is to deal with issues where if large
    sources were removed we can fill them in, we should probably only do this if we need to.

    Parameters:
        image_sub (np.ndarray): Summed image of BTI or GTI frames.
        image_mask_source (np.ndarray): Mask of the sources.

    Returns:
        image_sub_background_template (np.ndarray): The background template for the image subset.
        count_sub_outside_sources (float): The number of counts outside the sources (total counts in background)
    """
    sigma_blurring = 0.5
    inpaint_method = INPAINT_NS # INPAINT_TELEA

    image_sub_no_source                       = inpaint(image_sub.astype(np.float32), image_mask_source.astype(np.uint8), inpaintRadius=2, flags=inpaint_method)
    count_sub_outside_sources                 = np.nansum(image_sub[~image_mask_source])
    image_sub_no_source_template              = image_sub_no_source / count_sub_outside_sources
    image_sub_no_source_template_blur         = convolve(image_sub_no_source_template, Gaussian2DKernel(sigma_blurring))
    image_mask_missing_pixels                 = (image_sub > 0) & np.isnan(image_sub_no_source_template_blur)
    image_sub_no_source_template_blur_inpaint = inpaint(image_sub_no_source_template_blur.astype(np.float32), image_mask_missing_pixels.astype(np.uint8), inpaintRadius=2, flags=inpaint_method)
    image_sub_background_template             = np.where(image_sub > 0, image_sub_no_source_template_blur_inpaint, np.nan)

    compare_images(images=[image_sub, image_sub_no_source],
                   titles=['image_sub', 'image_sub_no_source'],
                   log=False)

    compare_images(images=[image_sub_no_source_template,
                           image_sub_no_source_template_blur,
                           image_sub_no_source_template_blur_inpaint,
                           image_sub_background_template],
                   titles=['image_sub_no_source_template',
                           'image_sub_no_source_template_blur',
                           'image_sub_no_source_template_blur_inpaint',
                           'image_sub_background_template'],
                   log=False)
    return image_sub_background_template, count_sub_outside_sources


def calc_source_template(image_sub, image_sub_background_template, image_mask_source, data_cube,
                         count_sub_outside_sources, subset_bin_idx):
    """
    Calculate the average contribution from the sources in the field of the observation.

    This is done by masking out the background and dividing by the effective exposed frames.
    This essentially gives the image of the average contribution of the sources in each frame.

    Parameters:
        image_sub (np.ndarray): image of GTI or BTIs obtained by summing the frames.
        image_sub_background_template (np.ndarray): The template for the BTI or GTI obtained via calc_background_template.
        image_mask_source (np.ndarray): Mask of the sources.
        data_cube (DataCube): DataCube object.
        count_sub_outside_sources (float): The number of counts outside the sources (total counts in background).
        subset_bin_idx (np.ndarray): The bins corresponding to the subset (either the GTI or BTIs).

    Returns:
        image_source_template (np.ndarray): The average contribution of the sources in each frame.
    """
    image_sub_source_only1   = image_sub - image_sub_background_template * count_sub_outside_sources
    image_sub_source_only2   = np.where(image_mask_source, image_sub_source_only1, 0)           # Replace everything that is not a source with 0
    image_sub_source_only3   = np.where(image_sub_source_only2 > 0, image_sub_source_only2, 0)  # Replace negative values with 0
    effective_exposed_frames = np.sum(data_cube.relative_frame_exposures[subset_bin_idx])       # Analagous to n_gti_bin if relative exposures = 1
    image_source_template    = image_sub_source_only3 / effective_exposed_frames                # Average value of the source contribution per frame.

    compare_images(images=[image_sub_source_only1, image_sub_source_only2, image_sub_source_only3, image_source_template],
                   titles=['image_sub_source_only1', 'image_sub_source_only2', 'image_sub_source_only3', 'image_source_template'],
                   log=False)
    return image_source_template

def calc_cube_mu(data_cube, wcs):
    """
    Calculates an expectation (mu) data cube.

    Any departure from this cube corresponds to variability.

    The background is dealt with by assuming that all GTIs and BTIs follow
    respective templates (i.e., once each frame is divided by its total counts,
    they all look the same).

    The sources are dealt with assuming they are constant.
    We take their net emission, and distribute it evenly across all frames.

    Parameters:
        data_cube (DataCube): DataCube object.
        wcs (astropy.wcs.WCS): wcs object.
    Returns:
        cube_mu (np.ndarray): The expectation (mu) data cube.
    """
    cube_n = data_cube.data
    bti_bin_idx = data_cube.bti_bin_idx
    gti_bin_idx = data_cube.gti_bin_idx
    n_gti_bin = data_cube.n_gti_bin
    n_bti_bin = data_cube.n_bti_bin
    cube_gti = cube_n[:, :, gti_bin_idx]
    cube_bti = cube_n[:, :, bti_bin_idx]

    # Get the summed Images.
    image_total = np.nansum(cube_n, axis=2)
    image_gti = np.nansum(cube_gti, axis=2)

    # Create the source masks.
    # Two source masks are calculated then combined, the first comes from the pipeline detected
    # sources in the OBSMLI file. The second mask takes the remaining image and masks pixels that are
    # above 3 x 75% of the total image. These two masks are then combined using an OR statement.
    # This works well in most cases but often struggles when there are extended sources.
    image_mask_source_list = mask_known_sources(data_cube, wcs=wcs)  # Image mask from OBSMLI file.
    # source_threshold = np.nanpercentile(image_gti.flatten(), 99) # This or from detected sources
    source_threshold = 3 * np.nanpercentile(image_gti[~image_mask_source_list], q=75)
    image_mask_source_percentile = image_gti > source_threshold
    image_mask_source = image_mask_source_list | image_mask_source_percentile

    if n_gti_bin:
        logger.info('Calculating gti template...')
        image_gti = np.nansum(cube_gti, axis=2)
        image_gti_background_template, count_gti_outside_sources = calc_background_template(image_sub=image_gti, image_mask_source=image_mask_source)

    if n_bti_bin:
        logger.info('Calculating bti template...')
        image_bti = np.nansum(cube_bti, axis=2) # Image of all btis combined, with sources
        image_bti_background_template, count_bti_outside_sources = calc_background_template(image_sub=image_bti, image_mask_source=image_mask_source)

    # image_bti_no_source_template_blur=image_bti_no_source_template

    # Obtain the Image with the mean source contribution (zero everywhere that is not a source)
    if n_gti_bin > n_bti_bin:
        logger.info('Calculating source contribution using gtis')
        image_source_template = calc_source_template(image_gti, image_gti_background_template, image_mask_source,
                                                     data_cube, count_gti_outside_sources, gti_bin_idx)
    else:
        logger.info('Calculating source contribution using btis')
        image_source_template = calc_source_template(image_bti, image_bti_background_template, image_mask_source,
                                                     data_cube, count_bti_outside_sources, bti_bin_idx)

    # Get data cube outside the sources to obtain the background light curve.
    cube_mask_source       = np.repeat(image_mask_source[:, :, np.newaxis], cube_n.shape[2], axis=2)
    cube_n_outside_sources = np.where(cube_mask_source, np.nan, cube_n)
    lc_outside_sources     = np.nansum(cube_n_outside_sources, axis=(0, 1)) # Essentially the background light curve.


    logger.info('Creating expected (mu) cube...')
    # Create the expectation cube by combining the source and background templates.
    cube_mu = np.empty(cube_n.shape)
    if n_gti_bin:
        cube_mu[:,:,gti_bin_idx] = image_gti_background_template[:,:,np.newaxis] * lc_outside_sources[gti_bin_idx]
    if n_bti_bin:
        cube_mu[:,:,bti_bin_idx] = image_bti_background_template[:,:,np.newaxis] * lc_outside_sources[bti_bin_idx]
    cube_mu += image_source_template[:,:,np.newaxis] * data_cube.relative_frame_exposures
    cube_mu = np.where(np.nansum(cube_n, axis=(0,1)) > 0, cube_mu, np.nan)
    logger.info('Expected cube created!')
    return cube_mu




if __name__=="__main__":
    from exod.utils.path import data
    from exod.processing.pipeline import DataLoader
    from exod.xmm.event_list import EventList
    from exod.utils.path import read_observation_ids

    obsids = read_observation_ids(data / 'observations.txt')
    import random
    random.shuffle(obsids)
    for obsid in obsids:
        size_arcsec = 20
        time_interval = 10
        gti_only = False
        gti_threshold = 1.5
        min_energy = 0.2
        max_energy = 12.0

        try:
            observation = Observation(obsid)
            observation.get_files()
            observation.get_events_overlapping_subsets()
            img = observation.images[0]
            img.read(wcs_only=True)
            for ind_exp, subset_overlapping_exposures in enumerate(observation.events_overlapping_subsets):
                event_list = EventList.from_event_lists(subset_overlapping_exposures)
                # event_list = observation.events_processed_pn[0]
                # event_list.read()
                dl = DataLoader(event_list=event_list, time_interval=time_interval, size_arcsec=size_arcsec,
                                gti_only=gti_only, min_energy=min_energy, max_energy=max_energy)
                dl.run()
                data_cube = dl.data_cube
                estimated_cube = calc_cube_mu(data_cube=data_cube, wcs=img.wcs)
        except Exception as e:
            logger.warning(e)


