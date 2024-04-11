from exod.pre_processing.data_loader import DataLoader
from exod.xmm.event_list import EventList
from exod.xmm.observation import Observation
from exod.utils.path import data_processed, data_raw, read_observation_ids
from exod.pre_processing.read_events import get_PN_image_file
from exod.utils.logger import logger
from exod.utils.plotting import plot_image, compare_images

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


def compute_expected_cube_using_templates(data_cube, wcs=None):
    """
    Computes a baseline data_cube cube, combining background and sources.

    Any departure from this data_cube cube corresponds to variability.

    The background is dealt with by assuming that all GTIs and BTIs follow
    respective templates (i.e., once each frame is divided by its total counts,
    they all look the same).

    The sources are dealt with assuming they are constant. We take their net
    emission, and distribute it evenly across all frames.

    Parameters
    ----------
    wcs : astropy.wcs.WCS() object.
    """
    sigma_blurring = 0.5
    inpaint_method = INPAINT_NS # INPAINT_TELEA


    logger.info(f'Computing Expected Cube using templates.')
    cube_n = data_cube.data
    bti_indices = data_cube.bti_bin_idx
    gti_indices = data_cube.gti_bin_idx
    N_GTI = data_cube.n_gti_bin
    N_BTI = data_cube.n_bti_bin
    cube_GTI = cube_n[:, :, gti_indices]
    cube_BTI = cube_n[:, :, bti_indices]


    # Get the summed Images.
    image_total = np.nansum(cube_n, axis=2)
    image_GTI   = np.nansum(cube_GTI, axis=2)
    image_GTI   = np.where(image_GTI > 0, image_GTI, np.nan) # Replace 0s with NAN (why tho :/)

    # Create the source masks.
    # Two source masks are calculated then combined, the first comes from the pipeline detected
    # sources in the OBSMLI file. The second mask takes the remaining image and masks pixels that are
    # above 3 x 75% of the total image. These two masks are then combined using an OR statement.
    # This works well in most cases but often struggles when there are extended sources.
    image_mask_source_list = mask_known_sources(data_cube, wcs=wcs)  # Image mask from OBSMLI file.
    # source_threshold = np.nanpercentile(image_GTI.flatten(), 99) # This or from detected sources
    source_threshold = 3 * np.nanpercentile(image_GTI[~image_mask_source_list], q=75)
    image_mask_source_percentile = image_GTI > source_threshold
    image_mask_source = image_mask_source_list | image_mask_source_percentile

    compare_images(images=[image_mask_source_list, image_mask_source_percentile, image_mask_source],
                   titles=['image_mask_source_list', 'image_mask_source_percentile', 'image_mask_source'])

    # inpaint source regions to get image with no sources.
    image_GTI_no_source               = inpaint(image_GTI.astype(np.float32), image_mask_source.astype(np.uint8), inpaintRadius=2, flags=inpaint_method)
    count_GTI_outside_sources         = np.nansum(image_GTI[~image_mask_source]) # The total background count of GTIs, outside of the source regions
    image_GTI_no_source_template      = image_GTI_no_source / count_GTI_outside_sources
    image_GTI_no_source_template_blur = convolve(image_GTI_no_source_template, Gaussian2DKernel(sigma_blurring))

    compare_images(images=[image_GTI, image_GTI_no_source],
                   titles=['image_GTI', 'image_GTI_no_source'], log=False)

    compare_images(images=[image_GTI_no_source_template, image_GTI_no_source_template_blur],
                   titles=['image_GTI_no_source_template', 'image_GTI_no_source_template_blur'], log=False)

    # Perform the inpainting a second time to fill in missing pixels (gaps)
    image_mask_missing_pixels = (image_GTI > 0) & np.isnan(image_GTI_no_source_template_blur)
    image_GTI_no_source_template_blur_inpaint = inpaint(image_GTI_no_source_template_blur.astype(np.float32), image_mask_missing_pixels.astype(np.uint8), inpaintRadius=2, flags=inpaint_method)

    # Remove the pixels that were inpainted outside the original image
    image_GTI_background_template = np.where(image_total > 0, image_GTI_no_source_template_blur_inpaint, np.nan)

    compare_images(images=[image_GTI_no_source_template, image_GTI_no_source_template_blur],
                   titles=['image_GTI_no_source_template', 'image_GTI_no_source_template_blur'], log=False)

    compare_images(images=[image_GTI_no_source_template_blur_inpaint, image_GTI_background_template],
                   titles=['image_GTI_no_source_template_blur_inpaint', 'image_GTI_background_template'], log=False)


    # BTI template: compute the estimated BTI background
    if N_BTI:
        image_BTI                         = np.nansum(cube_BTI, axis=2) # Image of all BTIs combined, with sources
        image_BTI_no_source               = inpaint(image_BTI.astype(np.float32), image_mask_source.astype(np.uint8), inpaintRadius=2, flags=inpaint_method)
        count_BTI_outside_sources         = np.nansum(image_BTI[~image_mask_source])
        image_BTI_no_source_template      = image_BTI_no_source / count_BTI_outside_sources
        image_BTI_no_source_template_blur = convolve(image_BTI_no_source_template, Gaussian2DKernel(sigma_blurring))

        compare_images(images=[image_BTI, image_BTI_no_source], titles=['image_BTI', 'image_BTI_no_source'], log=False)
        compare_images(images=[image_BTI_no_source_template, image_BTI_no_source_template_blur],
                       titles=['image_BTI_no_source_template', 'image_BTI_no_source_template_blur'],
                       log=False)

        image_mask_missing_pixels = (image_GTI > 0) & np.isnan(image_GTI_no_source_template_blur)
        image_BTI_no_source_template_blur = inpaint(image_BTI_no_source_template_blur.astype(np.float32), image_mask_missing_pixels.astype(np.uint8), inpaintRadius=2, flags=inpaint_method)
        image_BTI_no_source_template_blur = np.where(image_total > 0, image_BTI_no_source_template_blur, np.nan)

    # image_BTI_no_source_template_blur=image_BTI_no_source_template

    # Obtain the Image with the mean source contribution (zero everywhere that is not a source)
    # The source contribution is calculated from exclusively form BTIs or GTIs, whichever are more numerous.
    if N_BTI < cube_n.shape[2]/2:
        image_GTI_source_only1   = image_GTI - image_GTI_background_template * count_GTI_outside_sources
        image_GTI_source_only2   = np.where(image_mask_source, image_GTI_source_only1, 0) # Replace everything that is not a source with 0
        image_GTI_source_only3   = np.where(image_GTI_source_only2 > 0, image_GTI_source_only2, 0) # Replace negative values with 0
        effective_exposed_frames = np.sum(data_cube.relative_frame_exposures[gti_indices]) # Analagous to N_GTI if relative exposures = 1
        image_source_only_mean   = image_GTI_source_only3 / effective_exposed_frames # Average value of the source contribution per frame.
    else:
        image_BTI_source_only1    = image_BTI - image_BTI_no_source_template_blur * count_GTI_outside_sources
        image_BTI_source_only2    = np.where(image_mask_source, image_BTI_source_only1, 0)
        image_BTI_source_only3    = np.where(image_BTI_source_only2 > 0, image_BTI_source_only2, 0)
        effective_exposed_frames  = np.sum(data_cube.relative_frame_exposures[bti_indices])
        image_source_only_mean    = image_BTI_source_only3 / effective_exposed_frames

    # Get data cube outside the sources to obtain the background light curve.
    cube_mask_source       = np.repeat(image_mask_source[:, :, np.newaxis], cube_n.shape[2], axis=2)
    cube_n_outside_sources = np.where(cube_mask_source, np.nan, cube_n)
    lc_outside_sources     = np.nansum(cube_n_outside_sources, axis=(0, 1)) # Essentially the background light curve.

    # We then create the expected (mu) cube.
    # For both GTI and BTI, we estimate their background by using the lightcurve and the templates.
    # We then add the constant source contribution.
    logger.info('Creating expected cube...')
    cube_mu = np.empty(cube_n.shape)
    cube_mu[:,:,gti_indices] = image_GTI_background_template[:,:,np.newaxis] * lc_outside_sources[gti_indices]
    if N_BTI:
        cube_mu[:,:,bti_indices] = image_BTI_no_source_template_blur[:,:,np.newaxis] * lc_outside_sources[bti_indices]
    cube_mu += image_source_only_mean[:,:,np.newaxis] * data_cube.relative_frame_exposures
    cube_mu = np.where(np.nansum(cube_n, axis=(0,1)) > 0, cube_mu, np.nan)
    logger.info('Expected cube created!')
    return cube_mu


def mask_known_sources(data_cube, wcs=None):
    def cropping_radius_counts(data_cube, epic_total_rate):
        if epic_total_rate > 1:
            return 80
        elif epic_total_rate > 0.1:
            return 40
        else:
            return data_cube.size_arcsec
    observation = Observation(data_cube.event_list.obsid)
    observation.get_source_list()
    obsmli_file_path = observation.source_list
    logger.info(f'OBSMLI file: {obsmli_file_path}')

    tab_src = Table(fits.open(obsmli_file_path)[1].data)

    # We include bright (i.e. large DET_ML) point sources & extended sources
    tab_src_point    = tab_src[(tab_src['EP_DET_ML'] > 8) & (tab_src['EP_EXTENT'] == 0)]
    tab_src_extended = tab_src[(tab_src['EP_DET_ML'] > 8) & (tab_src['EP_EXTENT'] > 0)]

    radius_point_sources    = [cropping_radius_counts(data_cube, counts) for counts in tab_src_point['EP_TOT']]
    radius_extended_sources = tab_src_extended['EP_EXTENT']

    logger.info(f'Total rows: {len(tab_src)} Point sources: {len(tab_src_point)} Extended Sources: {len(tab_src_extended)}')

    mask = np.full(data_cube.shape[:2], fill_value=False)

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
            mask[rr, cc] = True

    #         im[rr,cc] = np.nan
    #         circle = plt.Circle((x, y), radius_image, color=color, fill=False)
    #         plt.gca().add_patch(circle)

    # plt.imshow(im.T, origin='lower', norm=LogNorm(vmax=1e3), interpolation='none')
    # plt.show()
    # for x, y in zip(x_cube, y_cube):
    #     im[x - 1:x + 1, y - 1:y + 1] = 0

    # plt.figure(figsize=(10, 10))
    # plt.imshow(im.T, origin='lower', norm=LogNorm(), interpolation='none')
    return mask




if __name__=="__main__":
    from exod.utils.path import data
    obsids = read_observation_ids(data / 'observations.txt')
    for obsid in obsids:
        size_arcsec = 20
        time_interval = 10
        gti_only = False
        gti_threshold = 1.5
        min_energy = 0.2
        max_energy = 12.0

        threshold_sigma = 3

        try:
            # Load data
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
                                gti_only=gti_only, min_energy=min_energy, max_energy=max_energy,
                                gti_threshold=gti_threshold)
                dl.run()
                data_cube = dl.data_cube
                estimated_cube = compute_expected_cube_using_templates(data_cube=data_cube, wcs=img.wcst )
        except Exception as e:
            logger.warning(e)


