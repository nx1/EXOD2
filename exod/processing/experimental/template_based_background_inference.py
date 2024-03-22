from exod.pre_processing.data_loader import DataLoader
from exod.xmm.event_list import EventList
from exod.xmm.observation import Observation
from exod.utils.path import data_processed
from exod.pre_processing.read_events import get_PN_image_file
from exod.utils.logger import logger
from exod.utils.plotting import plot_image, compare_images

import numpy as np
from cv2 import inpaint, INPAINT_NS
from skimage.draw import disk
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
from astropy.convolution import convolve, Gaussian2DKernel
from scipy.interpolate import interp1d


def compute_expected_cube_using_templates(data_cube):
    """Computes a baseline expected cube, combining background and sources. Any departure from this expected cube
    corresponds to variability.
    The background is dealt with by assuming that all GTIs and BTIs follow respective templates (i.e., once each frame
    is divided by its total counts, they all look the same).
    The sources are dealt with assuming they are constant. We take their net emission, and distribute it evenly across
    all frames."""
    logger.info(f'Computing Expected Cube using templates.')
    cube = data_cube.data
    bti_indices = data_cube.bti_bin_idx
    gti_indices = data_cube.gti_bin_idx
    sigma_blurring = 0.5

    # Get the summed Images.
    image_nan   = np.full(cube.shape[:2], np.nan) # nan image, used for np.where operations
    image_total = np.nansum(cube, axis=2)
    image_GTI   = np.nansum(cube[:,:,gti_indices], axis=2)
    image_GTI   = np.where(image_GTI > 0, image_GTI, image_nan) # Replace 0s with NAN

    # Create the source masks.
    # Two source masks are calculated then combined, the first comes from the pipeline detected
    # sources in the OBSMLI file. The second mask takes the remaining image and masks pixels that are
    # above 3 x 75% of the total image. These two masks are then combined using an OR statement.
    # This works well in most cases but often struggles when there are extended sources.
    image_mask_source_list = mask_known_sources(data_cube) # Image mask from OBSMLI file.
    # source_threshold = np.nanpercentile(image_GTI.flatten(), 99) # This or from detected sources
    source_threshold = 3 * np.nanpercentile(image_GTI[~image_mask_source_list], q=75)
    image_mask_source_percentile = image_GTI > source_threshold
    image_mask_source = image_mask_source_list | image_mask_source_percentile

    compare_images(images=[image_mask_source_list, image_mask_source_percentile, image_mask_source],
                   titles=['image_mask_source_list', 'image_mask_source_percentile', 'image_mask_source'])

    # inpaint on source regions
    image_GTI_no_source               = inpaint(image_GTI.astype(np.float32), image_mask_source.astype(np.uint8), inpaintRadius=2, flags=INPAINT_NS)
    count_GTI_outside_sources         = np.nansum(image_GTI[~image_mask_source]) # The total background count of GTIs, outside of the source regions
    image_GTI_no_source_template      = image_GTI_no_source / count_GTI_outside_sources
    image_GTI_no_source_template_blur = convolve(image_GTI_no_source_template, Gaussian2DKernel(sigma_blurring))


    compare_images(images=[image_GTI,
                           image_GTI_no_source,
                           image_GTI_no_source_template,
                           image_GTI_no_source_template_blur],
                   titles=['image_GTI',
                           'image_GTI_no_source',
                           'image_GTI_no_source_template',
                           'image_GTI_no_source_template_blur'],
                   log=True)

    # Perform the inpainting a second time to fill in missing pixels (gaps)
    image_mask_missing_pixels = (image_GTI > 0) & np.isnan(image_GTI_no_source_template_blur)
    image_GTI_no_source_template_blur_inpaint = inpaint(image_GTI_no_source_template_blur.astype(np.float32),
                                                        image_mask_missing_pixels.astype(np.uint8), inpaintRadius=2, flags=INPAINT_NS)

    # Remove the pixels that were inpainted outside the original image
    image_GTI_background_template = np.where(image_total > 0, image_GTI_no_source_template_blur_inpaint, image_nan)

    # Plot the steps
    compare_images(images=[image_GTI_no_source_template,
                           image_GTI_no_source_template_blur,
                           image_GTI_no_source_template_blur_inpaint,
                           image_GTI_background_template],
                   titles=['GTI no source template',
                           'GTI no source template blur',
                           'GTI no source template inpainted',
                           'final GTI background template'])


    # BTI template: compute the estimated BTI background
    if len(bti_indices) > 0:
        logger.info(f'len(bti_indices)>0 ({len(bti_indices)} > 0)')
        image_BTI                         = np.nansum(cube[:,:,bti_indices], axis=2) # Image of all BTIs combined, with sources
        image_BTI_no_source               = inpaint(image_BTI.astype(np.float32), image_mask_source.astype(np.uint8), inpaintRadius=2, flags=INPAINT_NS)
        count_BTI_outside_sources         = np.nansum(image_BTI[~image_mask_source])
        image_BTI_no_source_template      = image_BTI_no_source / count_BTI_outside_sources
        image_BTI_no_source_template_blur = convolve(image_BTI_no_source_template, Gaussian2DKernel(sigma_blurring))

        compare_images(images=[image_BTI, image_BTI_no_source, image_BTI_no_source_template, image_BTI_no_source_template_blur],
                       titles=['image_BTI', 'image_BTI_no_source', 'image_BTI_no_source_template', 'image_BTI_no_source_template_blur'])

        image_mask_missing_pixels = (image_GTI > 0) & np.isnan(image_GTI_no_source_template_blur)
        image_BTI_no_source_template_blur = inpaint(image_BTI_no_source_template_blur.astype(np.float32),
                                                    image_mask_missing_pixels.astype(np.uint8), inpaintRadius=2, flags=INPAINT_NS)
        image_BTI_no_source_template_blur = np.where(image_total > 0, image_BTI_no_source_template_blur, image_nan)

    # image_BTI_no_source_template_blur=image_BTI_no_source_template

    # Source contribution
    if len(bti_indices) < cube.shape[2]/2:
        logger.info(f'len(bti_indices)<cube.shape[2]/2 {len(bti_indices)}<{cube.shape[2]/2}')

        # Get the net image of sources in GTIs (after inpainting to have background below sources)
        source_only_image_GTI = np.where(image_mask_source, image_GTI-image_GTI_background_template*count_GTI_outside_sources, np.zeros(image_GTI.shape))
        source_only_image_GTI = np.where(source_only_image_GTI>0, source_only_image_GTI, np.zeros(image_GTI.shape))

        # Assume sources are constant, counts per frame are obtained by dividing by # of GTI frames
        source_constant_contribution = source_only_image_GTI / len(gti_indices)
    else:
        logger.info(f'len(bti_indices)={len(bti_indices)}')
        # Get the net image of sources in BTIs (after inpainting to have background below sources)
        source_only_image_BTI = np.where(image_mask_source, image_BTI-image_BTI_no_source_template_blur*count_GTI_outside_sources, np.zeros(image_BTI.shape))
        source_only_image_BTI = np.where(source_only_image_BTI>0, source_only_image_BTI,np.zeros(image_GTI.shape))
        source_constant_contribution = source_only_image_BTI / np.sum(np.nansum(cube[:,:,bti_indices], axis=(0,1)) > 0)

    #Create expected cube
    observed_cube_outside_sources = np.where(np.repeat(image_mask_source[:, :, np.newaxis], cube.shape[2], axis=2),
                                              np.empty(cube.shape) * np.nan,
                                              cube)
    lightcurve_outside_sources = np.nansum(observed_cube_outside_sources, axis=(0, 1))
    # plt.figure(figsize=(10, 3))
    # plt.title('lightcurve_outside_sources')
    # plt.plot(lightcurve_outside_sources)
    # plt.show()

    # We then create the estimated cube. For both GTI and BTI, we estimate their background by using the lightcurve and
    # the templates. We then add the expected constant source contribution.
    logger.info('Creating Estimated Cube')
    estimated_cube = np.empty(cube.shape)
    estimated_cube[:,:,gti_indices] = image_GTI_background_template[:,:,np.newaxis] * lightcurve_outside_sources[gti_indices]
    if len(bti_indices) > 0:
        estimated_cube[:,:,bti_indices] = image_BTI_no_source_template_blur[:,:,np.newaxis] * lightcurve_outside_sources[bti_indices]
    logger.info(f'lc_min={np.min(lightcurve_outside_sources)} lc_nanmin={np.nanmin(lightcurve_outside_sources)}')
    estimated_cube += np.repeat(source_constant_contribution[:,:,np.newaxis], cube.shape[2], axis=2)
    estimated_cube = np.where(np.nansum(cube, axis=(0,1)) > 0, estimated_cube, np.empty(cube.shape)*np.nan)
    return estimated_cube

def mask_known_sources(data_cube):
    obsid= data_cube.event_list.obsid
    path_source_file = data_processed / f'{obsid}'
    source_file_path = list(path_source_file.glob('*EP*OBSMLI*.FTZ'))[0]
    tab_src = Table(fits.open(source_file_path)[1].data)

    # We include bright (i.e. large DET_ML) point sources & extended sources
    point_sources = tab_src[(tab_src['EP_DET_ML']>8)&(tab_src['EP_EXTENT']==0.)]
    radius_point_sources = [cropping_radius_counts(data_cube,counts) for counts in point_sources['EP_TOT']]
    extended_sources = tab_src[(tab_src['EP_DET_ML']>8)&(tab_src['EP_EXTENT']>0.)]
    radius_extended_sources = extended_sources['EP_EXTENT']

    # Get Image Coordinates
    img_file = get_PN_image_file(obsid=obsid)
    hdul = fits.open(img_file)
    header = hdul[0].header
    wcs = WCS(header=header)

    mask = np.full(data_cube.shape[:2],False)
    # Plot
    # im = np.nansum(data_cube.data, axis=2)
    # plt.figure(figsize=(10, 10))
    for tab_data, tab_radius, color in zip((point_sources,extended_sources),
                                           (radius_point_sources,radius_extended_sources),
                                           ('r','y')):
        if len(tab_data)>0:
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

            # Interpolate to Cube coordinates & Round to int
            interp_x_cube = interp1d(x=data_cube.bin_x, y=range(data_cube.shape[0]))
            interp_y_cube = interp1d(x=data_cube.bin_y, y=range(data_cube.shape[1]))
            x_cube = interp_x_cube(X)
            y_cube = interp_y_cube(Y)

            # # Plot
            # im = np.nansum(data_cube.data, axis=2)
            # plt.figure(figsize=(10, 10))
            # plt.imshow(im.T, origin='lower', norm=LogNorm(), interpolation='none')
            # plt.scatter(x_cube, y_cube, color='red', marker='x', s=5)
            for x,y,rad in zip(x_cube, y_cube, tab_radius):
                radius_image = rad/data_cube.size_arcsec
                rr, cc = disk((x, y), radius_image)
                kept_pixels = (rr>0) & (rr<data_cube.shape[0]-1) & (cc>0) & (cc<data_cube.shape[1]-1) #Keep mask pixels only if in image
                rr, cc = rr[kept_pixels], cc[kept_pixels]
                mask[rr, cc]=True
                # im[rr,cc]=np.nan
                # circle=plt.Circle((x, y), radius_image, color=color,fill=False)
                # plt.gca().add_patch(circle)
    # plt.imshow(im.T, origin='lower', norm=LogNorm(vmax=1e3), interpolation='none')
    # plt.show()
    # for x, y in zip(x_cube, y_cube):
    #     im[x - 1:x + 1, y - 1:y + 1] = 0
    #
    # plt.figure(figsize=(10, 10))
    # plt.imshow(im.T, origin='lower', norm=LogNorm(), interpolation='none')
    return mask

def cropping_radius_counts(datacube, epic_total_rate):
    if epic_total_rate>1:
        return 80
    elif epic_total_rate>0.1:
        return 40
    else:
        return datacube.size_arcsec

if __name__=="__main__":

    from exod.pre_processing.download_observations import read_observation_ids
    from exod.utils.path import data
    obsids = read_observation_ids(data / 'observations.txt')
    for obsid in obsids[:5]:
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
            for ind_exp, subset_overlapping_exposures in enumerate(observation.events_overlapping_subsets):
                event_list = EventList.from_event_lists(subset_overlapping_exposures)
                # event_list = observation.events_processed_pn[0]
                # event_list.read()
                dl = DataLoader(event_list=event_list, size_arcsec=size_arcsec, time_interval=time_interval,
                                gti_only=gti_only,
                                gti_threshold=gti_threshold, min_energy=min_energy, max_energy=max_energy)
                dl.run()
                data_cube = dl.data_cube
                estimated_cube = compute_expected_cube_using_templates(data_cube)
        except FileNotFoundError:
            pass
        except KeyError:
            pass
        except Exception as e:
            print(e)


