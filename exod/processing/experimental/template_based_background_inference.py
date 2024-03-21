from exod.pre_processing.data_loader import DataLoader
from exod.xmm.event_list import EventList
from exod.xmm.observation import Observation
from exod.utils.path import data_processed
from exod.pre_processing.read_events import get_PN_image_file

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from cv2 import inpaint, INPAINT_NS, INPAINT_TELEA, filter2D
from scipy.ndimage import gaussian_filter
from skimage.draw import disk
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
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
    cube=data_cube.data
    bti_indices=data_cube.bti_bin_idx

    sigma_blurring = 0.5
    image_total = np.nansum(cube,axis=2)

    #GTI template: compute the estimated GTI background
    gti_indices = [ind for ind in range(cube.shape[2]) if ind not in bti_indices] #Indices of GTIs
    image_GTI = np.nansum(cube[:,:,gti_indices], axis=2)
    image_GTI = np.where(image_GTI>0,image_GTI,np.full(cube.shape[:2],np.nan))#Image of all GTIs combined, with sources
    # source_threshold=np.nanpercentile(image_GTI.flatten(), 99) #This or from detected sources
    # boolean_mask_source = image_GTI > source_threshold #Mask to get only sources afterwards
    boolean_mask_source = mask_known_sources(data_cube)
    boolean_mask_source = boolean_mask_source|(image_GTI>3*np.nanpercentile(image_GTI[~boolean_mask_source],75))

    countGTI_outside_sources = np.nansum(image_GTI[~boolean_mask_source]) #The total background count of GTIs, outside of the source regions
    mask_source = np.uint8(boolean_mask_source[:, :, np.newaxis]) #Convert the boolean mask to int8 for inpainting
    no_source_image_GTI = inpaint(image_GTI.astype(np.float32)[:,:,np.newaxis], mask_source, 2, flags=INPAINT_NS) #Remove and inpaint on source regions
    unblurred_background_GTI_template = no_source_image_GTI/countGTI_outside_sources #Divide the inpainted background by the total count to have a template
    blurred_background_GTI_template = convolve(unblurred_background_GTI_template, Gaussian2DKernel(sigma_blurring))
    mask_missing_pixels = np.uint8(((image_GTI>0)&np.isnan(blurred_background_GTI_template))[:, :, np.newaxis])
    blurred_background_GTI_template = inpaint(blurred_background_GTI_template.astype(np.float32)[:,:,np.newaxis], mask_missing_pixels, 2, flags=INPAINT_NS)
    background_GTI_template = np.where(image_total>0, blurred_background_GTI_template ,np.empty(image_total.shape)*np.nan)
    # background_GTI_template=unblurred_background_GTI_template

    #BTI template: compute the estimated BTI background
    if len(bti_indices)>0:
        image_BTI = np.nansum(cube[:,:,bti_indices], axis=2) #Image of all BTIs combined, with sources
        countBTI_outside_sources = np.nansum(image_BTI[~boolean_mask_source]) #The total background count of BTIs, outside of the source regions
        no_source_image_BTI = inpaint(image_BTI.astype(np.float32)[:,:,np.newaxis], mask_source, 2, flags=INPAINT_NS) #Remove and inpaint on source regions
        unblurred_background_BTI_template = no_source_image_BTI/countBTI_outside_sources #Divide the inpainted background by the total count to have a template
        blurred_background_BTI_template = convolve(unblurred_background_BTI_template, Gaussian2DKernel(sigma_blurring))
        mask_missing_pixels = np.uint8(((image_GTI > 0) & np.isnan(blurred_background_GTI_template))[:, :, np.newaxis])
        blurred_background_BTI_template = inpaint(blurred_background_BTI_template.astype(np.float32)[:, :, np.newaxis],
                                                  mask_missing_pixels, 2, flags=INPAINT_NS)
        background_BTI_template = np.where(image_total>0, blurred_background_BTI_template, np.empty(image_total.shape)*np.nan)
    # background_BTI_template=unblurred_background_BTI_template

    #Source contribution
    if len(bti_indices)<cube.shape[2]/2:
        source_only_image_GTI = np.where(boolean_mask_source, image_GTI-background_GTI_template*countGTI_outside_sources,np.zeros(image_GTI.shape))#Get the net image of sources in GTIs (after inpainting to have background below sources)
        source_only_image_GTI = np.where(source_only_image_GTI>0, source_only_image_GTI,np.zeros(image_GTI.shape))
        source_constant_contribution = source_only_image_GTI / len(gti_indices)  # Assume sources are constant, counts per frame are obtained by dividing by # of GTI frames
    else:
        source_only_image_BTI = np.where(boolean_mask_source, image_BTI-background_BTI_template*countGTI_outside_sources,np.zeros(image_BTI.shape)) #Get the net image of sources in BTIs (after inpainting to have background below sources)
        source_only_image_BTI = np.where(source_only_image_BTI>0, source_only_image_BTI,np.zeros(image_GTI.shape))
        source_constant_contribution = source_only_image_BTI / np.sum(np.nansum(cube[:,:,bti_indices], axis=(0,1))>0)

    #Create expected cube
    # We don't create this cube because of memory usage. We do it below in a single line
    # observed_cube_outside_sources = np.where(np.repeat(boolean_mask_source[:, :, np.newaxis], cube.shape[2], axis=2),
    #                                          np.empty(cube.shape) * np.nan,
    #                                          cube)
    # lightcurve_outside_sources = np.nansum(observed_cube_outside_sources, axis=(0, 1))
    lightcurve_outside_sources = np.nansum(np.where(np.repeat(boolean_mask_source[:,:,np.newaxis],cube.shape[2],axis=2),
                                                    np.empty(cube.shape) * np.nan,
                                                    cube),
                                           axis=(0, 1)) #Lightcurve of background, outside of source regions
    #We then create the estimated cube. For both GTI and BTI, we estimate their background by using the lightcurve and
    #the templates. We then add the expected constant source contribution
    estimated_cube = np.empty(cube.shape)
    estimated_cube[:,:,gti_indices] = background_GTI_template[:,:,np.newaxis]*lightcurve_outside_sources[gti_indices]
    if len(bti_indices)>0:
        estimated_cube[:,:,bti_indices] = background_BTI_template[:,:,np.newaxis]*lightcurve_outside_sources[bti_indices]
    # print(np.min(lightcurve_outside_sources), np.nanmin(lightcurve_outside_sources))
    estimated_cube += np.repeat(source_constant_contribution[:,:,np.newaxis],cube.shape[2],axis=2)
    estimated_cube=np.where(np.nansum(cube,axis=(0,1))>0,estimated_cube,np.empty(cube.shape)*np.nan)

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
                data_cube=dl.data_cube
                estimated_cube=compute_expected_cube_using_templates(data_cube)
        except FileNotFoundError:
            pass
        except KeyError:
            pass


