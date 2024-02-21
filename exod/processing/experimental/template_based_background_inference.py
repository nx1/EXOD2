import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os
from exod.pre_processing.data_loader import DataLoader
from exod.xmm.observation import Observation
from exod.utils.path import data_processed
from exod.utils.synthetic_data import create_fake_burst
from cv2 import inpaint, INPAINT_NS, filter2D
from scipy.stats import poisson
from tqdm import tqdm
import cmasher as cmr
from scipy.stats import poisson

def compute_expected_cube_using_templates(cube, rejected):
    """Computes a baseline expected cube, combining background and sources. Any departure from this expected cube
    corresponds to variability.
    The background is dealt with by assuming that all GTIs and BTIs follow respective templates (i.e., once each frame
    is divided by its total counts, they all look the same).
    The sources are dealt with assuming they are constant. We take their net emission, and distribute it evenly across
    all frames."""

    #GTI template: compute the estimated GTI background
    kept = [ind for ind in range(cube.shape[2]) if ind not in rejected] #Indices of GTIs
    image_GTI = np.sum(cube[:,:,kept], axis=2) #Image of all GTIs combined, with sources
    source_threshold=np.nanpercentile(image_GTI.flatten(), 99) #This or from detected sources
    boolean_mask_source = image_GTI > source_threshold #Mask to get only sources afterwards

    countGTI_outside_sources = np.nansum(image_GTI[~boolean_mask_source]) #The total background count of GTIs, outside of the source regions
    mask_source = np.uint8(boolean_mask_source[:, :, np.newaxis]) #Convert the boolean mask to int8 for inpainting
    no_source_image_GTI = inpaint(image_GTI.astype(np.float32)[:,:,np.newaxis], mask_source, 2, flags=INPAINT_NS) #Remove and inpaint on source regions
    background_GTI_template = no_source_image_GTI/countGTI_outside_sources #Divide the inpainted background by the total count to have a template


    #BTI template: compute the estimated BTI background
    image_BTI = np.nansum(cube[:,:,rejected], axis=2) #Image of all BTIs combined, with sources
    countBTI_outside_sources = np.nansum(image_BTI[~boolean_mask_source]) #The total background count of BTIs, outside of the source regions
    no_source_image_BTI = inpaint(image_BTI.astype(np.float32)[:,:,np.newaxis], mask_source, 2, flags=INPAINT_NS) #Remove and inpaint on source regions
    background_BTI_template = no_source_image_BTI/countBTI_outside_sources #Divide the inpainted background by the total count to have a template


    #Source contribution
    source_only_image_GTI = image_GTI-no_source_image_GTI #Get the net image of sources in GTIs (after inpainting to have background below sources)
    source_constant_contribution = source_only_image_GTI/len(kept) #Assume sources are constant, counts per frame are obtained by dividing by # of GTI frames


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
    estimated_cube[:,:,kept] = background_GTI_template[:,:,np.newaxis]*lightcurve_outside_sources[kept]
    estimated_cube[:,:,rejected] = background_BTI_template[:,:,np.newaxis]*lightcurve_outside_sources[rejected]
    estimated_cube += np.repeat(source_constant_contribution[:,:,np.newaxis],cube.shape[2],axis=2)

    return estimated_cube

def compute_variability(observed_cube, estimated_cube):
    """We have access to an expected and observed cubes. Variability is residuals between them.
    It can be positive or negative, and exceed thresholds several times. V_cube is the basic tool to exploit for this,
    it corresponds to the residuals cube. A variability map can be built then, taking the largest excess (> or < 0)."""
    V_cube = (observed_cube-estimated_cube)/np.sqrt(observed_cube+estimated_cube)
    V_map_positive = np.nanmax(V_cube, axis=2)
    V_map_negative = np.nanmin(V_cube, axis=2)
    final_V_map = np.where(V_map_positive>-V_map_negative, V_map_positive, V_map_negative)
    return final_V_map

def compute_likelihood_variability(observed_cube, estimated_cube):
    """We have access to an expected and observed cubes. Variability is the log likelihood of observing counts above
    (resp. below) the observed value, knowing it follows a Poisson variable of expectation the estimated_cube.
    Here, this log likelihood takes negative values for eclipses. We take the largest of both possibilities."""
    V_cube = np.where(observed_cube>estimated_cube,
                      -np.log10(1-poisson.cdf(observed_cube,estimated_cube)),
                      +np.log10(poisson.cdf(observed_cube,estimated_cube)))
    image_V_min = np.nanmin(V_cube, axis=2)
    image_V_max = np.nanmax(V_cube, axis=2)
    final_V_map = np.where(image_V_max>-image_V_min, image_V_max, image_V_min)
    return final_V_map

if __name__=="__main__":
    obsid='0886121001'#'0831790701' #
    size_arcsec = 10
    time_interval = 1000
    gti_only = False
    gti_threshold = 0.5
    min_energy = 0.5
    max_energy = 12.0

    observation = Observation(obsid)
    observation.get_files()

    event_list = observation.events_processed_pn[0]
    event_list.read()

    img = observation.images[0]
    img.read(wcs_only=True)

    dl = DataLoader(event_list=event_list, size_arcsec=size_arcsec, time_interval=time_interval, gti_only=gti_only,
                    gti_threshold=gti_threshold, min_energy=min_energy, max_energy=max_energy)
    dl.run()

    cube = dl.data_cube.data
    rejected = dl.data_cube.bti_bin_idx


    estimated_cube = compute_expected_cube_using_templates(cube, rejected)
    image, expected_image = np.nansum(cube, axis=2), np.nansum(estimated_cube, axis=2)


    # fig, axes = plt.subplots(1,3)
    # axes[0].imshow(image, norm=LogNorm())
    # axes[1].imshow(expected_image, norm=LogNorm())
    # axes[2].imshow((image-expected_image)/np.sqrt(image+expected_image), vmin=-1, vmax=1)
    # plt.show()

    Vmap = np.where(expected_image>0,compute_variability(cube, estimated_cube),np.empty(cube.shape[:2])*np.nan)
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(image, norm=LogNorm(), interpolation='none')
    axes[1].imshow(expected_image, norm=LogNorm(), interpolation='none')
    m=axes[2].imshow(Vmap, vmin=-3, vmax=3, cmap='cmr.redshift_r', interpolation='none')
    cbar=plt.colorbar(mappable=m, ax=axes[2],fraction=0.046, pad=0.04)
    cbar.set_label(r'Max residuals ($\sigma$)')
    plt.show()

    Vmap = np.where(expected_image>0,compute_likelihood_variability(cube, estimated_cube),np.empty(cube.shape[:2])*np.nan)
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(image, norm=LogNorm(), interpolation='none')
    axes[1].imshow(expected_image, norm=LogNorm(), interpolation='none')
    m=axes[2].imshow(Vmap, vmin=-4, vmax=4, cmap='cmr.redshift_r', interpolation='none')
    cbar=plt.colorbar(mappable=m, ax=axes[2],fraction=0.046, pad=0.04)
    cbar.set_label(r'Transient log likelihood')
    plt.show()

    #Check result on frames. This uses pre-computed frames, run the code in compute_expected_cube_using_templates in the console to use this
    # for frame_index in range(cube.shape[2]):
    #     fig, axes = plt.subplots(2, 2)
    #     image = cube[:, :, frame_index]
    #     expected_image = estimated_cube[:, :, frame_index]
    #     # image = np.where(cube[:,:,frame_index]>0,cube[:,:,frame_index],np.nan)
    #     # expected_image=np.where(estimated_cube[:,:,frame_index]>0,estimated_cube[:,:,frame_index],np.nan)
    #     axes[0][0].imshow(image, norm=LogNorm(), interpolation='none')
    #     axes[0][1].imshow(expected_image, norm=LogNorm(), interpolation='none')
    #     m=axes[1][0].imshow((image - expected_image) / np.sqrt(image + expected_image), vmin=-2, vmax=2, cmap='cmr.redshift_r', interpolation='none')
    #     cbar = plt.colorbar(mappable=m, ax=axes[1][0], fraction=0.046, pad=0.04)
    #     cbar.set_label(r'Max residuals ($\sigma$)')
    #
    #     poisson_cdf = poisson.cdf(image,expected_image)
    #     residuals_poisson = np.where(image>expected_image,-np.log10(1-poisson.cdf(image,expected_image)),+np.log10(poisson.cdf(image,expected_image)))
    #     m=axes[1][1].imshow(residuals_poisson, vmin=-4, vmax=4, cmap='cmr.redshift_r', interpolation='none')
    #     cbar = plt.colorbar(mappable=m, ax=axes[1][1], fraction=0.046, pad=0.04)
    #     cbar.set_label(r'Transient log likelihood')
    #     fig.set_figwidth(10)
    #     fig.set_figheight(10)
    #
    #     for ax in axes.flatten():
    #         ax.axis("off")
    #     plt.show()



