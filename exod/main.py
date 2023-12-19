import os
from tqdm import tqdm
from exod.pre_processing.filtering_data import filter_events_file
from exod.pre_processing.read_events_files import read_EPIC_events_file
from exod.processing.variability_computation import compute_pixel_variability,convolve_variability_GaussianBlur
from exod.post_processing.extract_variability_regions import extract_variability_regions, get_regions_sky_position, \
    plot_variability_with_regions
from exod.post_processing.testing_variability import compute_proba_constant, plot_lightcurve_alerts,plot_lightcurve_alerts_with_background
from exod.post_processing.save_transient_sources import save_list_transients
from exod.utils.path import data_results,data_processed
from exod.processing.experimental.background_estimate import experimental_variability_Poisson_likelihood, compute_background_two_templates
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

def detect_transients(obsid, time_interval=1000, size_arcsec=10, box_size=3, gti_only=False,
                      emin=0.2, emax=12., threshold=8):
    cube,coordinates_XY,rejected = read_EPIC_events_file(obsid, size_arcsec, time_interval,box_size, gti_only=gti_only,
                                                emin=emin, emax=emax)
    #variability_map = compute_pixel_variability(cube)
    cube_background, cube_background_withsource = compute_background_two_templates(cube,rejected)
    variability_map = experimental_variability_Poisson_likelihood(cube,cube_background_withsource)

    variability_map = convolve_variability_GaussianBlur(variability_map, sigma=1)
    if f'{int(time_interval)}s' not in os.listdir(os.path.join(data_results,obsid)):
        os.makedirs(os.path.join(data_results,obsid,f'{int(time_interval)}s'))
    centers, bboxes = extract_variability_regions(variability_map, threshold)
    plot_variability_with_regions(variability_map, threshold, centers, bboxes,
                                os.path.join(data_results,obsid,f'{int(time_interval)}s','VariabilityRegions.png'))
    # tab_ra, tab_dec, tab_X, tab_Y=get_regions_sky_position(obsid, tab_centersofmass, coordinates_XY)
    # tab_p_values = compute_proba_constant(cube, bboxes)
    # if gti_only:
    #     plot_lightcurve_alerts(cube, bboxes,time_interval)
    # # else:
    plot_lightcurve_alerts_with_background(cube, cube_background,cube_background_withsource,bboxes,time_interval)
    # save_list_transients(obsid, tab_ra, tab_dec, tab_X, tab_Y, tab_p_values, time_interval)

if __name__ == "__main__":
    from exod.utils.make_animation import make_gif
    tab_images=[]
    for time_interval in tqdm(np.geomspace(5,10000,50)):
    # for time_interval in [100,200,1000]:# tqdm([10, 50, 200, 500, 1000, 5000]):
        detect_transients('0831790701',time_interval,15,3,False,
                          emin=0.2,emax=12., threshold=20)
        tab_images.append(os.path.join(data_results,'0831790701',f'{int(time_interval)}s','VariabilityRegions.png'))
    make_gif(tab_images,os.path.join(data_results,'0831790701','AllVariabilityImages.gif'),duration=500)