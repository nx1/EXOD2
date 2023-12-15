import os
from tqdm import tqdm
from exod.pre_processing.read_events_files import read_EPIC_events_file
from exod.processing.variability_computation import compute_pixel_variability,convolve_variability
from exod.post_processing.extract_variability_regions import extract_variability_regions, get_regions_sky_position, \
    plot_variability_with_regions
from exod.post_processing.testing_variability import compute_proba_constant, plot_lightcurve_alerts
from exod.post_processing.save_transient_sources import save_list_transients
from exod.utils.path import data_results
from exod.utils.logger import logger


def detect_transients(obsid, time_interval=1000, size_arcsec=10, box_size=3, gti_only=False, min_energy=0.2, max_energy=12.):
    # Read the event files and create the data cube
    cube,coordinates_XY = read_EPIC_events_file(obsid, size_arcsec, time_interval, box_size, gti_only=gti_only,
                                                min_energy=min_energy, max_energy=max_energy)
    # Calculate the variability Map
    variability_map = compute_pixel_variability(cube)

    # Save the variability_map with regions
    plot_outfile = data_results / f'{obsid}' / f'{time_interval}s' / 'VariabilityRegions.png'
    plot_variability_with_regions(variability_map=variability_map, threshold=8, outfile= plot_outfile)
    
    # Calculate the center of masses
    tab_centersofmass, bboxes = extract_variability_regions(variability_map=variability_map, threshold=8)
    logger.info(f'tab_centersofmass: {tab_centersofmass}')
    logger.info(f'bboxes: {bboxes}')

    tab_ra, tab_dec, tab_X, tab_Y = get_regions_sky_position(obsid=obsid, tab_centersofmass=tab_centersofmass, coordinates_XY=coordinates_XY)

    logger.info(f'tab_ra {tab_ra}')
    logger.info(f'tab_dec {tab_dec}')
    logger.info(f'tab_X {tab_X}')
    logger.info(f'tab_Y {tab_Y}')

    tab_p_values = compute_proba_constant(cube, bboxes)
    logger.info(tab_p_values)
    if gti_only:
        plot_lightcurve_alerts(cube, bboxes,time_interval)
    # else:
    #     cube_background, cube_background_withsource = compute_background(cube)
    #     # plot_lightcurve_alerts_with_background(cube, cube_background,cube_background_withsource,bboxes)
    # save_list_transients(obsid, tab_ra, tab_dec, tab_X, tab_Y, tab_p_values, time_interval)

if __name__ == "__main__":
    from exod.pre_processing.download_observations import read_observation_ids
    from exod.pre_processing.read_events_files import read_EPIC_events_file
    from exod.utils.path import data

    obsids = read_observation_ids(data / 'observations.txt')
    for obsid in obsids:
        args = {'obsid':obsid,
                'size_arcsec':15,
                'time_interval':10000,
                'box_size':3,
                'gti_only':True,
                'min_energy':0.2,
                'max_energy':12}
        try:
            detect_transients(**args)
        except Exception as e:
            logger.info(f'Could not process obsid={obsid} {e}')
