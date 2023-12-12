import os
from exod.pre_processing.filtering_data import filter_events_file,filter_events_file_gti_only
from exod.pre_processing.read_events_files import read_EPIC_events_file
from exod.processing.variability_computation import compute_pixel_variability
from exod.post_processing.extract_variability_regions import extract_variability_regions, get_regions_sky_position, \
    plot_variability_with_regions
from exod.post_processing.testing_variability import compute_proba_constant, plot_lightcurve_alerts
from exod.post_processing.save_transient_sources import save_list_transients
from exod.utils.path import data_results


def detect_transients(obsid, time_interval=1000, size_arcsec=10, box_size=3, gti_only=False, emin=0.2, emax=12.):
    cube,coordinates_XY = read_EPIC_events_file(obsid, size_arcsec, time_interval,box_size, gti_only=gti_only,
                                                emin=emin, emax=emax)
    variability_map = compute_pixel_variability(cube)
    if f'{time_interval}s' not in os.listdir(os.path.join(data_results,obsid)):
        os.makedirs(os.path.join(data_results,obsid,f'{time_interval}s'))
    plot_variability_with_regions(variability_map, 8,
                                os.path.join(data_results,obsid,f'{time_interval}s','VariabilityRegions.png'))
    tab_centersofmass, bboxes = extract_variability_regions(variability_map, 8)
    tab_ra, tab_dec, tab_X, tab_Y=get_regions_sky_position(obsid, tab_centersofmass, coordinates_XY)
    tab_p_values = compute_proba_constant(cube, bboxes)
    plot_lightcurve_alerts(cube, bboxes,time_interval)
    save_list_transients(obsid, tab_ra, tab_dec, tab_X, tab_Y, tab_p_values, time_interval)

if __name__ == "__main__":
    # Pre processing
    # filter_events_file_gti_only('0038541101', min_energy=0.2, max_energy=12)

    detect_transients('0831790701',200,15,3,True, emin=0.2,emax=2.)
