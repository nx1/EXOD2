#TODO: Use position, amplitude, time, KS and Chi2, match XMM and Simbad of the transient candidates, and save all of that
from astropy.table import Table
import os
import numpy as np
from exod.utils.path import data_results
from exod.post_processing.match_transient_candidates import xmm_lookup, simbad_lookup
from exod.post_processing.testing_variability import compute_proba_constant

def save_list_transients(obsid, tab_ra, tab_dec, tab_X, tab_Y,tab_p_values,time_interval):
    tab_simbad_names, tab_simbad_precise_type,tab_simbad_types, tab_simbad_sep = simbad_lookup(tab_ra, tab_dec)
    tab_xmm_names, tab_xmm_var, tab_xmm_sep = xmm_lookup(tab_ra, tab_dec)
    alert_id = np.arange(len(tab_ra))
    data=np.array([alert_id,[obsid]*len(tab_ra), tab_ra, tab_dec, tab_X, tab_Y,tab_p_values, tab_xmm_names, tab_xmm_var,
                   tab_xmm_sep,tab_simbad_names,tab_simbad_precise_type,tab_simbad_types,tab_simbad_sep])
    names=['AlertID','ObsID','RA','DEC','X','Y','KS_p_value','4XMM_IAUNAME','4XMM_SCVARFLAG','4XMM_AngSep',
           'Simbad_Name','Simbad_PreciseType','Simbad_GeneralType','Simbad_AngSep']
    data = np.transpose(data)
    t=Table(data, names=names)
    t.write(os.path.join(data_results,obsid,f'{time_interval}s','EXOD_Alerts.fits'), overwrite=True)

if __name__=='__main__':
    import matplotlib
    matplotlib.use('Agg')
    from exod.utils.path import data_processed
    from exod.pre_processing.read_events_files import read_EPIC_events_file
    from exod.processing.variability_computation import compute_pixel_variability
    from exod.post_processing.extract_variability_regions import extract_variability_regions, get_regions_sky_position,plot_variability_with_regions
    from exod.post_processing.testing_variability import compute_proba_constant, plot_lightcurve_alerts

    cube,coordinates_XY = read_EPIC_events_file('0831790701', 10, 100,3, gti_only=True)
    variability_map = compute_pixel_variability(cube)
    plot_variability_with_regions(variability_map, 8,
                                   os.path.join(data_results,'0831790701',f'{100}s','plot_test_varregions.png'))
    tab_centersofmass, bboxes = extract_variability_regions(variability_map, 8)
    tab_ra, tab_dec, tab_X, tab_Y=get_regions_sky_position('0831790701', tab_centersofmass, coordinates_XY)
    tab_p_values = compute_proba_constant(cube, bboxes)
    plot_lightcurve_alerts(cube, bboxes,100)
    save_list_transients('0831790701', tab_ra, tab_dec, tab_X, tab_Y, tab_p_values,100)