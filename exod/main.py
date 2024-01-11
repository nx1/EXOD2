import matplotlib.pyplot as plt
import pandas as pd

from exod.pre_processing.event_filtering import filter_obsid_events
from exod.pre_processing.download_observations import download_observation_events
from exod.processing.variability import extract_var_regions, get_regions_sky_position, \
    plot_var_with_regions, get_region_lightcurves, calc_KS_poission, plot_region_lightcurves, calc_var_img
from exod.utils.logger import logger, get_current_date_string


def detect_transients(obsid, metric='v_score', combine_events=True, **kwargs):
    """
    Parameters
    ----------
    obsid  : Observation ID
    metric : 'v_score', or 'l_score'
    combine_events : Combine events from each instrument
    kwargs :

    Returns
    -------
    """
    implemented_methods = ['v_score', 'l_score']
    logger.info(f'Detecting Transients using metric={metric} combine_events={combine_events}')

    # Set Default parameters
    logger.info('Setting Default common parameters')
    common_params = {'time_interval' : 1000,
                     'size_arcsec'   : 10,
                     'gti_only'      : False,
                     'min_energy'    : 0.2,
                     'max_energy'    : 12.0,
                     'clobber'       : False}

    logger.info('Updating default params from kwargs')
    common_params.update(kwargs)

    if metric == 'v_score':
        detect_transients_v_score(obsid=obsid, **kwargs)
    elif metric == 'l_score':
        detect_transients_l_score()
    else:
        raise ValueError(f'metric must be one of {implemented_methods} not {metric}!')


def detect_transients_l_score():
    raise NotImplementedError

def detect_transients_v_score(obsid, time_interval=1000, size_arcsec=10,
                              gti_only=False, min_energy=0.2,
                              max_energy=12.0, clobber=False):

    # Filter the events files
    filter_obsid_events(obsid=obsid,
                        min_energy=min_energy,
                        max_energy=max_energy,
                        clobber=clobber)

    # Read the event files and create the data cube
    cube, coordinates_XY = read_EPIC_events_file(obsid=obsid, size_arcsec=size_arcsec, time_interval=time_interval,
                                                 gti_only=gti_only, min_energy=min_energy, max_energy=max_energy)

    var_img = calc_var_img(cube=cube)

    # Get the dataframe describing the contiguous variable regions
    df_regions = extract_var_regions(var_img=var_img)

    # Calculate the sky coordinates from the detected regions
    df_sky = get_regions_sky_position(obsid=obsid, coordinates_XY=coordinates_XY, df_regions=df_regions)

    # Combine the Two
    df_regions = pd.concat([df_regions, df_sky], axis=1)
    logger.info(f'df_regions:\n{df_regions}')

    lcs = get_region_lightcurves(cube, df_regions)

    ks_results = [calc_KS_poission(lc) for lc in lcs]
    df_regions['KS_stat'] = [k.statistic for k in ks_results]
    df_regions['KS_pval'] = [k.pvalue for k in ks_results]
    df_regions['KS_loc']  = [k.statistic_location for k in ks_results]
    df_regions['KS_sign'] = [k.statistic_sign for k in ks_results]
    df_regions['obsid']   = obsid
    df_regions['time_interval'] = time_interval

    logger.info(f'df_regions:\n{df_regions}')
    df_regions_savepath = data_results / obsid / 'detected_regions.csv'
    df_regions.to_csv(df_regions_savepath, index=False)


    plot_outfile = data_results / f'{obsid}' / 'var_img.png'
    plot_var_with_regions(var_img=var_img, df_regions=df_regions, outfile=plot_outfile)
    

    if len(df_regions)<20:
        plot_region_lightcurves(lcs=lcs, df_regions=df_regions, obsid=obsid)
    #else:
    #    cube_background, cube_background_withsource = compute_background(cube)
    #    plot_lightcurve_alerts_with_background(cube, cube_background, cube_background_withsource, bboxes)

    #plt.show()

if __name__ == "__main__":
    from exod.pre_processing.download_observations import read_observation_ids
    from exod.pre_processing.read_events import read_EPIC_events_file
    from exod.utils.path import data, data_results
    import random

    # Get Simulation time
    timestr = get_current_date_string() 

    # Load observation IDs
    obsids = read_observation_ids(data / 'observations.txt')
    random.shuffle(obsids)

    all_res = []
    for obsid in obsids:
        args = {'obsid'         : obsid,
                'size_arcsec'   : 20,
                'time_interval' : 100,
                'gti_only'      : True,
                'min_energy'    : 0.2,
                'max_energy'    : 12,
                'clobber'       : False}

        res = args.copy()

        # detect_transients(**args, metric='v_score', combine_events=True)
        try:
            detect_transients(**args, metric='v_score', combine_events=True)
            # detect_transients(**args, metric='l_score', combine_events=True)
            res['status'] = 'Run'
        except Exception as e:
            logger.warning(f'Could not process obsid={obsid} {type(e).__name__} occured: {e}')
            res['status'] = f'{type(e).__name__ }| {e}'
        all_res.append(res)

    logger.info(f'EXOD Run Completed total observations: {len(obsids)}')

    df_results = pd.DataFrame(all_res)
    logger.info(f'df_results:\n{df_results}')

    savepath_csv = data_results / f'EXOD_simlist_{timestr}.csv'
    logger.info(f'Saving EXOD run results to: {savepath_csv}')
    
    df_results.to_csv(savepath_csv, index=False)
