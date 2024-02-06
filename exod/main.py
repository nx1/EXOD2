from exod.pre_processing.data_loader import DataLoader
from exod.pre_processing.event_filtering import filter_obsid_events, create_obsid_images
from exod.pre_processing.download_observations import download_observation_events
from exod.processing.variability import extract_var_regions, get_regions_sky_position, \
    plot_var_with_regions, get_region_lightcurves, calc_KS_poission, plot_region_lightcurves, \
    calc_var_img, plot_cube_statistics, filter_df_regions
from exod.utils.logger import logger, get_current_date_string
from exod.utils.path import data_results 
from exod.xmm.observation import Observation

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def run_pipeline(obsid, time_interval=1000, size_arcsec=10,
                 gti_only=False, gti_threshold=1.5, min_energy=0.2,
                 max_energy=12.0, sigma=5, clobber=False):

    # download_observation_events(obsid=obsid)

    # Filter the events files and Create Images
    filter_obsid_events(obsid=obsid, min_energy=min_energy, max_energy=max_energy, clobber=clobber)
    create_obsid_images(obsid=obsid, clobber=clobber)

    # Create the Observation class
    observation = Observation(obsid)
    observation.get_files()

    # Get the eventslist & image to use
    for evt in observation.events_processed:
        if 'PI' in evt.filename:
            event_list = evt

    img = observation.images[0]
    img.read(wcs_only=True)

    event_list.read()

    # Initialize the Data Loader
    data_loader = DataLoader(event_list=event_list,
                             size_arcsec=size_arcsec,
                             time_interval=time_interval,
                             gti_only=gti_only,
                             gti_threshold=gti_threshold,
                             min_energy=min_energy,
                             max_energy=max_energy)
    data_loader.run()

    df_bti = data_loader.df_bti
    df_bti_savepath = observation.path_results / 'bti.csv'
    logger.info(f'Saving df_regions to {df_bti_savepath}')
    df_bti.to_csv(df_bti_savepath, index=False)

    # Create Data Cube
    data_cube = data_loader.data_cube
    data_cube.video()
    # plot_cube_statistics(data_cube.data)

    """
    #################################
    # Bright Source Masking (perhaps do this iteratively)
    img_cr = np.nansum(data_cube.data, axis=2) / event_list.exposure
    q_val = 99.95 #99.9936 #99.73
    cr_bright_threshold = np.percentile(a=img_cr, q=q_val)
    img_cr_mask = img_cr > cr_bright_threshold
    print(cr_bright_threshold)

    fig, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].set_title('Count rate')
    ax[1].set_title(f'Mask CR_threshold={cr_bright_threshold}')
    im = ax[0].imshow(img_cr.T, origin='lower', interpolation='none')
    fig.colorbar(im, ax=ax[0])
    ax[1].imshow(img_cr_mask.T, origin='lower', interpolation='none')
    plt.show()

    ################################
    """

    # Calculate the Variability Image
    var_img = calc_var_img(cube=data_cube.data)

    # Get the dataframe describing the contiguous variable regions
    df_regions = extract_var_regions(var_img=var_img, sigma=sigma)

    # Calculate the sky coordinates from the detected regions
    df_regions = get_regions_sky_position(df_regions=df_regions, wcs=img.wcs, data_cube=data_cube)
    df_regions = filter_df_regions(df_regions)

    # Create Region Dataframe
    logger.info(f'df_regions:\n{df_regions}')
    df_regions_savepath = observation.path_results / 'regions.csv'
    logger.info(f'Saving df_regions to {df_regions_savepath}')
    df_regions.to_csv(df_regions_savepath, index=False)

    # Create Lightcurve dataframe 
    df_lcs = get_region_lightcurves(data_cube, df_regions)
    logger.info(f'df_lcs:\n{df_lcs}')
    df_lcs_savepath = observation.path_results / 'lcs.csv'
    logger.info(f'Saving df_lcs to {df_lcs_savepath}')
    df_lcs.to_csv(df_lcs_savepath, index=False)

    """
    # Calculate KS value #TODO move to post-processing?
    ks_results = [calc_KS_poission(lc) for lc in lcs]
    df_regions['KS_stat'] = [k.statistic for k in ks_results]
    df_regions['KS_pval'] = [k.pvalue for k in ks_results]
    df_regions['KS_loc']  = [k.statistic_location for k in ks_results]
    df_regions['KS_sign'] = [k.statistic_sign for k in ks_results]
    """

    # Plot Variable Regions
    plot_outfile = observation.path_results / 'var_img.png'
    plot_var_with_regions(var_img=var_img, df_regions=df_regions, outfile=plot_outfile)

    # Plot Lightcurves files
    if len(df_regions) < 50:
        plot_region_lightcurves(df_lcs=df_lcs, df_regions=df_regions, obsid=obsid)

    obs_info = observation.info
    evt_info = event_list.info
    dl_info = data_loader.info
    dc_info = data_cube.info
    plt.show()


if __name__ == "__main__":
    from exod.pre_processing.download_observations import read_observation_ids
    from exod.utils.path import data, data_results
    import random

    # Get Simulation time
    timestr = get_current_date_string() 

    # Load observation IDs
    obsids = read_observation_ids(data / 'observations.txt')
    # random.shuffle(obsids)

    all_res = []
    for obsid in obsids:
        args = {'obsid'         : obsid,
                'size_arcsec'   : 10.0,
                'time_interval' : 50,
                'gti_only'      : True,
                'gti_threshold' : 1.5,
                'min_energy'    : 0.2,
                'max_energy'    : 12.0,
                'sigma'         : 5,
                'clobber'       : False}

        res = args.copy()

        run_pipeline(**args)
        try:
            run_pipeline(**args)
            res['status'] = 'Run'
        except Exception as e:
            logger.warning(f'Could not process obsid={obsid} {type(e).__name__} occurred: {e}')
            res['status'] = f'{type(e).__name__ } | {e}'
        all_res.append(res)

    logger.info(f'EXOD Run Completed total observations: {len(obsids)}')

    df_results = pd.DataFrame(all_res)
    logger.info(f'df_results:\n{df_results}')

    savepath_csv = data_results / f'EXOD_simlist_{timestr}.csv'
    logger.info(f'Saving EXOD run results to: {savepath_csv}')
    
    df_results.to_csv(savepath_csv, index=False)
