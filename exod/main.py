from exod.pre_processing.data_loader import DataLoader
from exod.pre_processing.event_filtering import filter_obsid_events, create_obsid_images
from exod.utils.logger import logger, get_current_date_string
from exod.utils.path import save_df
from exod.xmm.observation import Observation
from exod.processing.detector import Detector, plot_var_with_regions, plot_region_lightcurves

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
    event_list = observation.events_processed_pn[0]
    event_list.read()

    img = observation.images[0]
    img.read(wcs_only=True)

    # Initialize the Data Loader
    dl = DataLoader(event_list=event_list,
                    size_arcsec=size_arcsec,
                    time_interval=time_interval,
                    gti_only=gti_only,
                    gti_threshold=gti_threshold,
                    min_energy=min_energy,
                    max_energy=max_energy)
    dl.run()

    # Create Data Cube
    # dl.data_cube.plot_cube_statistics()
    dl.data_cube.video()

    # Detection
    detector = Detector(data_cube=dl.data_cube, wcs=img.wcs, sigma=sigma)
    detector.run()
    plot_var_with_regions(var_img=detector.image_var, df_regions=detector.df_regions, savepath=observation.path_results / 'image_var.png')
    plot_region_lightcurves(df_lcs=detector.df_lcs, df_regions=detector.df_regions, savedir=observation.path_results)

    # Save Results
    save_df(df=dl.df_bti, savepath=observation.path_results / 'bti.csv')
    save_df(df=detector.df_lcs, savepath=observation.path_results / 'lcs.csv')
    save_df(df=detector.df_regions, savepath=observation.path_results / 'regions.csv')
    obs_info = observation.info
    evt_info = event_list.info
    dl_info = dl.info
    dc_info = dl.data_cube.info
    det_info = detector.info
    plt.show()


if __name__ == "__main__":
    from exod.pre_processing.download_observations import read_observation_ids
    from exod.utils.path import data, data_results

    # Get Simulation time
    timestr = get_current_date_string() 

    # Load observation IDs
    obsids = read_observation_ids(data / 'observations.txt')
    # random.shuffle(obsids)

    all_res = []
    for obsid in obsids:
        args = {'obsid'         : obsid,
                'size_arcsec'   : 10.0,
                'time_interval' : 1000,
                'gti_only'      : True,
                'gti_threshold' : 0.5,
                'min_energy'    : 0.2,
                'max_energy'    : 12.0,
                'sigma'         : 4,
                'clobber'       : False}

        res = args.copy()

        # run_pipeline(**args)
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
