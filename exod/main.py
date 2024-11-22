"""
Main script to run the EXOD pipeline.

This script will run the EXOD pipeline for all the observations in the `data/observations.txt` file.
"""
import random
import multiprocessing

from exod.processing.pipeline import parameter_grid, Pipeline, combine_results
from exod.utils.logger import logger, get_current_date_string
from exod.utils.path import data, data_results, read_observation_ids

import pandas as pd
import matplotlib.pyplot as plt

from exod.xmm.observation import Observation
from exod.utils.event_list_viewer import EventListViewer


def process_params(params):
    """
    Wrapper for handling one set of EXOD parameters provided as a dictionary.
    A set of parameters is simply a dictionary with the key as the parameter name
    and the value as the value for the parameter.

    This function is useful for when looping over parameters or wrapping it in a
    multiprocessing call.

    Args:
        params (dict): {'obsid':XXXXX, 'size_arcsec':20, ...}

    Returns:
        res (dict): params with run status added. either 'Run' for good or error for bad.
    """
    res = params.copy()
    try:
        p = Pipeline(**params)
        p.run()
        p.load_results()
        res['status'] = 'Run'
    except Exception as e:
        logger.warning(f'Could not process obsid={params["obsid"]} {type(e).__name__} occurred: {e}')
        res['status'] = f'{type(e).__name__} | {e}'
    return res

if __name__ == "__main__":
    obsids = read_observation_ids(data / 'observations.txt')#
    # random.shuffle(obsids)

    for obsid in obsids:
        obsid = '0724210501'
        p = Pipeline(obsid=obsid, size_arcsec=20, time_interval=5, remove_partial_ccd_frames=True, min_energy=0.2, max_energy=12.0)
        p.run()
        # p.load_results()
        plt.show()
        #for evt in p.observation.events_processed:
        #    elv = EventListViewer(evt.data)
        #    elv.show()

    # Use Multiprocessing
    # num_processes = 1
    # with multiprocessing.Pool(processes=num_processes) as pool:
    #     all_res = pool.map(process_params, parameter_grid(obsids=obsids))

    # Use Serial Processing
    # all_res = []
    # for params in parameter_grid(obsids=obsids):
    #     res = process_params(params)
    #     all_res.append(res)


    # logger.info(f'EXOD Run Completed total observations: {len(obsids)}')
    # df_results = pd.DataFrame(all_res)
    # logger.info(f'df_results:\n{df_results}')
    # savepath_csv = data_results / f'EXOD_simlist_{get_current_date_string()}.csv'

    # logger.info(f'Saving EXOD run results to: {savepath_csv}')
    # df_results.to_csv(savepath_csv, index=False)
    # combine_results(obsids=obsids)
