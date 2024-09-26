"""
Main script to run the EXOD pipeline.

This script will run the EXOD pipeline for all the observations in the `data/observations.txt` file.

"""
import random
import multiprocessing

from exod.processing.bayesian_pipeline import parameter_grid, Pipeline, combine_results
from exod.utils.logger import logger, get_current_date_string
from exod.utils.path import data, data_results, read_observation_ids
import pandas as pd

def process_params(params):
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
    obsids = read_observation_ids(data / 'observations.txt')

    p = Pipeline(obsid=obsids[0], size_arcsec=20, time_interval=15, min_energy=0.2, max_energy=12)
    p.run()




    # random.shuffle(obsids)

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
