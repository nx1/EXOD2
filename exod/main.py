import random
import multiprocessing

from exod.processing.bayesian_pipeline import parameter_grid, Pipeline
from exod.processing.bayesian_computations import PrecomputeBayesLimits
from exod.utils.logger import logger, get_current_date_string
from exod.utils.path import data, data_results, read_observation_ids
import exod.processing.detector as detector
import exod.processing.bayesian_pipeline as bayesian
import pandas as pd

if __name__ == "__main__":
    # Get Simulation time
    timestr = get_current_date_string()

    # Load observation IDs
    obsids = read_observation_ids(data / 'observations.txt')
    random.shuffle(obsids)

    all_res = []
    for params in parameter_grid(obsids=obsids):
        res = params
        try:
            p = Pipeline(**params)
            p.run()
            p.list_results()
            res['status'] = 'Run'
            all_res.append(res)
        except Exception as e:
            logger.warning(f'Could not process obsid={obsid} {type(e).__name__} occurred: {e}')
            res['status'] = f'{type(e).__name__} | {e}'

    # num_processes = 4
    # with multiprocessing.Pool(processes=num_processes) as pool:
    #     results = pool.map(process_obsid, obsids)


    logger.info(f'EXOD Run Completed total observations: {len(obsids)}')
    df_results = pd.DataFrame(all_res)
    logger.info(f'df_results:\n{df_results}')
    savepath_csv = data_results / f'EXOD_simlist_{timestr}.csv'
    logger.info(f'Saving EXOD run results to: {savepath_csv}')
    df_results.to_csv(savepath_csv, index=False)
