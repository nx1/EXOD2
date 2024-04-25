import random
import multiprocessing

from exod.processing.bayesian_pipeline import parameter_grid, Pipeline, combine_results
from exod.processing.bayesian_computations import PrecomputeBayesLimits
from exod.utils.logger import logger, get_current_date_string
from exod.utils.path import data, data_results, read_observation_ids
import exod.processing.detector as detector
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
    # Get Simulation time
    timestr = get_current_date_string()

    # Load observation IDs
    obsids = read_observation_ids(data / 'all_obsids.txt')
    random.shuffle(obsids)

    # Define the number of processes to use
    num_processes = 4

    # Create a pool of processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use the pool's map method to apply the process_params function to each set of parameters
        all_res = pool.map(process_params, parameter_grid(obsids=obsids))

    logger.info(f'EXOD Run Completed total observations: {len(obsids)}')
    df_results = pd.DataFrame(all_res)
    logger.info(f'df_results:\n{df_results}')
    savepath_csv = data_results / f'EXOD_simlist_{timestr}.csv'
    logger.info(f'Saving EXOD run results to: {savepath_csv}')
    df_results.to_csv(savepath_csv, index=False)

    combine_results(obsids=obsids)
