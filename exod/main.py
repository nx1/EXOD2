import random
import multiprocessing

from exod.processing.bayesian_computations import PrecomputeBayesLimits
from exod.utils.logger import logger, get_current_date_string
from exod.utils.path import data, data_results, read_observation_ids
import exod.processing.detector as detector
import exod.processing.bayesian_pipeline as bayesian
import pandas as pd


def process_obsid(obsid):
    args = {'obsid': obsid,
            'size_arcsec': 20.0,
            'time_interval': 50,
            'gti_threshold': 1.5,
            'min_energy': 2.0,
            'max_energy': 12.0,
            'gti_only': False,
            'remove_partial_ccd_frames': False,
            'clobber': False,
            'precomputed_bayes_limit': pre}
    res = args.copy()

    try:
        bayesian.run_pipeline(**args)
        res['status'] = 'Run'
    except Exception as e:
        logger.warning(f'Could not process obsid={obsid} {type(e).__name__} occurred: {e}')
        res['status'] = f'{type(e).__name__} | {e}'

    return res


if __name__ == "__main__":
    logger.warning('Before we start, have you run setsas? if not, Ctrl+C and go do it!')
    # input()


    # Get Simulation time
    timestr = get_current_date_string()

    # Load observation IDs
    obsids = read_observation_ids(data / '.txt')
    obsids = obsids[:8]
    # import random
    # random.shuffle(obsids)
    threshold_sigma = 3
    pre = PrecomputeBayesLimits(threshold_sigma=threshold_sigma)
    pre.load()
    
    all_res = []

    num_processes = 4
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_obsid, obsids)

    # for obsid in obsids:
    #     # obsid = '0109130501'
    #     obsid = '0112231801'
    #     res = process_obsid(obsid)

    for res in results:
        all_res.append(res)

    # for obsid in obsids:
    #     res = process_obsid(obsid=obsid)
    #     all_res.append(res)

    logger.info(f'EXOD Run Completed total observations: {len(obsids)}')
    df_results = pd.DataFrame(all_res)
    logger.info(f'df_results:\n{df_results}')
    savepath_csv = data_results / f'EXOD_simlist_{timestr}.csv'
    logger.info(f'Saving EXOD run results to: {savepath_csv}')
    df_results.to_csv(savepath_csv, index=False)
