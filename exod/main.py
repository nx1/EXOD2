from exod.utils.logger import logger, get_current_date_string
from exod.utils.path import data, data_results, read_observation_ids
import exod.processing.detector as detector
import exod.processing.bayesian as bayesian
import pandas as pd

if __name__ == "__main__":
    # Get Simulation time
    timestr = get_current_date_string()

    # Load observation IDs
    obsids = read_observation_ids(data / 'observations.txt')

    all_res = []
    for obsid in obsids:
        args = {'obsid': obsid,
                'size_arcsec': 20.0,
                'time_interval': 10,
                'gti_threshold': 1.5,
                'min_energy': 0.2,
                'max_energy': 10.0,
                'threshold_sigma': 3,
                'gti_only': False,
                'remove_partial_ccd_frames': True,
                'clobber': False}

        res = args.copy()

        # bayesian.run_pipeline(**args)
        try:
            bayesian.run_pipeline(**args)
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
