import os
from pathlib import Path
from exod.utils.logger import logger

base            = Path(os.environ['EXOD'])
data            = base / 'data'
utils           = base / 'exod' / 'utils'
data_raw        = data / 'raw'
data_processed  = data / 'processed'
data_results    = data / 'results'
data_combined   = data / 'results_combined'
data_util       = data / 'util'
data_plots      = data / 'plots'
logs            = base / 'logs'
docs            = base / 'docs'

all_paths = {
    'base'          : base,
    'data'          : data,
    'utils'         : utils,
    'data_raw'      : data_raw,
    'data_processed': data_processed,
    'data_results'  : data_results,
    'data_combined' : data_combined,
    'data_util'     : data_util,
    'data_plots'    : data_plots,
    'logs'          : logs,
    'docs'          : docs
}

savepaths_combined = {'bti'      : data_combined / 'df_bti.csv',
                      'regions'  : data_combined / 'df_regions.csv',
                      'lc'       : data_combined / 'df_lc.csv',
                      'run_info' : data_combined / 'df_run_info.csv',
                      'obs_info' : data_combined / 'df_obs_info.csv',
                      'dl_info'  : data_combined / 'df_dl_info.csv',
                      'dc_info'  : data_combined / 'df_dc_info.csv',
                      'evt_info' : data_combined / 'df_evt_info.csv'}
def create_all_paths():
    for name, path in all_paths.items():
        logger.info(f'Creating Path: {path}')
        os.makedirs(path, exist_ok=True)


def check_file_exists(file_path, clobber=True):
    """
    Check if a file exists and raise FileExistsError if clobber is False.

    Parameters:
    - file_path (str or Path): The path to the file.
    - clobber (bool): If True, overwrite the file if it exists.

    Raises:
    - FileExistsError: If the file exists and clobber is False.
    """
    if not clobber and Path(file_path).exists():
        raise FileExistsError(f'File {file_path} exists and clobber={clobber}!')


def read_observation_ids(file_path):
    """
    Read observation IDs from file.
    Each line should be a single observation.
    """
    with open(file_path, 'r') as file:
        obs_ids = [line.strip() for line in file.readlines()]
    return obs_ids


if __name__ == "__main__":
    create_all_paths()
    for name, path in all_paths.items():
        exists = "exists" if path.exists() else "does not exist"
        logger.info(f"{name:<15} : {path} : {exists}")



