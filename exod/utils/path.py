import os
from pathlib import Path
from exod.utils.logger import logger

base            = Path(os.environ['EXOD'])
data            = base / 'data'
data_raw        = data / 'raw'
data_processed  = data / 'processed'
data_results    = data / 'results'
data_combined   = data / 'results_combined'
logs            = base / 'logs'

all_paths = {
    'base': base,
    'data': data,
    'data_raw': data_raw,
    'data_processed': data_processed,
    'data_results': data_results,
    'data_combined': data_combined,
    'logs': logs
}

def create_all_paths():
    for name, path in all_paths.items():
        logger.info(f'Creating Path: {path}')
        os.makedirs(path, exist_ok=True)

def make_results_directory(obsid):
    data_results_obs = data_results / obsid
    logger.info(f'Creating dir {data_results_obs}')
    os.makedirs(data_results_obs, exist_ok=True)



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

if __name__ == "__main__":
    create_all_paths()
    for name, path in all_paths.items():
        exists = "exists" if path.exists() else "does not exist"
        logger.info(f"{name:<15} : {path} : {exists}")







