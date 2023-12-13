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

if __name__ == "__main__":
    create_all_paths()
    for name, path in all_paths.items():
        exists = "exists" if path.exists() else "does not exist"
        logger.info(f"{name:<15} : {path} : {exists}")







