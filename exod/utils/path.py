"""
EXOD2/ 
│
├── data/
│   ├── raw/
│   │   ├── downloaded_data/
│   │   │   ├── observation_id_1/
│   │   │   ├── observation_id_2/
│   ├── processed/
│   │       ├── observation_id_1/
│   │       ├── observation_id_2/
│   └── results/
│       ├── observation_id_1/
│       │   ├── result_file_1.csv
│       │   ├── result_file_2.fits
│       ├── observation_id_2/
│       │   ├── result_file_1.txt
│       │   ├── result_file_2.txt
├── logs/
│   ├── download_logs/
│   ├── process_logs/
│   └── results_logs/
"""
from pathlib import Path
from exod.utils.logger import logger

module         = Path(__file__).parent.parent            # contains all .py files : /EXOD2/exod
base           = module.parent             # Top level directory    : /EXOD2/
data           = base / 'data'      # Data Path              : /EXOD2/data
data_raw       = data / 'raw'              # This should point to folder with observations eg raw/0001730201
data_processed = data / 'processed'        # This will contain processed files from the raw observation processed/0001730201
data_results   = data / 'results'          # This file will contain the results for each obsid results/
data_combined  = data / 'results_combined' # Contains Combined results from /results/
logs           = base / 'logs'             # Contains log files.

all_paths = {
    'module': module,
    'base': base,
    'data': data,
    'data_raw': data_raw,
    'data_processed': data_processed,
    'data_results': data_results,
    'data_combined': data_combined,
    'logs': logs
}

if __name__ == "__main__":
    for name, path in all_paths.items():
        exists = "exists" if path.exists() else "does not exist"
        logger.info(f"{name:<15} : {path} : {exists}")







