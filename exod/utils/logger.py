import os
from pathlib import Path
import logging
from datetime import datetime

class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    light_blue = "\x1b[94m"
    reset = "\x1b[0m"

    # Different logger format options for da whole crew :D (pick one)
    fmt1 = '%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s'
    fmt2 = '%(levelname)s - %(asctime)s - %(message)s'
    fmt3 = '%(message)s'
    format = fmt3

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO:  light_blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def get_current_date_string():
    current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    return current_date

# Configure the logger
#logging.basicConfig(level=logging.INFO, format=fmt1)
logging.getLogger('matplotlib.font_manager').disabled = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler
current_date = get_current_date_string()
log_filepath = Path(os.environ['EXOD']) / 'logs' / f'exod_{current_date}.log'
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
#ch.setLevel(logging.WARNING)


# Create file handler
fh = logging.FileHandler(log_filepath)


ch.setFormatter(CustomFormatter())
fh.setFormatter(CustomFormatter())

logger.addHandler(ch)
logger.addHandler(fh)
