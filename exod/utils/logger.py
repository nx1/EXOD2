import logging


class CustomFormatter(logging.Formatter):
    fmt1 = '%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s'
    fmt2 = '%(levelname)s - %(asctime)s - %(message)s' 

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    light_blue = "\x1b[94m"
    reset = "\x1b[0m"
    format = fmt1

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



# Configure the logger
#logging.basicConfig(level=logging.INFO, format=fmt1)
logging.getLogger('matplotlib.font_manager').disabled = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)
