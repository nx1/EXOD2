import logging

# Configure the logger
fmt1 = '%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s'
fmt2 = '%(levelname)s - %(asctime)s - %(message)s' 
logging.basicConfig(level=logging.INFO, format=fmt2)
logging.getLogger('matplotlib.font_manager').disabled = True

# Create a logger instance
logger = logging.getLogger(__name__)
