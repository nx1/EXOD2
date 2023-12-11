import logging

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s')
logging.getLogger('matplotlib.font_manager').disabled = True

# Create a logger instance
logger = logging.getLogger(__name__)
