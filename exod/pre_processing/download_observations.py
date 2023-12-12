import requests

from exod.utils import path
from exod.utils.logger import logger

def read_observation_ids(file_path):
    """
    Read observation IDs from file.
    Each line should be a single obsid.
    """
    with open(file_path, 'r') as file:
        obs_ids = [line.strip() for line in file.readlines()]
    return obs_ids


obs_list_path = path.data / 'observations.txt'
output_sh_path = path.data / 'download_obs.sh'
save_dir = path.data_downloaded

logger.info(f'Observations List Path: {obs_list_path}')
logger.info(f'Download Script Path: {output_sh_path}')
logger.info(f'Save Directory: {save_dir}')

logger.info(f'Reading observations from {obs_list_path}')
observation_ids = read_observation_ids(obs_list_path)
logger.info(f'Found {len(observation_ids)} observations ids')

for obs in observation_ids:
    download_url = f'https://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno={obs}&level=PPS'
    file_path = save_dir / f'{obs}.tar'

    # Use requests to download the file
    response = requests.get(download_url)
    logger.info(response)
    if response.status_code == 200:
        logger.info('Response 200, downloading to {file_path}')
        with open(file_path, 'wb') as file:
            file.write(response.content)
        logger.info(f'Downloaded: {file_path}')
    else:
        logger.warning(f'Failed to download: {file_path}')

