import os
import subprocess
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

def download_observation_events(observation_id, clobber=False):
    """
    Download the post-processed event lists for PN, M1 and M2.
    """
    url_PN = f'http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno={obs}&instname=PN&level=PPS&name=PIEVLI'
    url_M1 = f'http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno={obs}&instname=M1&level=PPS&name=MIEVLI'
    url_M2 = f'http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno={obs}&instname=M2&level=PPS&name=MIEVLI'

    urls = {'PN':url_PN ,
            'M1':url_M1,
            'M2':url_M2}

    for inst, download_url in urls.items():
        response = requests.get(download_url)
        logger.info(response)
        if response.status_code == 200:
            # Get the filename from the response header
            cd = response.headers.get('content-disposition')
            filename = cd.split('filename=')[1].strip('";')

            # Create the folder to save to
            obs_path  = path.data_raw / f'{obs}'
            os.makedirs(obs_path, exist_ok=True)

            file_path = obs_path / f'{filename}'
            if file_path.is_file() and not clobber:
                logger.info(f'Skipping {file_path}. File already exists.')
            else:
                logger.info(f'Response 200, downloading to {file_path}')
                with open(file_path, 'wb') as file:
                    file.write(response.content)
                logger.info(f'Downloaded: {file_path}')

                # Deal with GUEST.tar files (these show up if you have multiple eventlists in an obs)
                if 'GUEST' in filename:
                    logger.info(f'GUEST tar file found! Extracting to current dir!')
                    cmd = f'tar -xvf {file_path} -C {obs_path} --strip-components=2'
                    logger.info(f'Executing: {cmd}')
                    subprocess.run(cmd, shell=True)

        else:
            logger.warning(f'Failed to download event files for: {obs} {inst}')


if __name__ == "__main__":
    clobber = False
    
    obs_list_path  = path.data / 'observations.txt'
    output_sh_path = path.data / 'download_obs.sh'

    logger.info(f'Observations List Path: {obs_list_path}')
    logger.info(f'Download Script Path: {output_sh_path}')

    logger.info(f'Reading observations from {obs_list_path}')
    observation_ids = read_observation_ids(obs_list_path)
    logger.info(f'Found {len(observation_ids)} observations ids')
    
    for obs in observation_ids:
        download_observation_events(observation_id=obs, clobber=clobber)

