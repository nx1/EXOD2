from glob import glob
import re
import subprocess

from exod.utils.logger import logger
from exod.utils import path

if __name__ == '__main__':
    logger.info('Extracting Downloaded files')
    dl_path = path.data_downloaded
    logger.info(f'Looking in Path: {dl_path}')

    wildcard = f'{dl_path}/*.gz'

    gz_files = glob(wildcard)
    gz_files = gz_files[1:]
    print(gz_files)

    N_files = len(gz_files)
    if N_files == 0:
        logger.info('No files found to extract')
        exit()

    logger.info(f'.tar files to process: {N_files} press any key to start Ctrl+C to exit')

    for f in gz_files:
        obsid = re.search(pattern=r'\d{10}', string=f).group()
        extract_path = path.data_raw / obsid

        logger.info(f'extracting file: {f} obsid={obsid} --> {extract_path}')

        command = ["gunzip", "-k", "-N", "0002970201_PN.gz"]

        subprocess.call(command)




