from glob import glob
import tarfile
from exod.utils import path
from exod.utils.logger import logger

logger.info('Extracting Downloaded files')
dl_path = path.data_downloaded
extract_path = path.data_raw
logger.info(f'Looking in Path: {dl_path}')
logger.info(f'Extraction_path: {extract_path}')


wildcard = f'{dl_path}/*.tar'

tar_gz_files = glob(wildcard)
tar_gz_files=tar_gz_files[1:]
print(tar_gz_files)

N_files = len(tar_gz_files)
if N_files == 0:
    logger.info('No files found to extract')
    exit()

logger.info(f'.tar files to process: {N_files} press any key to start Ctrl+C to exit')

for f in tar_gz_files:
    logger.info(f'extracting file: {f} --> {extract_path}')
    with tarfile.open(f, 'r') as tar:
        tar.extractall(extract_path)



