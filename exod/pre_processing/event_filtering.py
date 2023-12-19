"""
Requires having pre-set 'setsas' and 'export CCFPATH=' in the terminal.
"""
import os
import shlex
import subprocess

from exod.utils.path import data_raw, data_processed, check_file_exists
from exod.utils.logger import logger

def run_cmd(cmd):
    logger.info('Running Command with os.system:')
    logger.info(cmd)
    ret = os.system(cmd)
    logger.info(f'Return Code: {ret}')
    if ret != 0:
        raise OSError(f'Failed to run command!: {cmd} \n Return code: {ret}')
    else:
        return ret



def get_raw_and_processed_obs_path(obsid):
    path_raw_obs       = data_raw / f'{obsid}'
    path_processed_obs = data_processed / f'{obsid}'
    os.makedirs(path_processed_obs, exist_ok=True)
    return path_raw_obs, path_processed_obs

def filter_PN_file(obsid, min_energy=0.2, max_energy=12., clobber=False):
    path_raw_obs, path_processed_obs = get_raw_and_processed_obs_path(obsid)
    raw_PN_file        = list(path_raw_obs.glob('*PIEVLI*'))[0]
    clean_PN_file      = path_processed_obs / f'PN_pattern_clean.fits'
    imagePN            = path_processed_obs / f'PN_image.fits'

    check_file_exists(clean_PN_file, clobber=clobber)

    min_PI, max_PI = int(min_energy*1000), int(max_energy*1000)

    cmd=(f'evselect table={raw_PN_file} withfilteredset=Y filteredset={clean_PN_file} destruct=Y keepfilteroutput=T '
         f'expression="#XMMEA_EP && (PATTERN<=4) && (PI in [{min_PI}:{max_PI}])" -V 0')
    run_cmd(cmd)

    cmd=(f'evselect table={clean_PN_file} imagebinning=binSize imageset={imagePN} withimageset=yes xcolumn=X ycolumn=Y'
         f' ximagebinsize=80 yimagebinsize=80 -V 0')
    run_cmd(cmd)

def filter_M1_file(obsid, min_energy=0.2, max_energy=12., clobber=False):
    path_raw_obs, path_processed_obs = get_raw_and_processed_obs_path(obsid)
    raw_M1_file        = list(path_raw_obs.glob('*M1*EVLI*'))[0]
    clean_M1_file      = path_processed_obs / f'M1_pattern_clean.fits'
    imageM1            = path_processed_obs / f'M1_image.fits'

    check_file_exists(clean_M1_file, clobber=clobber)

    min_PI, max_PI = int(min_energy*1000), int(max_energy*1000)

    cmd=(f'evselect table={raw_M1_file} withfilteredset=Y filteredset={clean_M1_file} destruct=Y keepfilteroutput=T '
         f'expression="#XMMEA_EM && (PATTERN<=12) && (PI in [{min_PI}:{max_PI}])" -V 0')
    run_cmd(cmd)

    cmd=(f'evselect table={clean_M1_file} imagebinning=binSize imageset={imageM1} withimageset=yes xcolumn=X ycolumn=Y'
         f' ximagebinsize=80 yimagebinsize=80 -V 0')
    run_cmd(cmd)


def filter_M2_file(obsid, min_energy=0.2, max_energy=12., clobber=False):
    path_raw_obs, path_processed_obs = get_raw_and_processed_obs_path(obsid)
    raw_M2_file        = list(path_raw_obs.glob('*M2*EVLI*'))[0]
    clean_M2_file      = path_processed_obs / f'M2_pattern_clean.fits'
    imageM2            = path_processed_obs / f'M2_image.fits'

    check_file_exists(clean_M2_file, clobber=clobber)

    min_PI, max_PI = int(min_energy*1000), int(max_energy*1000)

    cmd=(f'evselect table={raw_M2_file} withfilteredset=Y filteredset={clean_M2_file} destruct=Y keepfilteroutput=T '
         f'expression="#XMMEA_EM && (PATTERN<=12) && (PI in [{min_PI}:{max_PI}])" -V 0')
    run_cmd(cmd)

    cmd=(f'evselect table={clean_M2_file} imagebinning=binSize imageset={imageM2} withimageset=yes xcolumn=X ycolumn=Y'
         f' ximagebinsize=80 yimagebinsize=80 -V 0')
    run_cmd(cmd)



def filter_observation_events(obsid, min_energy=0.2, max_energy=12., clobber=False):
    """
    Filter observation event lists for PN, M1 and M2.

    Parameters:
    - obsid (str): Observation identifier.
    - min_energy (float): Minimum energy threshold.
    - max_energy (float): Maximum energy threshold.

    Returns:
    - dict: Dictionary indicating the status of each filter.
    """

    filter_result = {'obsid':obsid, 'PN': 'Not Run', 'M1': 'Not Run', 'M2': 'Not Run'}

    filters = [
        {'name': 'PN', 'function': filter_PN_file},
        {'name': 'M1', 'function': filter_M1_file},
        {'name': 'M2', 'function': filter_M2_file}
    ]

    for f in filters:
        try:
            filter_func = f['function']
            filter_func(obsid, min_energy=min_energy, max_energy=max_energy, clobber=clobber)
            filter_result[f['name']] = 'Success'
        except IndexError as e:
            logger.warning(f'Could not process {f["name"]} event file for obsid={obsid} {e}')
            filter_result[f['name']] = 'IndexError'
    return filter_result


if __name__ == "__main__":
    from exod.pre_processing.download_observations import read_observation_ids
    from exod.utils.path import data
    import pandas as pd

    obsids = read_observation_ids(data / 'observations.txt')
    all_res = []
    for obs in obsids:
        res = filter_observation_events(obs, min_energy=0.2, max_energy=12., clobber=True)
        all_res.append(res)

    df = pd.DataFrame(all_res)
    print(df)

  
