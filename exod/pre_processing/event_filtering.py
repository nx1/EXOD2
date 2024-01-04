"""
Requires having pre-set 'setsas' and 'export CCFPATH=' in the terminal.
"""
import os

from exod.utils.path import data_raw, data_processed
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


def get_raw_event_files(obsid):
    logger.info(f'Getting raw event files for obsid: {obsid}')
    path_raw_obs = data_raw / obsid
    event_files = list(path_raw_obs.glob('*EVLI*FTZ'))
    logger.info(f'Found {len(event_files)} files.')
    return event_files


def filter_PN_events_file(infile, outfile, min_energy=0.2, max_energy=12.0, clobber=False):
    logger.info(f'Filtering PN Events file: \n raw       : {infile} \n processed : {outfile}')
    if outfile.exists() and clobber is False:
        logger.info(f'File {outfile} exists and clobber={clobber}!')
    else:
        min_PI, max_PI = int(min_energy*1000), int(max_energy*1000)
        cmd=(f'evselect table={infile} withfilteredset=Y filteredset={outfile} destruct=Y keepfilteroutput=T '
             f'expression="#XMMEA_EP && (PATTERN<=4) && (PI in [{min_PI}:{max_PI}])" -V 0')
        run_cmd(cmd)


def filter_M1_events_file(infile, outfile, min_energy=0.2, max_energy=12., clobber=False):
    logger.info(f'Filtering Events file: \n raw       : {infile} \n processed : {outfile}')
    if outfile.exists() and clobber is False:
        logger.info(f'File {outfile} exists and clobber={clobber}!')
    else:
        min_PI, max_PI = int(min_energy*1000), int(max_energy*1000)
        cmd=(f'evselect table={infile} withfilteredset=Y filteredset={outfile} destruct=Y keepfilteroutput=T '
             f'expression="#XMMEA_EM && (PATTERN<=12) && (PI in [{min_PI}:{max_PI}])" -V 0')
        run_cmd(cmd)


def filter_M2_events_file(infile, outfile, min_energy=0.2, max_energy=12., clobber=False):
    logger.info(f'Filtering Events file: \n raw       : {infile} \n processed : {outfile}')
    if outfile.exists() and clobber is False:
        logger.info(f'File {outfile} exists and clobber={clobber}!')
    else:
        min_PI, max_PI = int(min_energy*1000), int(max_energy*1000)
        cmd=(f'evselect table={infile} withfilteredset=Y filteredset={outfile} destruct=Y keepfilteroutput=T '
             f'expression="#XMMEA_EM && (PATTERN<=12) && (PI in [{min_PI}:{max_PI}])" -V 0')
        run_cmd(cmd)


def filter_obsid_events(obsid, min_energy=0.2, max_energy=12.0, clobber=False):
    path_raw_obs, path_processed_obs = get_raw_and_processed_obs_path(obsid)
    event_files = get_raw_event_files(obsid)
    for raw_filepath in event_files:
        stem = raw_filepath.stem  # The stem is the name of the file without extensions
        filtered_filename = stem + '_FILT.fits' # The output filename
        filtered_filepath = path_processed_obs / filtered_filename # The output filepath
        if 'PN' in stem:
            filter_PN_events_file(infile=raw_filepath, outfile=filtered_filepath,
                                  min_energy=min_energy, max_energy=max_energy, clobber=clobber)
        if 'M1' in stem:
            filter_M1_events_file(infile=raw_filepath, outfile=filtered_filepath,
                                  min_energy=min_energy, max_energy=max_energy, clobber=clobber)
        if 'M2' in stem:
            filter_M2_events_file(infile=raw_filepath, outfile=filtered_filepath,
                                  min_energy=min_energy, max_energy=max_energy, clobber=clobber)


def create_image_file(infile, outfile, ximagebinsize=80, yimagebinsize=80, clobber=False):
    logger.info(f'Filtering Events file: \n raw       : {infile} \n processed : {outfile}')
    logger.info(f'ximagebinsize = {ximagebinsize} yimagebinsize = {yimagebinsize}')
    if (outfile.exists() and clobber is False):
        logger.info(f'File {outfile} exists and clobber={clobber}!')
    else:
        cmd = (f'evselect table={infile} imagebinning=binSize imageset={outfile} withimageset=yes xcolumn=X ycolumn=Y'
               f' ximagebinsize={ximagebinsize} yimagebinsize={yimagebinsize} -V 0')
        run_cmd(cmd)

def create_obsid_images(obsid, ximagebinsize=80, yimagebinsize=80, clobber=False):
    path_raw_obs, path_processed_obs = get_raw_and_processed_obs_path(obsid)
    event_files = get_raw_event_files(obsid)
    for raw_filepath in event_files:
        stem = raw_filepath.stem  # The stem is the name of the file without extensions
        img_filename = stem + '_IMG.fits' # The output filename
        img_filepath = path_processed_obs / img_filename # The output filepath
        create_image_file(infile=raw_filepath, outfile=img_filepath,
                          ximagebinsize=ximagebinsize, yimagebinsize=yimagebinsize, clobber=clobber)

if __name__ == "__main__":
    from exod.pre_processing.download_observations import read_observation_ids
    from exod.utils.path import data
    obsids = read_observation_ids(data / 'observations.txt')
    for obsid in obsids:
        filter_obsid_events(obsid=obsid)
        create_obsid_images(obsid=obsid)

  
