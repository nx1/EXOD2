"""
Requires having pre-set 'setsas' and 'export CCFPATH=' in the terminal.
"""
import os

from exod.utils.path import data_raw, data_processed, read_observation_ids
from exod.utils.logger import logger


def run_cmd(cmd):
    logger.info('Running Command with os.system:')
    logger.info(cmd)
    ret = os.system(cmd)
    logger.info(f'Return Code: {ret}')
    if ret != 0:
        raise OSError(f'Failed to run command!: {cmd}\nReturn code: {ret}')
    else:
        return ret


def check_for_timing_mode(filename):
    if 'TI' in filename:
        raise NotImplementedError(f'Timing mode is not supported!')


def filter_PN_events_file(infile, outfile, min_energy=0.2, max_energy=12.0, clobber=False):
    if outfile.exists() and clobber is False:
        logger.info(f'File {outfile} exists and clobber={clobber}!')
        return None

    logger.info(f'Filtering PN Events file:\nraw       : {infile}\nprocessed : {outfile}')
    min_PI, max_PI = int(min_energy*1000), int(max_energy*1000)
    cmd=(f'evselect table={infile} withfilteredset=Y filteredset={outfile} destruct=Y keepfilteroutput=T '
         f'expression="#XMMEA_EP && (PATTERN<=4) && (PI in [{min_PI}:{max_PI}])" -V 0')
    run_cmd(cmd)


def filter_M1_events_file(infile, outfile, min_energy=0.2, max_energy=12., clobber=False):
    if outfile.exists() and clobber is False:
        logger.info(f'File {outfile} exists and clobber={clobber}!')
        return None

    logger.info(f'Filtering Events file:\nraw       : {infile}\nprocessed : {outfile}')
    min_PI, max_PI = int(min_energy*1000), int(max_energy*1000)
    cmd=(f'evselect table={infile} withfilteredset=Y filteredset={outfile} destruct=Y keepfilteroutput=T '
         f'expression="#XMMEA_EM && (PATTERN<=12) && (PI in [{min_PI}:{max_PI}])" -V 0')
    run_cmd(cmd)


def filter_M2_events_file(infile, outfile, min_energy=0.2, max_energy=12., clobber=False):
    if outfile.exists() and clobber is False:
        logger.info(f'File {outfile} exists and clobber={clobber}!')
        return None
    logger.info(f'Filtering Events file:\nraw       : {infile}\nprocessed : {outfile}')
    min_PI, max_PI = int(min_energy*1000), int(max_energy*1000)
    cmd=(f'evselect table={infile} withfilteredset=Y filteredset={outfile} destruct=Y keepfilteroutput=T '
         f'expression="#XMMEA_EM && (PATTERN<=12) && (PI in [{min_PI}:{max_PI}])" -V 0')
    run_cmd(cmd)


def filter_obsid_events(observation, min_energy=0.2, max_energy=12.0, clobber=False):
    observation.get_files()

    for event in observation.events_raw:
        raw_filepath = event.path
        stem = raw_filepath.stem  # The stem is the name of the file without extensions
        check_for_timing_mode(stem)
        filtered_filename = stem + '_FILT.fits' # The output filename
        filtered_filepath = observation.path_processed / filtered_filename # The output filepath
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
    if (outfile.exists() and clobber is False):
        logger.info(f'File {outfile} exists and clobber={clobber}!')
        return None

    logger.info(f'Creating image file:\nraw       : {infile}\nprocessed : {outfile}')
    logger.info(f'ximagebinsize = {ximagebinsize} yimagebinsize = {yimagebinsize}')
    cmd = (f'evselect table={infile} imagebinning=binSize imageset={outfile} withimageset=yes xcolumn=X ycolumn=Y'
           f' ximagebinsize={ximagebinsize} yimagebinsize={yimagebinsize} -V 0')
    run_cmd(cmd)


def create_obsid_images(observation, ximagebinsize=80, yimagebinsize=80, clobber=False):
    observation.get_files()
    for event in observation.events_raw:
        raw_filepath = event.path
        stem = raw_filepath.stem  # The stem is the name of the file without extensions
        check_for_timing_mode(stem)
        img_filename = stem + '_IMG.fits' # The output filename
        img_filepath = observation.path_processed / img_filename # The output filepath
        create_image_file(infile=raw_filepath, outfile=img_filepath,
                          ximagebinsize=ximagebinsize, yimagebinsize=yimagebinsize, clobber=clobber)


def espfilt(eventfile):
    """
    https://xmm-tools.cosmos.esa.int/external/sas/current/doc/espfilt/espfilt.html
    https://xmm-tools.cosmos.esa.int/external/sas/current/doc/espfilt/node9.html
    """
    cmd = f'espfilt eventfile={eventfile}'
    run_cmd(cmd)


if __name__ == "__main__":
    from exod.utils.path import data
    obsids = read_observation_ids(data / 'observations.txt')
    for obsid in obsids:
        filter_obsid_events(observation=obsid)
        create_obsid_images(observation=obsid)

  
