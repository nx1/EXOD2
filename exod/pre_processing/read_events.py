from exod.utils.logger import logger
from exod.utils.path import data_processed

import warnings
import numpy as np
from astropy.table import vstack
from itertools import combinations
from scipy.cluster.hierarchy import DisjointSet
from astropy.io import fits
from astropy.wcs import FITSFixedWarning
from astropy.units import UnitsWarning

warnings.filterwarnings(action='ignore', category=FITSFixedWarning, append=True)
warnings.filterwarnings(action='ignore', category=UnitsWarning, append=True)


def get_filtered_events_files(obsid):
    """Return all the filtered event files found for a given observation ID."""
    logger.info(f'Getting Image files for observation: {obsid}')
    data_processed_obs = data_processed / f'{obsid}'
    evt_files = data_processed_obs.glob('*FILT.fits')
    evt_files = list(evt_files)
    if evt_files:
        logger.info(f'Found {len(evt_files)} filtered event files')
        return evt_files
    else:
        raise FileNotFoundError(f'No Event files found for observation: {obsid}')



def get_PN_image_file(obsid):
    """Return the first PN image for a given observation ID"""
    logger.info(f'Getting Image files for observation: {obsid}')
    data_processed_obs = data_processed / f'{obsid}'
    img_files = data_processed_obs.glob('*PI*IMG.fits')
    img_files = list(img_files)
    if img_files:
        return img_files[0]
    else:
        raise FileNotFoundError(f'No PN Images found for observation: {obsid}')

def get_inner_time_bounds(data_list):
    """Get the latest start time and the earliest end time across all detectors."""
    start_times = [np.min(data['TIME']) for data in data_list]
    end_times   = [np.max(data['TIME']) for data in data_list]

    latest_start_time = max(start_times)
    earliest_end_time = min(end_times)

    return latest_start_time, earliest_end_time


def get_start_end_time(event_file):
    hdul = fits.open(event_file)
    h = hdul[1].header
    TSTART, TSTOP = h['TSTART'], h['TSTOP']
    return [TSTART, TSTOP]


def get_overlapping_eventlist_subsets(obsid):
    """
    Return the overlapping eventlists for a given observation.

    In most cases this will just return a list of length 1 like
    [{M1.fits, M2.fits, PN.fits'}] with the event files. However,
    for some observations it will return multiple entires if there
    have been multiple seperate observations.

    There is a slight issue currently however and that is if more than
    3 event lists overlap with each other it does not correctly pull out
    the combination. I have made this case error if it happens for now...
    """
    logger.info(f'Getting overlapping eventlists for observation={obsid}')
    def intervals_overlap(I1, I2):
        return I1[0] <= I2[1] and I1[1] >= I2[0]

    files = get_filtered_events_files(obsid)
    disjoint_set = DisjointSet(files)
    file_intervals = {f: get_start_end_time(f) for f in files}

    for d1, d2 in combinations(file_intervals.items(), r=2):
        f1, I1 = d1[0], d1[1]
        f2, I2 = d2[0], d2[1]
            
        if intervals_overlap(I1, I2):
            disjoint_set.merge(f1, f2)
            logger.info(f'f1: {f1.stem} : {I1} OVERLAP!')
            logger.info(f'f2: {f2.stem} : {I2} OVERLAP!')
        else:
            logger.info(f'f1: {f1.stem} : {I1}')
            logger.info(f'f2: {f2.stem} : {I2}')

    subsets = disjoint_set.subsets()
    logger.info(f'Found {len(subsets)} subsets.')

    for s in subsets:
        if len(s) > 3:
            raise ValueError(f'Overlapping subset has {len(s)} event files! (>3) '
                             f'I still havent figured out how to separate these out')
    return subsets

def get_epic_data(obsid):
    """Get the merged EPIC data for a given observation."""
    event_list_subsets = get_overlapping_eventlist_subsets(obsid=obsid)

    data_to_stack = []
    for subset in event_list_subsets:
        for path in subset:
            if 'PN' in path.stem:
                tab, header = read_pn_events_list(path)
                data_to_stack.append(tab)
            if 'M1' or 'M2' in path.stem:
                tab, header = read_mos_events_list(path)
                data_to_stack.append(tab)

    logger.info('Combining event lists')
    data_EPIC = vstack(data_to_stack, metadata_conflicts='silent')

    logger.info('Trimming TIME on data cube_n based on all data')
    time_min, time_max = get_inner_time_bounds(data_to_stack)

    data_EPIC = data_EPIC[(data_EPIC['TIME'] >= time_min) & (data_EPIC['TIME'] <= time_max)]
    logger.info(f'data_EPIC:\n{data_EPIC}')
    return data_EPIC, time_min, time_max


