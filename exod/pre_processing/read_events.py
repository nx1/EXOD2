from exod.pre_processing.bti import get_high_energy_lc, get_bti, get_rejected_idx, plot_bti
from exod.utils.logger import logger
from exod.utils.path import data_processed, data_results
from exod.xmm.epic_submodes import PN_SUBMODES, MOS_SUBMODES

import warnings
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_dd
from astropy.table import Table, vstack
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


def get_image_files(obsid):
    """Return all the image files found for a given observation ID"""
    logger.info(f'Getting Image files for observation: {obsid}')
    data_processed_obs = data_processed / f'{obsid}'
    img_files = data_processed_obs.glob('*IMG.fits')
    img_files = list(img_files)
    if img_files:
        logger.info(f'Found {len(img_files)} image files')
        return img_files
    else:
        raise FileNotFoundError(f'No Images found for observation: {obsid}')


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




def PN_remove_borders(data_pn):
    """
    Remove Edge rows that are contaminated.

    #TODO remove edges for the smaller PN frame mode!
    """
    logger.warning('Ejecting PN Borders **MAY HAVE TO BE ADAPTED FOR OBSERVING MODES**')
    rawx_exclude = [0,1,3,4,61,61,63,64]
    rawy_exclude = [0,1,2,3,4,197,198,199,200]
    logger.info(f'length pre: {len(data_pn)}')
    for rawx in rawx_exclude:
        data_pn = data_pn[data_pn['RAWX'] != rawx]
    for rawy in rawy_exclude:
        data_pn = data_pn[data_pn['RAWY'] != rawy]

    logger.info(f'length post: {len(data_pn)}')
    return data_pn




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


def get_pn_data(obsid):
    """What the hell am I doing ffs."""
    files = get_filtered_events_files(obsid)
    for f in files:
        if 'PI' in f.stem:
            data, header = read_pn_events_list(f)
            return data
    raise KeyError(f'No PN data found for {obsid}')

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

    logger.info('Trimming TIME on data cube based on all data')
    time_min, time_max = get_inner_time_bounds(data_to_stack)

    data_EPIC = data_EPIC[(data_EPIC['TIME'] >= time_min) & (data_EPIC['TIME'] <= time_max)]
    logger.info(f'data_EPIC:\n{data_EPIC}')
    return data_EPIC, time_min, time_max


def crop_data_cube(cube_EPIC, extent, nb_pixels):
    """Crop the surrounding areas of the datacube that are empty."""
    logger.info('Cropping Data Cube...')
    logger.info(f'Getting non-empty regions in data cube')
    idx_nonempty = np.where(np.sum(cube_EPIC, axis=2) > 0)

    bbox_img = (np.min(idx_nonempty[0]), np.max(idx_nonempty[0]) + 1,
                np.min(idx_nonempty[1]), np.max(idx_nonempty[1]) + 1)

    logger.info(f'Cropping EPIC cube between bbox_img: {bbox_img}')
    cube_EPIC = cube_EPIC[bbox_img[0]:bbox_img[1], bbox_img[2]:bbox_img[3]]

    logger.info('Getting XY Coordinates')
    coordinates_XY = (np.linspace(0, extent, nb_pixels + 1)[bbox_img[0]:bbox_img[1]],
                      np.linspace(0, extent, nb_pixels + 1)[bbox_img[2]:bbox_img[3]])
    return cube_EPIC, coordinates_XY
