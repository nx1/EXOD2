from exod.utils.logger import logger
from exod.utils.path import data_processed, data_results
from exod.pre_processing.epic_submodes import PN_SUBMODES, MOS_SUBMODES

import os
import numpy as np
from scipy.stats import binned_statistic_dd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table, vstack
from itertools import combinations
from scipy.cluster.hierarchy import DisjointSet

def check_eventlist_instrument_and_submode(evt_file):
    """
    Check to see if the event list is supported by EXOD.
    e.g reject timing mode observations.

    Parameters
    ----------
    evt_file : Path to event file

    Returns
    -------
    header : fits header of hdu=1
    """
    logger.info(f'Getting fits header for {evt_file}')
    header = fits.getheader(evt_file, hdu=1)
    instrument = header['INSTRUME']
    submode = header['SUBMODE']
    logger.info(f'INSTRUME: {instrument} SUBMODE: {submode}')
    if instrument == 'EPN':
        if PN_SUBMODES[submode] == False:
            raise NotImplementedError(f'submode: {submode} is not supported')
    elif instrument in ['EMOS1', 'EMOS2']:
        if MOS_SUBMODES[submode] == False:
            raise NotImplementedError(f'submode: {submode} is not supported')
    else:
        raise KeyError(f'instrument: {instrument} is not recognized')
    return header


def PN_remove_bad_rows(data_pn):
    logger.info('Removing Bad PN Rows Struder et al. 2001b')
    data_pn = data_pn[~((data_pn['CCDNR']==4)&(data_pn['RAWX']==12))&
                      ~((data_pn['CCDNR']==5)&(data_pn['RAWX']==11))&
                      ~((data_pn['CCDNR']==10)&(data_pn['RAWX']==28))]
    return data_pn


def PN_remove_borders(data_pn):
    """
    Remove Edge rows that are contaminated.

    #TODO remove edges for the smaller PN frame mode!
    """
    logger.warning('Ejecting PN Borders **MAY HAVE TO BE ADAPTED FOR OBSERVING MODES**')
    data_pn = data_pn[~(data_pn['RAWX']==0)&~(data_pn['RAWX']==64)&
                      ~(data_pn['RAWY']==0)&~(data_pn['RAWY']==200)]
    return data_pn


def read_pn_events_list(evt_file, hdu=1, remove_bad_rows=True, remove_borders=True):
    header = check_eventlist_instrument_and_submode(evt_file=evt_file)
    hdul   = fits.open(evt_file)
    header = hdul[hdu].header 
    data   = Table(hdul[hdu].data)
    if remove_bad_rows:
        data = PN_remove_bad_rows(data)
    if remove_borders:
        data = PN_remove_borders(data)
    return data, header


def read_mos_events_list(evt_file, hdu=1):
    header = check_eventlist_instrument_and_submode(evt_file=evt_file)
    header = fits.getheader(evt_file, hdu=hdu)
    tab = Table.read(evt_file, hdu=hdu)
    return tab, header



def get_inner_time_bounds(data_list):
    """Get the latest start time and the earliest end time across all detectors."""
    start_times = [np.min(data['TIME']) for data in data_list]
    end_times = [np.max(data['TIME']) for data in data_list]

    latest_start_time = max(start_times)
    earliest_end_time = min(end_times)

    return latest_start_time, earliest_end_time

def _get_inner_time_bounds(data_M1, data_M2, data_pn):
    """Get the latest start time and the earliest end time across all detectors."""
    start_M1, end_M1 = np.min(data_M1['TIME']), np.max(data_M1['TIME'])
    start_M2, end_M2 = np.min(data_M2['TIME']), np.max(data_M2['TIME'])
    start_pn, end_pn = np.min(data_pn['TIME']), np.max(data_pn['TIME'])
    latest_start_time = max(start_M1, start_M2, start_pn)
    earliest_end_time = min(end_M1, end_M2, end_pn)
    return latest_start_time, earliest_end_time


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


def histogram_events_list(tab_evt, bin_size_seconds=100):
    """
    Create a histogram from a list of events.

    Parameters
    ----------
    tab_evt : Event list table with ['TIME'] column
    bin_size_seconds

    Returns
    -------
    hist, bin_edges
    """
    time_array = tab_evt['TIME']
    num_bins = int((np.max(time_array) - np.min(time_array)) / bin_size_seconds)
    bins = np.linspace(np.min(time_array), np.max(time_array), num_bins + 1)
    hist, bin_edges = np.histogram(time_array, bins=bins) 
    return hist, bin_edges

def read_EPIC_events_file(obsid, size_arcsec, time_interval, box_size=3, gti_only=False, min_energy=0.2, max_energy=12.0):
    """
    Read the EPIC event files and create the data cube, and the X,Y Coordinates the axial extents of the cube.

    TODO This function is doing way too much imho, I think we should be able to use the algorithm on data that is
    TODO not pre-binned spatially aswell as for individual cameras, I think we may have to think about this more.
    TODO I am halfway through re-doing this I think (see above functions)

    Parameters
    ----------
    obsid : str : Observation ID
    size_arcsec : float : Size in arcseconds of the final spatial grid on which the data is binned
    time_interval : float : temporal window size of data cube binning
    box_size : This is used to calculate the `cropping angles' which is basically the extents of the image
    gti_only : bool : If true use only the data found in GTIs (as specified >1.5 CR)
    min_energy : Minimum Energy for final EPIC data cube (this is already done at the filtering step no?)
    max_energy : see above.

    Returns
    -------
    cube_EPIC : np.ndarray containing the binned eventlist data (x,y,t)
    coordinates_XY : (X, Y) where X and Y are 1D-arrays describing the extents of the cube.
    """
    # Extraction Settings
    gti_threshold = 1.5                    # Values above this will be considered BTIs.
    pixel_size    = size_arcsec / 0.05     # Final Pixel size in DetX DetY values
    extent        = 70000                  # Temporary extent of the cube in DetX DetY values
    nb_pixels     = int(extent/pixel_size) #



    data_EPIC, time_max, time_min = get_epic_data(obsid=obsid)

    n_bins             = int(((time_max - time_min) / time_interval))
    time_stop          = time_min + n_bins * time_interval
    time_windows       = np.arange(time_min, time_stop + 1, time_interval)
    if gti_only:
        rejected_frame_idx = calc_rejected_frame_idx(data_EPIC, gti_threshold, time_interval, time_max, time_min)

    logger.info(f'Filtering Grouped Events list by energy min_energy={min_energy} max_energy={max_energy}')
    data_EPIC = data_EPIC[(min_energy * 1000 < data_EPIC['PI']) & (data_EPIC['PI'] < max_energy * 1000)]

    logger.info(f'Creating the data cube')
    bin_x = np.linspace(0, extent, nb_pixels+1)
    bin_y = np.linspace(0, extent, nb_pixels+1)
    bin_t = time_windows

    cube_EPIC = binned_statistic_dd((data_EPIC['X'], data_EPIC['Y'], data_EPIC['TIME']),
                                    values=None,
                                    statistic='count',
                                    bins=[bin_x, bin_y, bin_t])[0]

    logger.info(f'Getting empty indexes in data cube')
    indices_image = np.where(np.sum(cube_EPIC, axis=2) > 0)

    cropping_angles = (np.min(indices_image[0]) - box_size, np.max(indices_image[0]) + 1 + box_size,
                       np.min(indices_image[1]) - box_size, np.max(indices_image[1]) + 1 + box_size)
    logger.info(f'cropping_angles: {cropping_angles}')

    logger.info('Cropping EPIC cube between cropping_angles')
    cube_EPIC = cube_EPIC[cropping_angles[0]:cropping_angles[1],cropping_angles[2]:cropping_angles[3]]

    logger.info('Getting XY Coordinates')
    coordinates_XY = (np.linspace(0,extent, nb_pixels+1)[cropping_angles[0]:cropping_angles[1]],
                      np.linspace(0,extent, nb_pixels+1)[cropping_angles[2]:cropping_angles[3]])

    #Drop BTI if necessary
    if gti_only:
        logger.info('gti_only=True, dropping bad frames from Data Cube')
        dims_img  = (cube_EPIC.shape[0], cube_EPIC.shape[1],1)
        nan_image = np.full(shape=dims_img, fill_value=np.nan, dtype=np.float64)
        cube_EPIC[:,:,rejected_frame_idx] = nan_image

    return cube_EPIC, coordinates_XY


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
    for some observations it will return multiple entires,

    There is a slight issue currently however and that is if more than
    3 event lists overlap with each other it does not correct pull out
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

    subsets = disjoint_set.subsets()
    logger.info(f'Found {len(subsets)} subsets.')

    for s in subsets:
        if len(s) > 3:
            raise ValueError(f'Overlapping subset has {len(s)} event files! (>3) '
                             f'I still havent figured out how to seperate these out')
    return subsets

def get_epic_data(obsid):
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
    return data_EPIC, time_max, time_min


def calc_rejected_frame_idx(data_EPIC, gti_threshold, time_interval, time_max, time_min):
    min_energy_HE = 10.0  # minimum extraction energy for (High Energy) Background events
    max_energy_HE = 12.0  # maximum extraction energy for (High Energy) Background events
    gti_window_size = 100 # Window Size to use for GTI extraction

    logger.info('=======================================')
    logger.info('Calculating Rejected Frame in data cube')
    logger.info('=======================================')
    logger.info('Calculating Time windows')
    n_bins = int(((time_max - time_min) / time_interval))
    time_stop = time_min + n_bins * time_interval
    time_windows = np.arange(time_min, time_stop + 1, time_interval)
    time_windows_gti = np.arange(time_min, time_stop + 1, gti_window_size)

    logger.info(f'n_bins     = {n_bins}')
    logger.info(f'time_min   = {time_min}')
    logger.info(f'time_stop  = {time_stop}')
    logger.info(f'time_max   = {time_max}')

    logger.info(f'Extracting High Energy Lightcurve min_energy_HE={min_energy_HE} max_energy_HE={max_energy_HE}')
    data_HE    = np.array(data_EPIC['TIME'][(data_EPIC['PI'] > min_energy_HE * 1000) & (data_EPIC['PI'] < max_energy_HE * 1000)])
    lc_HE      = np.histogram(data_HE, bins=time_windows_gti)[0] / gti_window_size  # Divide by the bin size to get in ct/s
    lc_HE_good = np.where(lc_HE < gti_threshold, lc_HE, np.nan)  # These can be commented out once we know things work
    lc_HE_bad  = np.where(lc_HE >= gti_threshold, lc_HE, np.nan)  # These can be commented out once we know things work

    logger.info('Getting BTI start and stop indexs')
    gti_mask      = (lc_HE < gti_threshold).astype(int)
    bti_start_idx = np.where(np.diff(gti_mask) == -1)[0]
    bti_stop_idx  = np.where(np.diff(gti_mask) == 1)[0]
    logger.info(f'bti_start_idx = {bti_start_idx}')
    logger.info(f'bti_stop_idx  = {bti_stop_idx}')

    logger.info('Getting Associated Timebins for the Good time intervals')
    indices_timebinsleft_gtistart = np.searchsorted(time_windows, time_windows_gti[bti_start_idx])
    indices_timebinsleft_gtistop = np.searchsorted(time_windows, time_windows_gti[bti_stop_idx])
    logger.info(f'indices_timebinsleft_gtistart = {indices_timebinsleft_gtistart}')
    logger.info(f'indices_timebinsleft_gtistop  = {indices_timebinsleft_gtistop}')

    logger.info('Calculating rejected_idx BTI Indexs')
    rejected_idx = []
    for start, end in zip(indices_timebinsleft_gtistart, indices_timebinsleft_gtistop):
        rejected_idx.extend(range(start, end + 1))

    # Handle remaining BTIs at the end
    if len(indices_timebinsleft_gtistop) < len(indices_timebinsleft_gtistart):
        last_start = indices_timebinsleft_gtistart[-1] + 1
        remaining_indices = list(range(last_start, len(time_windows)))
        rejected_idx.extend(remaining_indices)

    # Remove duplicates and subtract 1 from each index
    rejected_idx = list(set(rejected_idx))
    rejected_idx = [index - 1 for index in rejected_idx]
    logger.info(f'rejected_idx = {rejected_idx}')
    return rejected_idx


if __name__ == "__main__":
    from exod.pre_processing.download_observations import read_observation_ids
    from exod.utils.path import data
    import pandas as pd

    obsids = read_observation_ids(data / 'observations.txt')
 
    for obs in obsids:
        try:
            cube, coordinates_XY = read_EPIC_events_file(obsid=obs,
                                                         size_arcsec=20,
                                                         time_interval=750,
                                                         gti_only=True)
        except Exception as e:
            logger.warning(f'Could not read {obs} {e}')
