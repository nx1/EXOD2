from exod.utils.logger import logger
from exod.utils.path import data_processed
from exod.pre_processing.epic_submodes import PN_SUBMODES, MOS_SUBMODES

import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_dd
from astropy.io import fits
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
                             f'I still havent figured out how to seperate these out')
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

    logger.info('Trimming TIME on data cube based on all data')
    time_min, time_max = get_inner_time_bounds(data_to_stack)

    data_EPIC = data_EPIC[(data_EPIC['TIME'] >= time_min) & (data_EPIC['TIME'] <= time_max)]
    logger.info(f'data_EPIC:\n{data_EPIC}')
    return data_EPIC, time_max, time_min


def get_HE_lc(data_EPIC):
    logger.info('Creating High Energy Lightcurve')
    min_energy_HE = 10.0  # minimum extraction energy for (High Energy) Background events
    max_energy_HE = 12.0  # maximum extraction energy for (High Energy) Background events
    gti_window_size = 100  # Window Size to use for GTI extraction
    logger.info(f'min_energy_HE = {min_energy_HE} max_energy_HE = {max_energy_HE} gti_window_size = {gti_window_size}')
    time_min = np.min(data_EPIC['TIME'])
    time_max = np.max(data_EPIC['TIME'])
    logger.info(f'time_min = {time_min} time_max = {time_max}')

    time_windows_gti = np.arange(time_min, time_max, gti_window_size)
    data_HE = np.array(data_EPIC['TIME'][(data_EPIC['PI'] > min_energy_HE * 1000) & (data_EPIC['PI'] < max_energy_HE * 1000)])
    lc_HE = np.histogram(data_HE, bins=time_windows_gti)[0] / gti_window_size  # Divide by the bin size to get in ct/s
    return time_windows_gti, lc_HE


def get_bti(time, data, threshold):
    """
    Get the bad time intervals for a given lightcurve.

    Parameters
    ----------
    time : Time Array
    data : Data Array
    threshold: Threshold for bad time intervals

    Returns
    -------
    bti : [{'START':500, 'STOP':600}, {'START':700, 'STOP':800}, ...]
    """

    mask = data > threshold  # you can flip this to get the gti instead (it works)
    if mask.all():
        logger.info('All Values above Treshold! Entire observation is bad :(')
        return []
    elif (~mask).all():
        logger.info('All Values below Threshold! Entire observation is good :)')
        return []

    int_mask = mask.astype(int)
    diff = np.diff(int_mask)
    idx_starts = np.where(diff == 1)[0]
    idx_ends = np.where(diff == -1)[0]
    time_starts = time[idx_starts]
    time_ends = time[idx_ends]

    first_crossing = diff[diff != 0][0]
    last_crossing = diff[diff != 0][-1]

    if (first_crossing, last_crossing) == (-1, 1):
        logger.info('Curve Started and Ended above threshold!')
        time_starts = np.append(time[0], time_starts)
        time_ends = np.append(time_ends, time[-1])

    elif len(idx_starts) < len(idx_ends):
        logger.info('Curve Started Above threshold! (but did not end above it)')
        time_starts = np.append(time[0], time_starts)

    elif len(idx_starts) > len(idx_ends):
        logger.info('Curve Ended Above threshold! (but did not start above it)')
        time_ends = np.append(time_ends, time[-1])

    else:
        logger.info('Curve started and ended below threshold, nothing to do.')

    assert len(time_starts) == len(time_ends)
    bti = [{'START': time_starts[i], 'STOP': time_ends[i]} for i in range(len(time_ends))]
    return bti


def get_rejected_idx(bti, time_windows):
    """
    Get the rejected indexs for an array of time windows
    given a list of bad time intervals.

    Parameters
    ----------
    bti : [{'START':300, 'STOP':500}, {'START':600, 'STOP':800}, ...]
    time_windows : array

    Returns
    -------
    rejected_idx : array of rejected indexs
    """
    t_starts = [b['START'] for b in bti]
    t_stops = [b['STOP'] for b in bti]
    idx_starts = np.searchsorted(time_windows, t_starts)
    idx_stops = np.searchsorted(time_windows, t_stops)

    rejected_idx = np.array([])
    for i in range(len(idx_starts)):
        idxs = np.arange(idx_starts[i], idx_stops[i], 1)
        rejected_idx = np.append(rejected_idx, idxs)
    rejected_idx = rejected_idx.astype(int)
    return rejected_idx


def plot_bti(time, data, threshold, bti):
    plt.figure(figsize=(10, 2.5))
    for b in bti:
        plt.axvspan(xmin=b['START'], xmax=b['STOP'], color='red', alpha=0.5)

    plt.scatter(time, data, label='Data')
    plt.axhline(threshold, color='red', label=f'Threshold={threshold}')
    plt.xlabel('Time')
    plt.ylabel(r'Window Count Rate $\mathrm{ct\ s^{{-1}}}$')
    plt.legend()
    #plt.show()


def read_EPIC_events_file(obsid, size_arcsec, time_interval, gti_only=False, min_energy=0.2, max_energy=12.0):
    """
    Read the EPIC event files and create the data cube, and the X,Y Coordinates the axial extents of the cube.

    Parameters
    ----------
    obsid : str : Observation ID
    size_arcsec : float : Size in arcseconds of the final spatial grid on which the data is binned
    time_interval : float : temporal window size of data cube binning
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
        time_window_gti, lc_HE = get_HE_lc(data_EPIC=data_EPIC)
        bti = get_bti(time=time_window_gti, data=lc_HE, threshold=gti_threshold)
        plot_bti(time=time_window_gti[:-1], data=lc_HE, threshold=gti_threshold, bti=bti)
        rejected_frame_idx = get_rejected_idx(bti=bti, time_windows=time_windows)


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

    cropping_angles = (np.min(indices_image[0]), np.max(indices_image[0]) + 1,
                       np.min(indices_image[1]), np.max(indices_image[1]) + 1)
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

if __name__ == "__main__":
    from exod.pre_processing.download_observations import read_observation_ids
    from exod.utils.path import data
    import pandas as pd

    obsids = read_observation_ids(data / 'observations.txt')

    all_res = []
    for obs in obsids:
        res = {}
        res['obsid'] = obs
        try:
            cube, coordinates_XY = read_EPIC_events_file(obsid=obs, size_arcsec=20, time_interval=750, gti_only=True)
            res['status'] = 'OK'
        except Exception as e:
            res['status'] = str(e)

        all_res.append(res)


    df = pd.DataFrame(all_res)
    logger.info(f'\n{df}')
