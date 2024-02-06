import numpy as np
from matplotlib import pyplot as plt

from exod.utils.logger import logger
from exod.utils.path import data_results


def get_high_energy_lc(data_EPIC):
    logger.info('Creating High Energy Lightcurve')
    min_energy_high_energy = 10.0  # minimum extraction energy for High Energy Background events
    max_energy_high_energy = 12.0  # maximum extraction energy for High Energy Background events
    gti_window_size = 100 # Window Size to use for GTI extraction
    logger.info(f'min_energy_high_energy = {min_energy_high_energy} max_energy_high_energy = {max_energy_high_energy} gti_window_size = {gti_window_size}')
    time_min = np.min(data_EPIC['TIME'])
    time_max = np.max(data_EPIC['TIME'])
    logger.info(f'time_min = {time_min} time_max = {time_max}')

    time_windows_gti = np.arange(time_min, time_max, gti_window_size)
    data_high_energy = np.array(data_EPIC['TIME'][(data_EPIC['PI'] > min_energy_high_energy * 1000) & (data_EPIC['PI'] < max_energy_high_energy * 1000)])
    lc_high_energy = np.histogram(data_high_energy, bins=time_windows_gti)[0] / gti_window_size  # Divide by the bin size to get in ct/s
    return time_windows_gti, lc_high_energy


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
        logger.info('All values above threshold! Entire observation is bad :(')
        raise ValueError(f'Entire Observation is a BTI')
    elif (~mask).all():
        logger.info('All values below Threshold! Entire observation is good :)')
        return []

    int_mask    = mask.astype(int)
    diff        = np.diff(int_mask)
    idx_starts  = np.where(diff == 1)[0]
    idx_ends    = np.where(diff == -1)[0]
    time_starts = time[idx_starts]
    time_ends   = time[idx_ends]

    first_crossing = diff[diff != 0][0]
    last_crossing  = diff[diff != 0][-1]

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
    idx_stops = np.searchsorted(time_windows, t_stops) - 1

    rejected_idx = np.array([])
    for i in range(len(idx_starts)):
        idxs = np.arange(idx_starts[i], idx_stops[i], 1)
        rejected_idx = np.append(rejected_idx, idxs)
    rejected_idx = rejected_idx.astype(int)
    return rejected_idx

def get_rejected_idx_bool(rejected_idx, time_windows):
    """
    Get the boolean array corresponding to if a time
    window was a bad time interval or not.

    Parameters
    ----------
    rejected_idx : [1,4,6]
    time_windows : [0, 1.5, 2.0, 3.5, 5.0, 6.5, 8.0]

    Returns
    -------
    rejected_frame_bool : [F,T,F,F,T,F,T,F]

    """
    arr = np.arange(len(time_windows))
    rejected_frame_bool = np.isin(arr, rejected_idx)
    return rejected_frame_bool

def plot_bti(time, data, threshold, bti, obsid):
    plt.figure(figsize=(10, 2.5))
    for b in bti:
        plt.axvspan(xmin=b['START'], xmax=b['STOP'], color='red', alpha=0.5)

    plt.scatter(time, data, label='Data', marker='.', s=5, color='black')
    plt.axhline(threshold, color='red', label=f'Threshold={threshold}')
    plt.xlabel('Time')
    plt.ylabel(r'Window Count Rate $\mathrm{ct\ s^{{-1}}}$')
    plt.legend()

    savepath = data_results / obsid / 'bti_plot.png'
    logger.info(f'saving bti plot to: {savepath}')
    plt.savefig(savepath)

    # plt.show()
