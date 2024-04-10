import numpy as np
from matplotlib import pyplot as plt

from exod.utils.logger import logger
from exod.utils.path import data_results


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
        return [{'START': time[0], 'STOP': time[-1]}]
        # raise ValueError(f'Entire Observation is a BTI')
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


def plot_bti(time, data, threshold, bti, savepath=None):
    plt.figure(figsize=(10, 2.5))
    for b in bti:
        plt.axvspan(xmin=b['START'], xmax=b['STOP'], color='red', alpha=0.5)

    plt.scatter(time, data, label='Data', marker='.', s=5, color='black')
    plt.axhline(threshold, color='red', label=f'Threshold={threshold}')

    plt.title('BTI Diagnostic Plot')
    plt.xlabel('Time')
    plt.ylabel(r'Window Count Rate $\mathrm{ct\ s^{{-1}}}$')
    plt.legend()


    if savepath:
        logger.info(f'saving bti plot to: {savepath}')
        plt.savefig(savepath)

    # plt.show()


def get_bti_bin_idx(bti, bin_t):
    """
    Get the rejected indexs for an array of time windows
    given a list of bad time intervals.

    Parameters
    ----------
    bti : [{'START':300, 'STOP':500}, {'START':600, 'STOP':800}, ...]
    bin_t : array

    Returns
    -------
    bti_bin_idx : array of indexs coreesponding to BTIs
    """
    t_starts = [b['START'] for b in bti]
    t_stops = [b['STOP'] for b in bti]
    idx_starts = np.searchsorted(bin_t, t_starts)
    idx_stops = np.searchsorted(bin_t, t_stops) - 1

    bti_bin_idx = np.array([])
    for i in range(len(idx_starts)):
        idxs = np.arange(idx_starts[i], idx_stops[i], 1)
        bti_bin_idx = np.append(bti_bin_idx, idxs)
    bti_bin_idx = bti_bin_idx.astype(int)
    logger.info(f'{len(bti_bin_idx)} bins in BTI out of {len(bin_t)}')
    return bti_bin_idx


def get_bti_bin_idx_bool(rejected_idx, bin_t):
    """
    Get the boolean array corresponding to if a time
    window was a bad time interval or not.

    Parameters
    ----------
    rejected_idx : [1,4,6]
    bin_t : [0, 1.5, 2.0, 3.5, 5.0, 6.5, 8.0]

    Returns
    -------
    bti_bin_idx_bool : [F,T,F,F,T,F,T,F]

    """
    arr = np.arange(len(bin_t))
    bti_bin_idx_bool = np.isin(arr, rejected_idx)
    return bti_bin_idx_bool
