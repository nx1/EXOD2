import os
import numpy as np
from scipy.stats import binned_statistic_dd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table, vstack

from exod.utils.logger import logger
from exod.utils.path import data_processed,data_results
from exod.pre_processing.epic_submodes import PN_SUBMODES, MOS_SUBMODES

def check_eventlist_instrument_and_submode(evt_file):
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
    logger.warning('Ejecting PN Borders **MAY HAVE TO BE ADAPTED FOR OBSERVING MODES**')
    data_pn = data_pn[~(data_pn['RAWX']==0)&~(data_pn['RAWX']==64)&
                      ~(data_pn['RAWY']==0)&~(data_pn['RAWY']==200)]
    return data_pn

def read_pn_events_list(evt_file, hdu=1, remove_bad_rows=True, remove_borders=True):
    header = check_eventlist_instrument_and_submode(evt_file=evt_file)
    header = fits.getheader(evt_file, hdu=hdu)
    data   = Table(fits.open(evt_file)[hdu].data)
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


def get_inner_time_bounds(data_M1, data_M2, data_pn):
    """Get the latest start time and the earliest end time across all detectors."""
    start_M1, end_M1 = np.min(data_M1['TIME']), np.max(data_M1['TIME'])
    start_M2, end_M2 = np.min(data_M2['TIME']), np.max(data_M2['TIME'])
    start_pn, end_pn = np.min(data_pn['TIME']), np.max(data_pn['TIME'])
    latest_start_time = max(start_M1,start_M2,start_pn)
    earliest_end_time = min(end_M1, end_M2, end_pn)
    return latest_start_time, earliest_end_time

def read_EPIC_events_file(obsid, size_arcsec, time_interval, box_size=3, gti_only=False, min_energy=0.2, max_energy=12.0):
    """
    Reads the EPIC events files. Returns the cube and the coordinates_XY (used for WCS conversion)
    :argument obsid of the target observation
    :argument size_arcsec is the size in arseconds of the final spatial grid onto which data is binned,
    :argument time_interval is the same but for temporal dimension
    """
    # Extraction Settings
    threshold_GTI = 1.5                    # Values above this will be considered BTIs.
    pixel_size    = size_arcsec / 0.05     # Final Pixel size in DetX DetY values
    extent        = 70000                  # Temporary extent of the cube in DetX DetY values
    nb_pixels     = int(extent/pixel_size) #
    min_energy_HE = 10.0                   # minimum extraction energy for (High Energy) Background events
    max_energy_HE = 12.0                   # maximum extraction energy for (High Energy) Background events

    # Set up Directories
    path_processed_obs = data_processed / f'{obsid}'
    path_results_obs = data_results / f'{obsid}'
    path_time_interval = path_results_obs / f'{time_interval}s'
    os.makedirs(path_results_obs, exist_ok=True)
    os.makedirs(path_time_interval, exist_ok=True)

    # Get Cleaned Event Files
    pn_file = path_processed_obs / 'PN_pattern_clean.fits'
    m1_file = path_processed_obs / 'M1_pattern_clean.fits'
    m2_file = path_processed_obs / 'M2_pattern_clean.fits'

    logger.info(f'Reading PN events file {pn_file}')
    data_pn = Table(fits.open(pn_file)[1].data)['X','Y','TIME', 'RAWX','RAWY','CCDNR','PI']
    data_pn = PN_remove_bad_rows(data_pn)
    data_pn = PN_remove_borders(data_pn)

    logger.info(f'Reading M1 events file {m1_file}')
    data_M1 = Table(fits.open(m1_file)[1].data)['X','Y','TIME','PI']

    logger.info(f'Reading M2 events file {m2_file}')
    data_M2 = Table(fits.open(m2_file)[1].data)['X','Y','TIME','PI']

    logger.info('Combining event lists')
    data_EPIC = vstack([data_pn, data_M1, data_M2])

    logger.info('Trimming TIME on data cube based on all data')
    time_min, time_max = get_inner_time_bounds(data_M1=data_M1, data_M2=data_M2, data_pn=data_pn)
    data_EPIC = data_EPIC[(data_EPIC['TIME'] >= time_min) &
                          (data_EPIC['TIME'] <= time_max)]

    logger.info('Calculating Time windows')
    n_bins           = int(((time_max - time_min) / time_interval))
    time_stop        = time_min + n_bins * time_interval
    time_windows     = np.arange(time_min, time_stop+1, time_interval)
    time_windows_gti = np.arange(time_min, time_stop+1, 100)

    logger.info(f'n_bins     = {n_bins}')
    logger.info(f'time_min   = {time_min}')
    logger.info(f'time_stop  = {time_stop}')
    logger.info(f'time_max   = {time_max}')

    logger.info(f'Extracting High Energy Lightcurve min_energy_HE={min_energy_HE} max_energy_HE={max_energy_HE}')
    data_HE = np.array(data_EPIC['TIME'][(data_EPIC['PI']>min_energy_HE*1000) & (data_EPIC['PI']<max_energy_HE*1000)])
    lc_HE   = np.histogram(data_HE, bins=time_windows_gti)[0]/100
    lc_HE_good = np.where(lc_HE<threshold_GTI, lc_HE, np.nan)  # These can be commented out once we know things work
    lc_HE_bad  = np.where(lc_HE>=threshold_GTI, lc_HE, np.nan) # These can be commented out once we know things work

    logger.info('Calculating GTI Table')
    gti_tab = (lc_HE<threshold_GTI).astype(int)

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
    #cube_EPIC[np.where(np.sum(cube_EPIC, axis=2) < 1)] = np.full(len(time_windows) - 1, np.nan)

    logger.info('Getting BTI start and stop indexs')
    bti_start_idx = np.where(np.diff(gti_tab) == -1)[0]
    bti_stop_idx  = np.where(np.diff(gti_tab) == 1)[0]
    logger.info(f'bti_start_idx = {bti_start_idx}')
    logger.info(f'bti_stop_idx  = {bti_stop_idx}')

    logger.info('Getting Associated Timebins for the Good time intervals')
    indices_timebinsleft_gtistart = np.searchsorted(time_windows, time_windows_gti[bti_start_idx])
    indices_timebinsleft_gtistop  = np.searchsorted(time_windows, time_windows_gti[bti_stop_idx])
    logger.info(f'indices_timebinsleft_gtistart = {indices_timebinsleft_gtistart}')
    logger.info(f'indices_timebinsleft_gtistop = {indices_timebinsleft_gtistop}')

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
    logger.info(f'rejected_idx={rejected_idx}')


    logger.info('Plotting Light Curve')

    def plot_lc():
        plt.figure(figsize=(10,3))
        plt.title(f'Mean Counts per Frame | {obsid}')
        plt.scatter(time_windows_gti[:-1], lc_HE_good, c='b', s=5, label=f'GTI Data')
        plt.scatter(time_windows_gti[:-1], lc_HE_bad, c='r', s=5, label=f'BTI Data')
        plt.axhline(threshold_GTI, c='b', ls='dotted', lw=1.0, label=f'GTI threshold = {threshold_GTI}')

        for ind in rejected_idx: # Plot rejected indexs
            plt.axvspan(time_windows[ind],time_windows[ind+1],facecolor='r', alpha=0.2)
            plt.axvline(time_windows[ind], color='red', lw=1.0, ls='dotted')
        plt.yscale('log')
        plt.xlabel('Time (s)')
        plt.ylabel('Counts (N)')
        plt.legend()

        fig_savepath = path_time_interval / f'Lightcurve_HighEnergy.png'
        plt.savefig(fig_savepath)
        # plt.show()

    plot_lc()

    #Drop BTI if necessary
    if gti_only:
        logger.info('gti_only=True, dropping bad frames from Data Cube')
        dims_img  = (cube_EPIC.shape[0], cube_EPIC.shape[1],1)
        nan_image = np.full(shape=dims_img, fill_value=np.nan, dtype=np.float64)
        cube_EPIC[:,:,rejected_idx] = nan_image

    return cube_EPIC, coordinates_XY

if __name__ == "__main__":
    #cube,coordinates_XY = read_EPIC_events_file('0831790701', 20, 500,gti_only=True)

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
            logger.info(f'Could not read {obs} {e}')



    # for frame_ind in range(cube.shape[2]):
    #     frame = cube[:,:,frame_ind]
    #     plt.imshow(frame)
    #     plt.savefig(os.path.join(data_processed,'0831790701',f'TestGTI/Test_{frame_ind}.png'))
    #     plt.close()
