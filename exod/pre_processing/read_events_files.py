import os
import numpy as np
from scipy.stats import binned_statistic_dd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table, vstack

from exod.utils.logger import logger
from exod.utils.path import data_processed,data_results

def read_EPIC_events_file(obsid, size_arcsec, time_interval, box_size=3, gti_only=False, min_energy=0.2, max_energy=12, instr=["pn","M1","M2"]):
    """
    Reads the EPIC events files. Returns the cube and the coordinates_XY (used for WCS conversion)
    :argument obsid of the target observation
    :argument size_arcsec is the size in arseconds of the final spatial grid onto which data is binned,
    :argument time_interval is the same but for temporal dimension
    """
    path_processed_obs = data_processed / f'{obsid}'
    path_results_obs = data_results / f'{obsid}'
    path_time_interval = path_results_obs / f'{time_interval}s'

    # Create the observation folder if it does not exist
    os.makedirs(path_results_obs, exist_ok=True)
    os.makedirs(path_time_interval, exist_ok=True)

    threshold_GTI = 1.5
    pixel_size = size_arcsec / 0.05  # Size of a end pixel in DetX DetY values
    extent = 70000 #Temporary extent of the cube in DetX DetY values
    nb_pixels = int(extent/pixel_size)

    pn_file = path_processed_obs / 'PN_pattern_clean.fits'
    m1_file = path_processed_obs / 'M1_pattern_clean.fits'
    m2_file = path_processed_obs / 'M2_pattern_clean.fits'

    logger.info(f'Reading PN events file {pn_file}')
    data_pn = Table(fits.open(pn_file)[1].data)['X','Y','TIME', 'RAWX','RAWY','CCDNR','PI']

    logger.info('Removing Bad PN Rows Struder et al. 2001b')
    data_pn = data_pn[~((data_pn['CCDNR']==4)&(data_pn['RAWX']==12))&
                      ~((data_pn['CCDNR']==5)&(data_pn['RAWX']==11))&
                      ~((data_pn['CCDNR']==10)&(data_pn['RAWX']==28))]

    logger.info('Ejecting PN Borders')
    # Eject borders. Might need to adapt this to observing modes
    data_pn = data_pn[~(data_pn['RAWX']==0)&~(data_pn['RAWX']==64)&
                      ~(data_pn['RAWY']==0)&~(data_pn['RAWY']==200)]

    logger.info(f'Reading M1 events file {m1_file}')
    data_M1 = Table(fits.open(m1_file)[1].data)['X','Y','TIME','PI']

    logger.info(f'Reading M2 events file {m1_file}')
    data_M2 = Table(fits.open(m2_file)[1].data)['X','Y','TIME','PI']

    logger.info('Getting Start and end times for events lists')
    start_M1, end_M1 = np.min(data_M1['TIME']), np.max(data_M1['TIME'])
    start_M2, end_M2 = np.min(data_M2['TIME']), np.max(data_M2['TIME'])
    start_pn, end_pn = np.min(data_pn['TIME']), np.max(data_pn['TIME'])

    logger.info('Combining event lists')
    list_to_stack = []
    if "pn" in instr:
        list_to_stack.append(data_pn)
    if "M1" in instr:
        list_to_stack.append(data_M1)
    if "M2" in instr:
        list_to_stack.append(data_M2)

    data_EPIC = vstack(list_to_stack)

    logger.info('Trimming TIME on data cube')
    data_EPIC = data_EPIC[(max(start_M1,start_M2,start_pn)<data_EPIC['TIME']) &
                          (data_EPIC['TIME']<min(end_M1,end_M2,end_pn))]

    logger.info('Calculating Time windows')
    start_time = np.min(data_EPIC['TIME'])
    max_time   = np.max(data_EPIC['TIME'])
    n_bins     = int(np.ceil((max_time - start_time) / time_interval))
    stop_time  = start_time + n_bins * time_interval

    time_windows     = np.arange(start_time, stop_time+1, time_interval)
    time_windows_gti = np.arange(start_time, stop_time+1, 100)

    logger.info(f'start_time = {start_time}')
    logger.info(f'stop_time  = {stop_time}')
    logger.info(f'max_time   = {max_time}')
    logger.info(f'n_bins     = {n_bins}')

    logger.info('Calculating High Energy Lightcurve')
    data_HE = np.array(data_EPIC['TIME'][(data_EPIC['PI']>10000) & (data_EPIC['PI']<12000)])
    lc_HE   = np.histogram(data_HE, bins=time_windows_gti)[0]/100

    logger.info('Calculating GTI Table')
    gti_tab = (lc_HE<threshold_GTI).astype(int)

    logger.info(f'Filtering Grouped Events list by energy emin={min_energy} emax={max_energy}')
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


    # Plotting
    logger.info('Plotting Light Curve') 
    plt.figure(figsize=(15,5))
    plt.title(f'{obsid}')
    plt.scatter(time_windows_gti[:-1], np.where(lc_HE<threshold_GTI, lc_HE, np.nan), c='b', s=5)
    plt.scatter(time_windows_gti[:-1], np.where(lc_HE>=threshold_GTI, lc_HE, np.nan), c='r', s=5)
    bti_start = np.where(np.diff(gti_tab) == -1)[0]
    bti_stop = np.where(np.diff(gti_tab) == 1)[0]
    indices_timebinsleft_gtistart = np.searchsorted(time_windows, time_windows_gti[bti_start])
    indices_timebinsleft_gtistop = np.searchsorted(time_windows, time_windows_gti[bti_stop])
    rejected = [list(range(start, end + 1)) for (start, end) in zip(indices_timebinsleft_gtistart, indices_timebinsleft_gtistop)]
    if len(indices_timebinsleft_gtistop)<len(indices_timebinsleft_gtistart): #If you have remaining BTIs at the end
        rejected+=[list(range(indices_timebinsleft_gtistart[len(indices_timebinsleft_gtistop)], len(time_windows)))]
    rejected = list(set([index - 1 for bti in rejected for index in bti]))

    for ind in rejected:
        plt.axvspan(time_windows[ind],time_windows[ind+1],facecolor='r',alpha=0.2)
    plt.yscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('Rate')
    fig_savepath = path_time_interval / f'Lightcurve_HighEnergy.png'
    plt.savefig(fig_savepath)
    plt.show()

    #Drop BTI if necessary
    if gti_only:
        nan_image = np.empty((cube_EPIC.shape[0], cube_EPIC.shape[1],1))
        nan_image[:] = np.nan
        cube_EPIC[:,:,rejected]=nan_image

    return cube_EPIC, coordinates_XY


if __name__ == "__main__":
    #cube,coordinates_XY = read_EPIC_events_file('0831790701', 20, 500,gti_only=True)

    from exod.pre_processing.download_observations import read_observation_ids
    from exod.utils.path import data
    import pandas as pd

    obsids = read_observation_ids(data / 'observations.txt')
 
    for obs in obsids:
        try:
            cube,coordinates_XY = read_EPIC_events_file(obsid=obs,
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
