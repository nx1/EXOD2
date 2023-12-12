from astropy.table import Table, vstack
from scipy.stats import binned_statistic_dd
from astropy.io import fits
import numpy as np
import os
from exod.utils.path import data_processed

def read_EPIC_events_file(obsid, size_arcsec, time_interval, box_size=3, gti_only=False, emin=0.2, emax=12,
                          instr=["pn","M1","M2"]):
    """Reads the EPIC events files. Returns the cube and the coordinates_XY (used for WCS conversion)
    :argument obsid of the target observation
    :argument size_arcsec is the size in arseconds of the final spatial grid onto which data is binned,
    :argument time_interval is the same but for temporal dimension"""

    pixel_size = size_arcsec / 0.05  # Size of a end pixel in DetX DetY values
    extent = 70000 #Temporary extent of the cube in DetX DetY values
    nb_pixels = int(extent/pixel_size)

    if gti_only:
        pn_file = os.path.join(data_processed,obsid,f'PN_clean.fits')
        m1_file = os.path.join(data_processed,obsid,f'M1_clean.fits')
        m2_file = os.path.join(data_processed,obsid,f'M2_clean.fits')
    else:
        pn_file = os.path.join(data_processed,obsid,f'PN_pattern_clean.fits')
        m1_file = os.path.join(data_processed,obsid,f'M1_pattern_clean.fits')
        m2_file = os.path.join(data_processed,obsid,f'M2_pattern_clean.fits')
    data_pn = Table(fits.open(pn_file)[1].data)['X','Y','TIME', 'RAWX','RAWY','CCDNR','PI']

    # Bad rows in Struder et al. 2001b
    data_pn = data_pn[~((data_pn['CCDNR']==4)&(data_pn['RAWX']==12))&
                      ~((data_pn['CCDNR']==5)&(data_pn['RAWX']==11))&
                      ~((data_pn['CCDNR']==10)&(data_pn['RAWX']==28))]
    # Eject borders. Might need to adapt this to observing modes
    data_pn = data_pn[~(data_pn['RAWX']==0)&~(data_pn['RAWX']==64)&
                      ~(data_pn['RAWY']==0)&~(data_pn['RAWY']==200)]

    data_M1 = Table(fits.open(m1_file)[1].data)['X','Y','TIME','PI']
    data_M2 = Table(fits.open(m2_file)[1].data)['X','Y','TIME','PI']

    # Save starting and ending time for each of them
    start_M1, end_M1 = np.min(data_M1['TIME']), np.max(data_M1['TIME'])
    start_M2, end_M2 = np.min(data_M2['TIME']), np.max(data_M2['TIME'])
    start_pn, end_pn = np.min(data_pn['TIME']), np.max(data_pn['TIME'])

    list_to_stack = []
    if "pn" in instr:
        list_to_stack.append(data_pn)
    if "M1" in instr:
        list_to_stack.append(data_M1)
    if "M2" in instr:
        list_to_stack.append(data_M2)
    data_EPIC = vstack(list_to_stack)

    data_EPIC = data_EPIC[(max(start_M1,start_M2,start_pn)<data_EPIC['TIME']) &
                          (data_EPIC['TIME']<min(end_M1,end_M2,end_pn))]
    data_EPIC = data_EPIC[(emin*1000<data_EPIC['PI']) & (data_EPIC['PI']<emax*1000)]

    #Create the data cube
    n_bins = int(np.ceil((np.max(data_EPIC['TIME']) - np.min(data_EPIC['TIME'])) / time_interval))
    stop_time = np.min(data_EPIC['TIME']) + n_bins * time_interval
    start_time = np.min(data_EPIC['TIME'])
    time_windows = np.arange(start_time, stop_time+1, time_interval)

    cube_EPIC = binned_statistic_dd((data_EPIC['X'],data_EPIC['Y'],data_EPIC['TIME']), values=None, statistic='count',
                                        bins=(np.linspace(0,extent, nb_pixels+1),
                                              np.linspace(0,extent, nb_pixels+1),
                                              time_windows))[0]

    # Crop the cube
    indices_image = np.where(np.sum(cube_EPIC, axis=2) > 0)
    cropping_angles = (np.min(indices_image[0]) - box_size, np.max(indices_image[0]) + 1 + box_size,
                       np.min(indices_image[1]) - box_size, np.max(indices_image[1]) + 1 + box_size)
    cube_EPIC = cube_EPIC[cropping_angles[0]:cropping_angles[1],cropping_angles[2]:cropping_angles[3]]
    coordinates_XY = (np.linspace(0,extent, nb_pixels+1)[cropping_angles[0]:cropping_angles[1]],
                   np.linspace(0,extent, nb_pixels+1)[cropping_angles[2]:cropping_angles[3]])
    #cube_EPIC[np.where(np.sum(cube_EPIC, axis=2) < 1)] = np.full(len(time_windows) - 1, np.nan)

    return cube_EPIC, coordinates_XY

if __name__ == "__main__":
    cube,coordinates_XY = read_EPIC_events_file('0831790701', 20, 1000)
