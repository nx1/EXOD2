from astropy.table import Table, vstack
from scipy.stats import binned_statistic_dd
from astropy.io import fits
import numpy as np
from exod.utils.path import data_processed

def read_EPIC_events_file(size_arcsec, time_interval):
    """Reads the EPIC events files.
    :argument size_arcsec is the size in arseconds of the final spatial grid onto which data is binned,
    :argument time_interval is the same but for temporal dimension"""

    pixel_size = size_arcsec / 0.05  # Size of a end pixel in DetX DetY values
    extent = 70000 #Temporary extent of the cube in DetX DetY values
    nb_pixels = int(extent/pixel_size)

    data_pn = Table(fits.open(f'{data_processed}PN_pattern_clean.fits')[1].data)['X','Y','TIME','RAWX','RAWY','CCDNR']

    # Bad rows in Struder et al. 2001b
    data_pn = data_pn[~((data_pn['CCDNR']==4)&(data_pn['RAWX']==12))&
                      ~((data_pn['CCDNR']==5)&(data_pn['RAWX']==11))&
                      ~((data_pn['CCDNR']==10)&(data_pn['RAWX']==28))]
    # Eject borders. Might need to adapt this to observing modes
    data_pn = data_pn[~(data_pn['RAWX']==0)&~(data_pn['RAWX']==64)&
                      ~(data_pn['RAWY']==0)&~(data_pn['RAWY']==200)]

    data_M1 = Table(fits.open(f'{data_processed}M1_pattern_clean.fits')[1].data)['X','Y','TIME']
    data_M2 = Table(fits.open(f'{data_processed}M2_pattern_clean.fits')[1].data)['X','Y','TIME']
    data_EPIC = vstack((data_pn,data_M1,data_M2))

    #Create the data cube
    n_bins = int(np.ceil((np.max(data_EPIC['TIME']) - np.min(data_EPIC['TIME'])) / time_interval))
    stop_time = np.min(data_EPIC['TIME']) + n_bins * time_interval
    start_time = np.min(data_EPIC['TIME'])
    time_windows = np.arange(start_time, stop_time+1, time_interval)

    cube_EPIC = binned_statistic_dd((data_EPIC['X'],data_EPIC['Y'],data_EPIC['TIME']), values=None, statistic='count',
                                        bins=(np.linspace(0,extent, nb_pixels+1),
                                              np.linspace(0,extent, nb_pixels+1),
                                              time_windows))[0]
    return cube_EPIC