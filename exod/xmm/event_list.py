from exod.utils.logger import logger
from exod.utils.path import data_util
from exod.xmm.epic_submodes import ALL_SUBMODES

from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack, setdiff


class EventList:
    """
    Class for handling EventList objects.

    Attributes:
        path (Path): Path to the event list file.
        filename (str): Name of the event list file.
        data (Table): Data from the event list file.
        is_read (bool): Flag indicating if the event list file has been read.
        is_merged (bool): Flag indicating if the event list is a merged one.

    Methods:
        read: Reads the event list file and extracts necessary information.
        from_event_lists: Creates a merged EventList from a list of existing ones.
        filter_by_energy: Filters the event list data by energy range.
        check_submode: Checks if the submode of the event list is supported.
        is_supported_submode: Returns if the submode of the event list is supported.
        remove_MOS_central_ccd: Removes the central CCD of MOS if not in PrimeFullWindow.
        remove_bad_rows: Removes bad PN rows as per Struder et al. 2001b.
        remove_borders: Removes borders from the event list data.
        get_ccd_bins: Gets the CCD bins for the instrument.
        unload_data: Unloads the data from the event list to save memory.
        info: Returns a dictionary containing information about the event list.
    """
    def __init__(self, path):
        self.path      = Path(path)
        self.filename  = self.path.name
        self.data      = None
        self.is_read   = False
        self.is_merged = False

        self.N_event_lists = 1

    def __repr__(self):
        return f'EventList({self.path})'

    def read(self, remove_bad_rows=True, remove_borders=True, remove_MOS_central_ccd=True, remove_hot_pixels=True):
        if self.is_read:
            return None
        self.hdul = fits.open(self.path)
        self.header = self.hdul[1].header
        self.data = Table(self.hdul[1].data)

        self.obsid      = self.header['OBS_ID']
        self.instrument = self.header['INSTRUME']
        self.submode    = self.header['SUBMODE']
        self.date       = self.header['DATE-OBS']
        self.revolution = self.header['REVOLUT']
        self.object     = self.header['OBJECT']
        self.exposure   = self.header['TELAPSE']
        self.pnt_angle  = self.header['PA_PNT']
        # self.N_events   = self.header['NAXIS2']
        # self.mean_rate  = self.N_events / self.exposure
        self.time_min   = np.min(self.data['TIME'])
        self.time_max   = np.max(self.data['TIME'])

        # self.check_submode()
        if remove_bad_rows:
            self.remove_bad_rows()
        if remove_borders:
            self.remove_borders()
        if remove_MOS_central_ccd:
            self.remove_MOS_central_ccd()
        if remove_hot_pixels:
            self.remove_hot_pixels()
        self.is_read = True

    @classmethod
    def from_event_lists(cls, event_lists):
        """
        Create a merged EventList from a list of existing ones.
        event_lists = [EventList, EventList, EventList]
        """
        # EventList Object to return
        event_list = cls.__new__(cls) # Create the object without calling .__init__()

        for e in event_lists:
            e.read()

        # Remove unsupported EventLists
        event_lists = [e for e in event_lists if e.is_supported_submode()]
        if not event_lists:
            raise NotImplementedError(f'None of the supplied event lists are in a supported submode!')

        # Store the starts and ends of all event lists
        starts = [e.time_min for e in event_lists]
        stops = [e.time_max for e in event_lists]
        latest_start = max(starts)
        earliest_stop = min(stops)

        # Combine the data into a single table
        data_stacked = vstack([e.data for e in event_lists])

        # Crop when instruments are not all online
        # data_stacked = data_stacked[(data_stacked['TIME']>latest_start)&(data_stacked['TIME']<earliest_stop)]

        # Unload the data from the constituent event lists to save memory
        # for e in event_lists:
        #     e.unload_data()

        # Write Attributes
        event_list.path     = 'merged'
        event_list.filename = str([e.filename for e in event_lists])
        event_list.is_read  = True
        event_list.is_merged = True

        event_list.hdul   = None
        event_list.header = None
        event_list.data   = data_stacked

        event_list.event_lists   = event_lists
        event_list.N_event_lists = len(event_lists)
        event_list.obsid         = event_lists[0].obsid
        event_list.instrument    = [e.instrument for e in event_lists]
        event_list.submode       = [e.submode for e in event_lists]
        event_list.date          = event_lists[0].date
        event_list.revolution    = event_lists[0].revolution
        event_list.object        = event_lists[0].object
        event_list.time_min      = latest_start
        event_list.time_max      = earliest_stop
        event_list.exposure      = event_list.time_max - event_list.time_min
        event_list.pnt_angle     = event_lists[0].pnt_angle
        # event_list.N_events      = len(data_stacked)
        # event_list.mean_rate     = event_list.N_events / event_list.exposure
        return event_list

    def filter_by_energy(self, min_energy, max_energy):
        logger.info(f'Filtering Events list by energy min_energy={min_energy} max_energy={max_energy}')
        self.data = self.data[(min_energy * 1000 < self.data['PI']) & (self.data['PI'] < max_energy * 1000)]

    def check_submode(self):
        if not ALL_SUBMODES[self.submode]:
            raise NotImplementedError(f"The submode {self.submode} is not supported")

    def is_supported_submode(self):
        return ALL_SUBMODES[self.submode]

    def remove_MOS_central_ccd(self):
        if (self.instrument in ('EMOS1','EMOS2')) and (self.submode!='PrimeFullWindow'):
            logger.info('Removing central CCD of MOS because NOT in PrimeFullWindow')
            self.data = self.data[~(self.data['CCDNR'] == 1)]

    def remove_bad_rows(self):
        if self.instrument == 'EPN':
            logger.info('Removing bad PN rows Struder et al. 2001b')
            self.data = self.data[~((self.data['CCDNR'] == 4) & (self.data['RAWX'] == 12)) &
                                  ~((self.data['CCDNR'] == 5) & (self.data['RAWX'] == 11)) &
                                  ~((self.data['CCDNR'] == 10) & (self.data['RAWX'] == 28))]

    def remove_hot_pixels(self):
        """
        Remove hot pixels from the event list data.
        """
        hot_pixels = {'EPN': data_util / 'hotpix_PN.csv',
                      'EMOS1': data_util / 'hotpix_M1.csv',
                      'EMOS2': data_util / 'hotpix_M2.csv'}
        pre = self.N_events
        tab_hotpix = Table.read(hot_pixels[self.instrument], format='csv')
        tab_hotpix = tab_hotpix[(self.revolution > tab_hotpix['REV1']) & (self.revolution < tab_hotpix['REV2'])]
        if len(tab_hotpix) == 0:
            logger.info(f'No Hot Pixels found for {self.instrument}')
            return None
        self.data = setdiff(table1=self.data, table2=tab_hotpix, keys=['CCDNR', 'RAWX', 'RAWY'])
        logger.info(f'Removed Hot Pixels {self.instrument} | Pre: {pre} Post: {self.N_events} (-{pre - self.N_events})')


    def remove_borders(self):
        """
        For PN the RAWY is the long axis. The removal of 1px from each side gets rid of the weird hot-spot
        that appears between two of the CCDs.

        PrimeFullWindow             PrimeLargeWindow   PrimeSmallWindow
        & PrimeFullWindowExtended
        RAWY MAX: 200               RAWY MAX: 200      RAWY MAX: 200
        RAWY MIN: 13                RAWY MIN: 102      RAWY MIN: 137
        RAWX MAX: 64                RAWX MAX: 64       RAWX MAX: 64
        RAWX MIN: 1                 RAWX MIN: 1        RAWX MIN: 1
        """
        margin = 3
        if self.instrument == 'EPN' and (self.submode == 'PrimeFullWindow' or self.submode == 'PrimeFullWindowExtended'):
            logger.info(f'Removing Borders: {self.instrument} {self.submode}')
            self.data = self.data[self.data['RAWY'] > 20+margin]
            self.data = self.data[self.data['RAWY'] < 200-margin]
            self.data = self.data[self.data['RAWX'] < 64-margin]
            self.data = self.data[self.data['RAWX'] > 1+margin]

        if self.instrument == 'EPN' and self.submode == 'PrimeLargeWindow':
            logger.info(f'Removing Borders: {self.instrument} {self.submode}')
            self.data = self.data[self.data['RAWY'] < 200-margin]
            self.data = self.data[self.data['RAWX'] < 64-margin]
            self.data = self.data[self.data['RAWX'] > 1+margin]

    def get_ccd_bins(self):
        """Get the CCD bins for the instrument. This is used in calculating the bad CCD bins."""
        ccd_bins = list(set(self.data['CCDNR']))
        ccd_bins.append(ccd_bins[-1] + 1)  # Just to get a right edge for the final bin
        return ccd_bins

    def unload_data(self):
        del(self.data)
        self.is_read = False

    @property
    def N_events(self):
        return len(self.data)

    @property
    def mean_rate(self):
        return self.N_events / self.exposure

    @property
    def info(self):
        info = {
            'filename'      : self.filename,
            'obsid'         : self.obsid,
            'N_event_lists' : self.N_event_lists,
            'instrument'    : self.instrument,
            'submode'       : self.submode,
            'date'          : self.date,
            'revolution'    : self.revolution,
            'object'        : self.object,
            'exposure'      : self.exposure,
            'pnt_angle'     : self.pnt_angle,
            'N_events'      : self.N_events,
            'mean_rate'     : self.mean_rate
            }
        for k, v in info.items():
            logger.info(f'{k:>10} : {v}')
        return info

if __name__ == "__main__":
    evt = EventList('../../data/processed/0891803001/P0891803001PNS003PIEVLI0000_FILT.fits')
    evt.read()
    print(repr(evt.header))