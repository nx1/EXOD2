from exod.utils.logger import logger
from exod.xmm.epic_submodes import ALL_SUBMODES

from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.table import Table


class EventList:
    def __init__(self, path):
        self.path = Path(path)
        self.filename = self.path.name

    def __repr__(self):
        return f'EventList({self.path})'

    def read(self):
        self.hdul = fits.open(self.path)
        self.header = self.hdul[1].header
        self.data = Table(self.hdul[1].data)

        self.obsid      = self.header['OBS_ID']
        self.instrument = self.header['INSTRUME']
        self.submode    = self.header['SUBMODE']
        self.date       = self.header['DATE-OBS']
        self.object     = self.header['OBJECT']
        self.exposure   = self.header['TELAPSE']
        self.N_events   = self.header['NAXIS2']
        self.mean_rate  = self.N_events / self.exposure
        self.time_min   = np.min(self.data['TIME'])
        self.time_max   = np.max(self.data['TIME'])

        self.check_submode()
        self.remove_bad_rows()
        self.remove_borders()

    def filter_by_energy(self, min_energy, max_energy):
        logger.info(f'Filtering Events list by energy min_energy={min_energy} max_energy={max_energy}')
        self.data = self.data[(min_energy * 1000 < self.data['PI']) & (self.data['PI'] < max_energy * 1000)]

    def check_submode(self):
        if not ALL_SUBMODES[self.submode]:
            raise NotImplementedError(f"The submode {self.submode} is not supported")

    def remove_bad_rows(self):
        if self.instrument == 'EPN':
            logger.info('Removing Bad PN Rows Struder et al. 2001b')
            self.data = self.data[~((self.data['CCDNR'] == 4) & (self.data['RAWX'] == 12)) &
                                  ~((self.data['CCDNR'] == 5) & (self.data['RAWX'] == 11)) &
                                  ~((self.data['CCDNR'] == 10) & (self.data['RAWX'] == 28))]

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


    @property
    def info(self):
        info = {
            'filename'   : self.filename,
            'obsid'      : self.obsid,
            'instrument' : self.instrument,
            'submode'    : self.submode,
            'date'       : self.date,
            'object'     : self.object,
            'exposure'   : self.exposure,
            'N_events'   : self.N_events,
            'mean_rate'  : self.mean_rate
            }
        for k, v in info.items():
            logger.info(f'{k:>10} : {v}')
        return info
