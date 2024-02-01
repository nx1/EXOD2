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
        self.time_min   = np.min(self.data['TIME'])
        self.time_max   = np.max(self.data['TIME'])

        self.check_submode()
        self.check_bad_rows()

    def filter_by_energy(self, min_energy, max_energy):
        logger.info(f'Filtering Events list by energy min_energy={min_energy} max_energy={max_energy}')
        self.data = self.data[(min_energy * 1000 < self.data['PI']) & (self.data['PI'] < max_energy * 1000)]

    def check_submode(self):
        if not ALL_SUBMODES[self.submode]:
            raise NotImplementedError(f"The submode {self.submode} is not supported")

    def check_bad_rows(self):
        if self.instrument == 'EPN':
            logger.info('Removing Bad PN Rows Struder et al. 2001b')
            self.data = self.data[~((self.data['CCDNR'] == 4) & (self.data['RAWX'] == 12)) &
                                  ~((self.data['CCDNR'] == 5) & (self.data['RAWX'] == 11)) &
                                  ~((self.data['CCDNR'] == 10) & (self.data['RAWX'] == 28))]

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
            'N_events'   : self.N_events
            }
        for k, v in info.items():
            logger.info(f'{k:<10} : {v}')
        return info
