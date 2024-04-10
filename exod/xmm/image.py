from exod.utils.logger import logger

import warnings
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning

warnings.filterwarnings(action='ignore', category=FITSFixedWarning, append=True)


class Image:
    def __init__(self, path):
        self.path = Path(path)
        self.filename = self.path.name
        self.is_read = False

    def __repr__(self):
        return f'Image({self.filename})'

    def read(self, wcs_only=False):
        if self.is_read:
            return None
        self.hdul = fits.open(self.path)
        self.header = self.hdul[0].header
        if not wcs_only:
            self.data = self.hdul[0].data
        self.wcs = WCS(self.header)
        self.is_read = True