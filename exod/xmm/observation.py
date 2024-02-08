from exod.xmm.event_list import EventList
from exod.xmm.image import Image
from exod.utils.path import data_raw, data_processed, data_results
from exod.utils.logger import logger

import os


class Observation:
    def __init__(self, obsid):
        self.obsid = obsid
        self.path_raw = data_raw / obsid
        self.path_processed = data_processed / obsid
        self.path_results = data_results / obsid
        self.make_dirs()

    def __repr__(self):
        return f'Observation({self.obsid})'

    def make_dirs(self):
        os.makedirs(self.path_raw, exist_ok=True)
        os.makedirs(self.path_processed, exist_ok=True)
        os.makedirs(self.path_results, exist_ok=True)

    def get_event_lists_raw(self):
        evt_raw = list(self.path_raw.glob('*EVLI*FTZ'))
        self.events_raw = [EventList(e) for e in evt_raw]

    def get_event_lists_processed(self):
        evt_processed = list(self.path_processed.glob('*FILT.fits'))
        self.events_processed = [EventList(e) for e in evt_processed]
        self.events_processed_pn = [e for e in self.events_processed if 'PI' in e.filename]
        self.events_processed_mos = [e for e in self.events_processed if 'M1' in e.filename or 'M2' in e.filename]

    def get_images(self):
        img_processed = list(self.path_processed.glob('*IMG.fits'))
        self.images = [Image(i) for i in img_processed]

    def get_files(self):
        self.get_event_lists_raw()
        self.get_event_lists_processed()
        self.get_images()

    @property
    def info(self):
        info = {'obsid' : self.obsid}

        for i, evt in enumerate(self.events_raw):
            info[f'evt_raw_{i}'] = evt.filename

        for i, evt in enumerate(self.events_processed):
            info[f'evt_filt_{i}'] = evt.filename

        for i, img in enumerate(self.images):
            info[f'img_{i}'] = img.filename

        for k, v in info.items():
            logger.info(f'{k:>10} : {v}')
        return info