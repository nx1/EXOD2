from exod.utils.path import data_raw, data_processed, data_results
from exod.utils.logger import logger

from astropy.io import fits

class Observation:
    def __init__(self, obsid):
        self.obsid = obsid
        self.path_raw = data_raw / obsid
        self.path_processed = data_processed / obsid
        self.path_results = data_results / obsid

    def __repr__(self):
        return f'Observation({self.obsid})'

    def get_event_lists_raw(self):
        evt_raw = list(self.path_raw.glob('*EVLI*FTZ'))
        self.events_raw = [EventList(e) for e in evt_raw]

    def get_event_lists_processed(self):
        evt_processed = list(self.path_processed.glob('*FILT.fits'))
        self.events_processed = [EventList(e) for e in evt_processed]

    def get_images(self):
        img_processed = list(self.path_processed.glob('*IMG'))
        self.images = [Image(i) for i in img_processed]

    def get_files(self):
        self.get_images()
        self.get_event_lists_raw()
        self.get_event_lists_processed()


class EventList:
    def __init__(self, path):
        self.path = path

    def __repr__(self):
        return f'EventList({self.path})'

    def read(self):



class Image:
    def __init__(self, path):
        self.path = path

    def __repr__(self):
        return f'Image({self.path})'

if __name__ == "__main__":
    obs = Observation('0760380201')
    obs.get_files()