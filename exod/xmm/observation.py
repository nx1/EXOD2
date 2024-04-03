from exod.xmm.event_list import EventList
from exod.xmm.image import Image
from exod.utils.path import data_raw, data_processed, data_results
from exod.utils.logger import logger

import os
from scipy.cluster.hierarchy import DisjointSet
from itertools import combinations


class Observation:
    def __init__(self, obsid):
        self.obsid = obsid
        self.path_raw = data_raw / obsid / 'product'
        self.path_processed = data_processed / obsid
        self.path_results = data_results / obsid
        self.make_dirs()

        self.events_raw = []
        self.events_processed = []
        self.events_processed_pn = []
        self.events_processed_mos1 = []
        self.events_processed_mos2 = []
        self.events_overlapping_subsets = []

        self.images = []

        self.source_list = []

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
        self.events_processed      = [EventList(e) for e in evt_processed]
        self.events_processed_pn   = [e for e in self.events_processed if 'PI' in e.filename]
        self.events_processed_mos1 = [e for e in self.events_processed if 'M1' in e.filename]
        self.events_processed_mos2 = [e for e in self.events_processed if 'M2' in e.filename]

    def get_source_list(self):
        self.source_list = self.path_processed.glob('*EP*OBSMLI*FTZ')

    def get_images(self):
        img_processed = list(self.path_processed.glob('*IMG.fits'))
        self.images = [Image(i) for i in img_processed]

    def get_files(self):
        self.get_event_lists_raw()
        self.get_event_lists_processed()
        self.get_images()

    def get_events_overlapping_subsets(self):
        self.events_overlapping_subsets = get_events_overlapping_subsets(self)


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


def get_overlapping_eventlist_subsets(event_lists):
    """
    Return the overlapping eventlists for a given observation.

    event_lists : list of EventLists that have been .read()

    In most cases this will just return a list of 1 list like
    [[M1.fits, M2.fits, PN.fits'] with the event files. However,
    for some observations it will return multiple entires if there
    have been multiple seperate observations. This function will return
    all the overlapping subsets. e.g. for 2 subsets this will return: [[],[]]
    """

    def intervals_overlap(I1, I2):
        return I1[0] <= I2[1] and I1[1] >= I2[0]

    disjoint_set = DisjointSet(event_lists)
    file_intervals = {e: [e.time_min, e.time_max] for e in event_lists}

    for d1, d2 in combinations(file_intervals.items(), r=2):
        f1, I1 = d1[0], d1[1]
        f2, I2 = d2[0], d2[1]

        if intervals_overlap(I1, I2):
            disjoint_set.merge(f1, f2)
            logger.debug(f'f1: {f1.filename} : {I1} OVERLAP!')
            logger.debug(f'f2: {f2.filename} : {I2} OVERLAP!')
        else:
            logger.debug(f'f1: {f1.filename} : {I1} NO OVERLAP!')
            logger.debug(f'f2: {f2.filename} : {I2} NO OVERLAP!')
        logger.debug('=============')

    subsets = disjoint_set.subsets()
    logger.info(f'Found {len(subsets)} overlapping subsets.')

    subsets_to_return = []
    for s in subsets:
        if len(s) > 3:
            logger.info(f'Overlapping subset has more than 3 event files ({len(s)})')
            logger.info(f'Just going to take the longest three')
            m1_evt = [e for e in s if 'M1' in e.filename]
            m2_evt = [e for e in s if 'M2' in e.filename]
            pn_evt = [e for e in s if 'PN' in e.filename]

            m1_longest = max(m1_evt, key=lambda x: x.exposure)
            m2_longest = max(m2_evt, key=lambda x: x.exposure)
            pn_longest = max(pn_evt, key=lambda x: x.exposure)
            subset = [m1_longest, m2_longest, pn_longest]
            subsets_to_return.append(subset)
        else:
            subsets_to_return.append(list(s))
    return subsets_to_return


def get_events_overlapping_subsets(observation):
    """
    Get the overlapping eventlists for a given obsid.

    Returns the subsets as a list of lists
    [[001PI.fits, 001M1.fits, 001M2.fits], [002PI.fits, 002M1.fits, 002M2.fits]]
    """
    observation.get_event_lists_processed()

    # No eventlists
    if len(observation.events_processed) == 0:
        raise KeyError(f'No eventlists found for {observation.obsid}')

    # If there are just three eventlists and 1 of M1, M2 and PN then just combine them.
    if (len(observation.events_processed_pn) == 1) & (len(observation.events_processed_mos1) == 1) & (
            len(observation.events_processed_mos2) == 1):
        subsets = [[e for e in observation.events_processed]]
        return subsets
    else:
        for evt in observation.events_processed:
            evt.read()  # Read all the eventlists to get the times.
        subsets = get_overlapping_eventlist_subsets(observation.events_processed)
        return subsets



if __name__ == "__main__":
    observation = Observation('0792180301')
    observation.get_files()
    observation.get_events_overlapping_subsets()
    for subset in observation.events_overlapping_subsets:
        print(subset)