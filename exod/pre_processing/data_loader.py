from exod.processing.data_cube import DataCubeXMM
from exod.pre_processing.bti import get_bti, get_bti_bin_idx, get_bti_bin_idx_bool
from exod.utils.logger import logger

import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_dd


class DataLoader:
    """
    event_list : EventList object with data already loaded.
    size_arcsec : float : Size in arcseconds of the final spatial grid on which the data is binned
    time_interval : Time in seconds for data cube binning
    gti_only : If true use only the data found in GTIs
    min_energy : Minimum energy for final data cube
    max_energy : Maximum energy for final data cube
    gti_threshold : Count rate below which will be considered good time intervals
    """
    def __init__(self, event_list, time_interval=50, size_arcsec=10, gti_only=False, min_energy=0.2, max_energy=12.0,
                 gti_threshold=0.5, remove_partial_ccd_frames=True):
        self.event_list    = event_list
        self.time_interval = time_interval
        self.size_arcsec   = size_arcsec
        self.gti_only      = gti_only
        self.min_energy    = min_energy
        self.max_energy    = max_energy
        self.gti_threshold = gti_threshold
        self.remove_partial_ccd_frames = remove_partial_ccd_frames

    def __repr__(self):
        return f"DataLoader(events_list={self.event_list})"

    def run(self):
        self.calculate_bti() # This needs to be called first as the next step filters the eventlist.
        self.event_list.filter_by_energy(self.min_energy, self.max_energy)
        self.create_data_cube()
        self.data_cube.calc_gti_bti_bins(bti=self.bti)
        if self.remove_partial_ccd_frames:
            self.data_cube.remove_frames_partial_CCDexposure()
        if self.gti_only:
            self.data_cube.mask_bti()

    def get_high_energy_lc(self):
        min_energy_he = 10.0     # minimum extraction energy for High Energy Background events
        max_energy_he = 12.0     # maximum extraction energy for High Energy Background events
        time_interval_gti = min(self.time_interval, 100) #100  # Window Size to use for GTI extraction
        data = self.event_list.data
        time_min = self.event_list.time_min
        time_max = self.event_list.time_max
        logger.info(f'min_energy_he = {min_energy_he} max_energy_he = {max_energy_he} time_interval_gti = {time_interval_gti}')
        data_he = np.array(data['TIME'][(data['PI'] > min_energy_he * 1000) & (data['PI'] < max_energy_he * 1000)])

        t_bin_he = np.arange(time_min, time_max, time_interval_gti)
        lc_he = np.histogram(data_he, bins=t_bin_he)[0] / time_interval_gti  # Divide by the bin size to get in ct/s
        return t_bin_he, lc_he

    def calculate_bti(self):
        self.t_bin_he, self.lc_he = self.get_high_energy_lc()
        self.bti = get_bti(time=self.t_bin_he, data=self.lc_he, threshold=self.gti_threshold)
        self.df_bti = pd.DataFrame(self.bti)
        
    def create_data_cube(self):
        logger.info('Creating Data Cube...')
        data_cube = DataCubeXMM(self.event_list, self.size_arcsec, self.time_interval)
        self.data_cube = data_cube
        return data_cube

    def multiply_time_interval(self, n_factor):
        logger.info(f'Rebinning the cube with longer timebins by factor {n_factor}...')
        self.data_cube.multiply_time_interval(n_factor)
        self.time_interval = self.data_cube.time_interval
        self.calculate_bti()
        self.data_cube.calc_gti_bti_bins(bti=self.bti)


    @property
    def info(self):
        info = {
            "event_list"    : self.event_list.filename,
            "time_interval" : self.time_interval,
            "size_arcsec"   : self.size_arcsec,
            "gti_only"      : self.gti_only,
            "min_energy"    : self.min_energy,
            "max_energy"    : self.max_energy,
            "gti_threshold" : self.gti_threshold,
            "remove_partial_ccd_frames" : self.remove_partial_ccd_frames
        }
        for k, v in info.items():
            logger.info(f'{k:>13} : {v}')
        return info


if __name__ == "__main__":
    from exod.xmm.observation import Observation
    from exod.xmm.event_list import EventList
    obs = Observation('0860302501')
    obs.get_files()
    obs.get_events_overlapping_subsets()
    event_list = EventList.from_event_lists(obs.events_overlapping_subsets[0])
    dl = DataLoader(event_list=event_list, size_arcsec=15, time_interval=100, gti_only=False,
                    gti_threshold=1.5, min_energy=0.2, max_energy=12)
    dl_info = dl.info
    dl.run()
    print(dl.data_cube.shape, dl.data_cube.time_interval)
    dl.multiply_time_interval(3)
    print(dl.data_cube.shape, dl.data_cube.time_interval)


    # import matplotlib.pyplot as plt
    # plt.show()


