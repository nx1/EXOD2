from exod.processing.data_cube import DataCubeXMM
from exod.pre_processing.bti import get_high_energy_lc, get_bti, get_rejected_idx, plot_bti, get_rejected_idx_bool
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
    def __init__(self, event_list, time_interval=50, size_arcsec=10,
                 gti_only=True, min_energy=0.2, max_energy=12.0,
                 gti_threshold=0.5):

        self.event_list    = event_list
        self.time_interval = time_interval
        self.size_arcsec   = size_arcsec
        self.gti_only      = gti_only
        self.min_energy    = min_energy
        self.max_energy    = max_energy
        self.gti_threshold = gti_threshold

        self.pixel_size = size_arcsec / 0.05
        self.extent     = 51840
        self.n_pixels   = int(self.extent / self.pixel_size)
        self.bin_x = np.linspace(0, self.extent, self.n_pixels+1)
        self.bin_y = np.linspace(0, self.extent, self.n_pixels+1)
        self.bin_t = self.calc_time_bins()

    def __repr__(self):
        return f"DataLoader(events_list={self.event_list})"

    def run(self):
        if self.gti_only:
            self.calculate_bti()

        self.event_list.filter_by_energy(self.min_energy, self.max_energy)
        self.create_data_cube()

        if self.gti_only:
            self.drop_bti_from_data_cube()

    def calc_time_bins(self):
        t_lo = self.event_list.time_min
        t_hi = self.event_list.time_max
        t_i  = self.time_interval
        n_time_bins = int((t_hi - t_lo) / t_i)
        time_stop = t_lo + n_time_bins * t_i
        time_bins = np.arange(t_lo, time_stop + 1, t_i)
        return time_bins

    def calculate_bti(self):
        time_window_gti, lc_high_energy = get_high_energy_lc(self.event_list.data)
        bti = get_bti(time=time_window_gti, data=lc_high_energy, threshold=self.gti_threshold)
        self.df_bti = pd.DataFrame(bti)
        logger.info(f'df_bti:\n{self.df_bti}')
        plot_bti(time=time_window_gti[:-1], data=lc_high_energy, threshold=self.gti_threshold, bti=bti, obsid=self.event_list.obsid)
        self.rejected_frame_idx = get_rejected_idx(bti=bti, time_windows=self.bin_t)
        self.rejected_frame_bool = get_rejected_idx_bool(self.rejected_frame_idx, time_windows=self.bin_t)

    def create_data_cube(self):
        logger.info('Creating Data Cube...')
        data_cube = DataCubeXMM(self.event_list, self.size_arcsec, self.time_interval)
        data_cube.rejected_frame_bool = self.rejected_frame_bool #TODO refactor this guy
        self.data_cube = data_cube
        return data_cube


    def drop_bti_from_data_cube(self):
        logger.info('gti_only=True, dropping bad frames from Data Cube')
        img_shape = (self.data_cube.shape[0], self.data_cube.shape[1], 1)
        img_nan = np.full(shape=img_shape, fill_value=np.nan, dtype=np.float64)
        self.data_cube.data[:, :, self.rejected_frame_idx] = img_nan

    @property
    def info(self):
        info = {
            "events_list"   : repr(self.event_list),
            "time_interval" : self.time_interval,
            "size_arcsec"   : self.size_arcsec,
            "gti_only"      : self.gti_only,
            "min_energy"    : self.min_energy,
            "max_energy"    : self.max_energy,
            "gti_threshold" : self.gti_threshold,
            "pixel_size"    : self.pixel_size,
            "extent"        : self.extent,
            "n_pixels"      : self.n_pixels
        }
        for k, v in info.items():
            logger.info(f'{k:<13} : {v}')
        return info


if __name__ == "__main__":
    from exod.xmm.observation import Observation
    obs = Observation('0860302501')
    obs.get_files()
    evt = obs.events_processed[0]
    evt.read()
    dl = DataLoader(evt)
    dl_info = dl.info
    dl.run()
    import matplotlib.pyplot as plt
    plt.show()
