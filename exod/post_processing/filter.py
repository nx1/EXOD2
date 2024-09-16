"""
This module contains classes for filtering regions and lightcurves.
"""
from abc import ABC, abstractmethod


class FilterBase(ABC):
    """
    Base class for filters. All filters should inherit from this class.

    Parameters:
        name (str): Name of the filter.

    Methods:
        apply(df): Apply the filter to the input dataframe.
        get_parameters(): Return a dictionary of filter-specific parameters.
        info(): Return a dictionary with information about the filter.
    """
    def __init__(self, name):
        self.name = name
        self.df = None
        self.df_filtered = None
        self.df_removed = None

    @abstractmethod
    def apply(self, df):
        """This should set self.df_regions, self.df_filtered and self.removed."""

    @abstractmethod
    def get_parameters(self):
        """This should return a dictionary of filter-specific parameters."""

    def info(self):
        info = {}
        info['name'] = self.name
        info['parameters'] = self.get_parameters()
        info['n_original'] = len(self.df)
        info['n_filtered'] = len(self.df_filtered)
        info['n_removed'] = len(self.df_removed)
        return info

    def __repr__(self):
        return f'{self.name} filter'


class FilterRegMultipleDetections(FilterBase):
    def __init__(self, name, n_obs):
        super().__init__(name)
        self.n_obs = n_obs

    def get_parameters(self):
        return {'n_obs': self.n_obs}

    def apply(self, df_regions):
        self.df = df_regions
        self.df_filtered = self.df.groupby('obsid').filter(lambda x: len(x) < 10)
        self.df_removed = self.df.groupby('obsid').filter(lambda x: len(x) >= 10)
        return self.df_filtered


class FilterRegTimeBin(FilterBase):
    def __init__(self, name, t_bin):
        super().__init__(name)
        self.t_bin = t_bin
 
    def get_parameters(self):
        return {'t_bin': self.t_bin}

    def apply(self, df_regions):
        self.df = df_regions
        mask = self.df['runid'].str.contains(f'_{self.t_bin}_')
        self.df_filtered = self.df[mask]
        self.df_removed = self.df[~mask]
        return self.df_filtered


class FilterRegEnergyRange(FilterBase):
    def __init__(self, name, min_energy, max_energy):
        super().__init__(name)
        self.min_energy = min_energy
        self.max_energy = max_energy

    def get_parameters(self):
        return {'min_energy': self.min_energy, 'max_energy': self.max_energy}

    def apply(self, df_regions):
        self.df = df_regions
        mask = self.df['runid'].str.contains(f'{self.min_energy}_{self.max_energy}')
        self.df_filtered = self.df[mask]
        self.df_removed = self.df[~mask]
        return self.df_filtered


class FilterRegBright(FilterBase):
    def __init__(self, name, max_intensity_mean):
        super().__init__(name)
        self.max_intensity_mean = max_intensity_mean

    def get_parameters(self):
        return {'max_intensity_mean': self.max_intensity_mean}

    def apply(self, df_regions):
        self.df = df_regions
        mask = self.df['intensity_mean'] < self.max_intensity_mean
        self.df_filtered = self.df[mask]
        self.df_removed = self.df[~mask]
        return self.df_filtered


class FilterRegArea(FilterBase):
    def __init__(self, name, max_area_bbox):
        super().__init__(name)
        self.max_area_bbox = max_area_bbox

    def get_parameters(self):
        return {'max_area_bbox': self.max_area_bbox}

    def apply(self, df_regions):
        self.df = df_regions
        mask = self.df['area_bbox'] < self.max_area_bbox
        self.df_filtered = self.df[mask]
        self.df_removed = self.df[~mask]
        return self.df_filtered


class FilterLcMinCounts(FilterBase):
    def __init__(self, name, min_counts):
        super().__init__(name)
        self.min_counts = min_counts

    def get_parameters(self):
        return {'min_counts': self.min_counts}

    def apply(self, df_lc_stats):
        self.df = df_lc_stats
        mask = self.df['max'] > self.min_counts
        self.df_filtered = self.df[mask]
        self.df_removed  = self.df[~mask]
        return self.df_filtered


class FilterLcMaxCounts(FilterBase):
    def __init__(self, name, max_counts):
        super().__init__(name)
        self.max_counts = max_counts

    def get_parameters(self):
        return {'max_counts': self.max_counts}

    def apply(self, df_lc_stats):
        self.df = df_lc_stats
        mask = self.df['max'] < self.max_counts
        self.df_filtered = self.df[mask]
        self.df_removed  = self.df[~mask]
        return self.df_filtered


class FilterLcBccdRatio(FilterBase):
    def __init__(self, name, ratio_bccd_max):
        super().__init__(name)
        self.ratio_bccd_max = ratio_bccd_max

    def get_parameters(self):
        return {'ratio_bccd_max': self.ratio_bccd_max}

    def apply(self, df_lc_stats):
        self.df = df_lc_stats
        mask = self.df['ratio_bccd'] < self.ratio_bccd_max
        self.df_filtered = self.df[mask]
        self.df_removed  = self.df[~mask]
        return self.df_filtered


class FilterLcLength(FilterBase):
    def __init__(self, name, min_length):
        super().__init__(name)
        self.min_length = min_length

    def get_parameters(self):
        return {'min_length': self.min_length}

    def apply(self, df_lc_stats):
        self.df = df_lc_stats
        mask = self.df['len'] > self.min_length
        self.df_filtered = self.df[mask]
        self.df_removed  = self.df[~mask]
        return self.df_filtered


if __name__ == "__main__":
    #f1 = FilterRegMultipleDetections('multiple_detections', n_obs=10)
    f2 = FilterRegBright('max_intensity', max_intensity_mean=5000)
    f3 = FilterRegArea('max_bbox', max_area_bbox=16)
    f4 = FilterRegTimeBin('time_bin', t_bin=1)
    f5 = FilterRegEnergyRange('energy_range', min_energy=0.2, max_energy=12.0)

    filters = [f2, f3, f4, f5]

    from exod.utils.path import savepaths_combined
    import pandas as pd

    df_regions = pd.read_csv(savepaths_combined['regions'])

    print('Filtering:')
    for f in filters:
        df = df_regions.copy()
        df = f.apply(df)
        print(f.info())
