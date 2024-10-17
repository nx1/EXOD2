"""
This module contains classes for filtering regions and lightcurves.
"""
from abc import ABC, abstractmethod
from itertools import chain, combinations, product


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
        return f'Filter(name="{self.name}")'


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


class FilterLcSigmaPeak(FilterBase):
    def __init__(self, name, min_sigma):
        super().__init__(name)
        self.min_sigma = min_sigma 

    def get_parameters(self):
        return {'min_sigma': self.min_sigma}

    def apply(self, df_lc_stats):
        self.df = df_lc_stats
        mask = self.df['sigma_max_B_peak'] > self.min_sigma
        self.df_filtered = self.df[mask]
        self.df_removed  = self.df[~mask]
        return self.df_filtered


class FilterLcSigmaEclipse(FilterBase):
    def __init__(self, name, min_sigma):
        super().__init__(name)
        self.min_sigma = min_sigma 

    def get_parameters(self):
        return {'min_sigma': self.min_sigma}

    def apply(self, df_lc_stats):
        self.df = df_lc_stats
        mask = self.df['sigma_max_B_eclipse'] > self.min_sigma
        self.df_filtered = self.df[mask]
        self.df_removed  = self.df[~mask]
        return self.df_filtered


class FilterLcSigmaPeakOrEclipse(FilterBase):
    def __init__(self, name, min_sigma):
        super().__init__(name)
        self.min_sigma = min_sigma 

    def get_parameters(self):
        return {'min_sigma': self.min_sigma}

    def apply(self, df_lc_stats):
        self.df = df_lc_stats
        mask = (self.df['sigma_max_B_eclipse'] > self.min_sigma) | (self.df['sigma_max_B_peak'] > self.min_sigma)
        self.df_filtered = self.df[mask]
        self.df_removed  = self.df[~mask]
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
        mask = self.df['n_max'] < self.max_counts
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


class FilterCmatchSeperation(FilterBase):
    def __init__(self, name, max_sep, direction='lower', sep_col='SEP_ARCSEC'):
        super().__init__(name)
        self.max_sep   = max_sep 
        self.direction = direction
        self.sep_col   = sep_col

    def get_parameters(self):
        return {'max_sep'   : self.max_sep,
                'direction' : self.direction}

    def apply(self, df_cmatch):
        self.df = df_cmatch
        if self.direction == 'lower':
            mask = self.df[self.sep_col] < self.max_sep
        elif self.direction == 'greater':
            mask = self.df[self.sep_col] > self.max_sep
        else:
            raise ValueError(f"Direction {self.direction} not recognized. Use 'lower' or 'greater'.")
        
        self.df_filtered = self.df[mask]
        self.df_removed  = self.df[~mask]
        return self.df_filtered


class FilterCmatchDR14Variable(FilterBase):
    def __init__(self, name):
        super().__init__(name)

    def get_parameters(self):
        return {'Variable': True}

    def apply(self, df_cmatch):
        self.df = df_cmatch
        mask = ~df_cmatch['SC_VAR_FLAG']
        self.df_filtered = self.df[mask]
        self.df_removed = self.df[~mask]
        return self.df_filtered


def generate_combinations_with_one_or_none(filters):
    """
    Generates all combinations of zero or one filter from the given list.

    This function creates an iterable that yields combinations of filters, 
    where each combination includes either no filter or exactly one filter 
    from the provided list. The function returns an iterator that produces 
    these combinations on demand.

    Parameters:
        filters (list): A list of Filters() to generate combinations from.

    Returns:
        itertools.chain: An iterator yielding tuples that contain either 
        no filters or one filter from the provided list. Each tuple is a 
        valid combination.
        
    Example:
        filters = ['filter1', 'filter2']
        result = generate_combinations_with_one_or_none(filters)
        print(list(result)) 
        # Output: [(), ('filter1',), ('filter2',)]
    """
    return chain.from_iterable(combinations(filters, r) for r in range(2))


def generate_valid_combinations(*filter_lists):
    """
    Generates all valid combinations of filters from multiple lists, allowing at most one filter from each list.

    This function takes multiple lists of filters as input and generates all possible combinations 
    where each combination contains zero or one filter from each list. The Cartesian product 
    of the combinations from each list is computed to create all valid combinations across 
    multiple filter categories.

    Parameters:
        *filter_lists (list of lists): Variable number of filter lists. Each argument is a 
        list of filters representing a different filter category. The function will combine 
        filters from each list, with each combination containing at most one filter from 
        each category.

    Returns:
        list: A list of valid filter combinations. Each combination is represented as a list of filters, 
        where the filters are chosen from the input lists. The result includes combinations with zero 
        or one filter from each list.
        
    Example:
        filters_energy = ['E1', 'E2']
        filters_time = ['T1', 'T2']
        result = generate_valid_combinations(filters_energy, filters_time)
        print(result)
        # Output: [[], ['E1'], ['E2'], ['T1'], ['T2'], ['E1', 'T1'], ['E1', 'T2'], ['E2', 'T1'], ['E2', 'T2']]
    """
    combinations_lists = [generate_combinations_with_one_or_none(filters) for filters in filter_lists]
    valid_combinations = product(*combinations_lists)
    valid_combinations = [list(chain.from_iterable(combo)) for combo in valid_combinations]
    return valid_combinations


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
