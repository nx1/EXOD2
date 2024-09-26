import pandas as pd
from tqdm import tqdm
from exod.post_processing.filter import *
from exod.utils.path import savepaths_combined
from exod.post_processing.cluster_regions import ClusterRegions

class Subset:
    def __init__(self, filters, df_regions):
        self.filters    = filters
        self.df         = df_regions.copy()
        self.name       = '_'.join([f.name for f in filters])
        self.n_reg      = 0
        self.n_unique   = 0

    def __repr__(self):
        return f"Subset({self.name})"

    def calc(self):
        for f in self.filters:
            self.df = f.apply(self.df)
        self.n_reg = len(self.df)
        self.n_unique = self.df['cluster_label'].nunique()
        # print(f"Subset: {self.name!s:<70} | N_reg: {self.n_reg:<7} | N_unique: {self.n_unique:<7}")

    def get_df(self):
        return self.df


class SubsetManager:
    def __init__(self):
        self.subsets = []

    def add_subset(self, subset):
        self.subsets.append(subset)
        self.number_subsets()

    def add_subsets(self, subsets):
        self.subsets.extend(subsets)
        self.number_subsets()

    def number_subsets(self):
        for i, s in enumerate(self.subsets):
            s.id = i

    def calc_all(self):
        print(f'Calculating {len(self.subsets)} subsets...')
        for s in tqdm(self.subsets):
            s.calc()

    def get_subset_by_index(self, index):
        return self.subsets[index]


def get_filters():
    filters_energy = [FilterRegEnergyRange(name='E_band_full', min_energy=0.2, max_energy=12.0),
                      FilterRegEnergyRange(name='E_band_soft', min_energy=0.2, max_energy=2.0),
                      FilterRegEnergyRange(name='E_band_hard', min_energy=2.0, max_energy=12.0)]
    
    filters_time = [FilterRegTimeBin(name='t_bin_5',   t_bin=5),
                    FilterRegTimeBin(name='t_bin_50',  t_bin=50),
                    FilterRegTimeBin(name='t_bin_200', t_bin=200)]
    
    filters_sigma = [FilterLcSigmaPeak('3_sigma_peak',       min_sigma=3),
                     FilterLcSigmaPeak('5_sigma_peak',       min_sigma=5),
                     FilterLcSigmaEclipse('3_sigma_eclipse', min_sigma=3),
                     FilterLcSigmaEclipse('5_sigma_eclipse', min_sigma=5)]
    
    filters_cmatch = [FilterCmatchSeperation('DR14 < 40"',   max_sep=40, direction='lower',    sep_col='DR14_SEP_ARCSEC'),
                      FilterCmatchSeperation('SIMBAD < 40"', max_sep=40, direction='lower',    sep_col='SIMBAD_SEP_ARCSEC'),
                      FilterCmatchSeperation('DR14 > 40"',   max_sep=40, direction='greater',  sep_col='DR14_SEP_ARCSEC'),
                      FilterCmatchSeperation('SIMBAD > 40"', max_sep=40, direction='greater',  sep_col='SIMBAD_SEP_ARCSEC')]
    
    filters = [filters_energy, filters_time, filters_sigma, filters_cmatch]
    return filters

if __name__ == "__main__":
    df_lc_features    = pd.read_csv(savepaths_combined['lc_features'])
    df_cmatch_dr14    = pd.read_csv(savepaths_combined['cmatch_dr14'])
    df_cmatch_simbad  = pd.read_csv(savepaths_combined['cmatch_simbad'])
    df_regions        = pd.read_csv(savepaths_combined['regions'])
    cluster_regions   = ClusterRegions(df_regions)

    print(len(df_lc_features), len(df_cmatch_dr14), len(df_cmatch_simbad), len(df_regions))
    df_regions['sigma_max_B_peak']    = df_lc_features['sigma_max_B_peak']
    df_regions['sigma_max_B_eclipse'] = df_lc_features['sigma_max_B_eclipse']
    df_regions['DR14_SEP_ARCSEC']     = df_regions['cluster_label'].map(df_cmatch_dr14['SEP_ARCSEC'])
    df_regions['SIMBAD_SEP_ARCSEC']   = df_regions['cluster_label'].map(df_cmatch_simbad['SEP_ARCSEC'])

    filters = get_filters()
    valid_combinations = generate_valid_combinations(*filters)
    sm = SubsetManager()
    sm.add_subsets([Subset(f, df_regions) for f in valid_combinations])
    sm.calc_all()

    for s in sm.subsets:
        print(f'{s.name!s:<70} | N_reg: {s.n_reg:<7} | N_unique: {s.n_unique:<7}')
