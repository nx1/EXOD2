from exod.utils.path import data_combined

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import ks_2samp
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks


def count_recurring_peaks(data, threshold):
    peaks, _ = find_peaks(data)
    high_peaks = data[peaks] > threshold
    num_high_peaks = np.sum(high_peaks)
    return num_high_peaks


def calc_features(df_lc, idx):
    parts = idx.strip("()").split(", ")
    parts = [part.strip("'") for part in parts]

    length = len(df_lc)
    n_bccd = df_lc['bccd'].sum()
    n_bti = df_lc['bti'].sum()
    ratio_bccd = n_bccd / length
    ratio_bti = n_bti / length

    ks = ks_2samp(df_lc['n'], df_lc['mu'])

    num_B_peak_above_6_4    = count_recurring_peaks(df_lc['B_peak_log'].values, threshold=6.4)
    num_B_eclipse_above_5_5 = count_recurring_peaks(df_lc['B_eclipse_log'].values, threshold=5.5)

    res = {'key'    : i,
       'runid'      : parts[0],
       'label'      : parts[1],
       'len'        : len(df_lc),
       'n_bccd'     : df_lc['bccd'].sum(),
       'n_bti'      : df_lc['bti'].sum(),
       'ratio_bccd' : ratio_bccd,
       'ratio_bti'  : ratio_bti,
       'ks_stat'    : ks.statistic,
       'ks_pval'    : ks.pvalue,
       'n_min'  : df_lc['n'].min(),
       'n_max'  : df_lc['n'].max(),
       'n_mean' : df_lc['n'].mean(),
       'n_std'  : df_lc['n'].std(),
       'n_skew' : skew(df_lc['n']),
       'n_kurt' : kurtosis(df_lc['n']),
       'mu_min'  : df_lc['n'].min(),
       'mu_max'  : df_lc['n'].max(),
       'mu_mean' : df_lc['n'].mean(),
       'mu_std'  : df_lc['n'].std(),
       'mu_skew' : skew(df_lc['n']),
       'mu_kurt' : kurtosis(df_lc['n']),
       'B_peak_log_max'    : df_lc['B_peak_log'].max(),
       'B_eclipse_log_max' : df_lc['B_eclipse_log'].max(),
       'num_B_peak_above_6_4'    : num_B_peak_above_6_4,
       'num_B_eclipse_above_5_5' : num_B_eclipse_above_5_5}
    return res


if __name__ == "__main__":
    df_lc_indexs = pd.read_csv(data_combined / 'merged_with_dr14/df_lc_indexs.csv', index_col='Unnamed: 0')
    all_res = []
    for i, r in tqdm(df_lc_indexs.iterrows()):
        df_lc = pd.read_hdf(data_combined / 'merged_with_dr14/df_lc.h5', start=r['start'], stop=r['stop'])
        res = calc_features(df_lc, idx=i)
        all_res.append(res)

    df_lc_features = pd.DataFrame(all_res)
    df_lc_features.to_csv(data_combined / 'merged_with_dr14/df_lc_features.csv', index=False)