from exod.post_processing.estimate_variability_properties import count_distant_peaks
from exod.utils.path import data_combined, savepaths_combined
from exod.utils.logger import logger

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


def largest_peak_info(df_lc):
    """
    Find the largest peak in the light curve and return some information about it.

    Parameters:
        df_lc (pd.DataFrame): DataFrame containing the light curve data.

    Returns:
        n_max_idx (int): Index of the largest peak.
        n_max_last_bin (bool): True if the largest peak is in the last bin.
        n_max_first_bin (bool): True if the largest peak is in the first bin.
        n_max_isolated_flare (bool): True if the largest peak is surrounded by zeros.
    """
    n_max_last_bin       = False
    n_max_first_bin      = False
    n_max_isolated_flare = False

    n = df_lc['n'].values
    n_max_idx = np.argmax(n)
    if n_max_idx + 1 == len(n):
        n_max_last_bin = True
        return n_max_idx, n_max_last_bin, n_max_first_bin, n_max_isolated_flare
    elif n_max_idx + 1 == -1:
        n_max_first_bin = True
        return n_max_idx, n_max_last_bin, n_max_first_bin, n_max_isolated_flare

    val_before = n[n_max_idx - 1]
    val_after = n[n_max_idx + 1]
    if (val_before == 0) & (val_after == 0):
        n_max_isolated_flare = True
        return n_max_idx, n_max_last_bin, n_max_first_bin, n_max_isolated_flare
    return n_max_idx, n_max_last_bin, n_max_first_bin, n_max_isolated_flare


def calc_features(df_lc, key):
    parts    = key.strip("()").split(", ")
    parts    = [part.strip("'") for part in parts]
    runid    = parts[0]
    label    = parts[1]
    runid_sp = runid.split("_")
    obsid    = runid_sp[0]
    subset   = int(runid_sp[1])
    t_bin    = int(runid_sp[2])
    E_low    = float(runid_sp[3])
    E_hi     = float(runid_sp[4])

    length = len(df_lc)
    n_bccd = df_lc['bccd'].sum()
    n_bti = df_lc['bti'].sum()
    ratio_bccd = n_bccd / length
    ratio_bti = n_bti / length

    ks = ks_2samp(df_lc['n'], df_lc['mu'])



    peaks_or_eclipses = (df_lc['B_peak_log'] > 6.4) | (df_lc['B_eclipse_log'] > 5.5)
    t_bin_bin_spacing = {5 : 1000, 50 : 100, 200 : 5}
    bins_min_between_peaks = t_bin_bin_spacing[t_bin]
    n_peaks = count_distant_peaks(peaks_or_eclipses, bins_min_between_peaks)

    num_B_peak_above_6_4    = count_recurring_peaks(df_lc['B_peak_log'].values, threshold=6.4)
    num_B_eclipse_above_5_5 = count_recurring_peaks(df_lc['B_eclipse_log'].values, threshold=5.5)

    n_max_idx, n_max_last_bin, n_max_first_bin, n_max_isolated_flare = largest_peak_info(df_lc)

    res = {'key'                     : key,
           'runid'                   : runid,
           'label'                   : label,
           'obsid'                   : obsid,
           'subset'                  : subset,
           't_bin'                   : t_bin,
           'E_low'                   : E_low,
           'E_hi'                    : E_hi,
           'len'                     : length,
           'n_bccd'                  : df_lc['bccd'].sum(),
           'n_bti'                   : df_lc['bti'].sum(),
           'ratio_bccd'              : ratio_bccd,
           'ratio_bti'               : ratio_bti,
           'ks_stat'                 : ks.statistic,
           'ks_pval'                 : ks.pvalue,
           'n_min'                   : df_lc['n'].min(),
           'n_max'                   : df_lc['n'].max(),
           'n_mean'                  : df_lc['n'].mean(),
           'n_std'                   : df_lc['n'].std(),
           'n_sum'                   : df_lc['n'].sum(),
           'n_skew'                  : skew(df_lc['n']),
           'n_kurt'                  : kurtosis(df_lc['n']),
           'n_max_idx'               : n_max_idx,
           'n_max_isolated_flare'    : n_max_isolated_flare,
           'n_max_first_bin'         : n_max_first_bin,
           'n_max_last_bin'          : n_max_last_bin,
           'mu_min'                  : df_lc['n'].min(),
           'mu_max'                  : df_lc['n'].max(),
           'mu_mean'                 : df_lc['n'].mean(),
           'mu_std'                  : df_lc['n'].std(),
           'mu_skew'                 : skew(df_lc['n']),
           'mu_kurt'                 : kurtosis(df_lc['n']),
           'B_peak_log_max'          : df_lc['B_peak_log'].max(),
           'B_eclipse_log_max'       : df_lc['B_eclipse_log'].max(),
           'num_B_peak_above_6_4'    : num_B_peak_above_6_4,
           'num_B_eclipse_above_5_5' : num_B_eclipse_above_5_5,
           'bin_min_between_peaks'   : bins_min_between_peaks,
           'n_peaks'                 : n_peaks}
    return res


def extract_lc_features(clobber=True):
    if not clobber:
        if savepaths_combined['lc_features'].exists():
            logger.info(f'Light curve features already exist at {savepaths_combined["lc_features"]}. Skipping...')
            df_lc_features = pd.read_csv(savepaths_combined['lc_features'])
            return df_lc_features

    df_lc_indexs = pd.read_csv(savepaths_combined['lc_idx'], index_col='Unnamed: 0')
    all_res = []
    for i, r in tqdm(df_lc_indexs.iterrows(), desc="Extracting Lightcurve Features."):
        df_lc = pd.read_hdf(savepaths_combined['lc'], start=r['start'], stop=r['stop'])
        res = calc_features(df_lc, key=i)
        all_res.append(res)
    df_lc_features = pd.DataFrame(all_res)
    logger.info(f'Saving light curve features to {savepaths_combined["lc_features"]}')
    df_lc_features.to_csv(savepaths_combined['lc_features'], index=False)
    return df_lc_features


if __name__ == "__main__":
    extract_lc_features()