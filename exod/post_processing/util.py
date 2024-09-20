from exod.utils.path import savepaths_combined
import pandas as pd

def get_lc(key, df_lc_idx):
    """ label : ('0761070101_0_5_0.2_12.0', '0')"""
    key = str(key)
    start, stop = df_lc_idx.loc[key]
    df_lc = pd.read_hdf(savepaths_combined['lc'], start=start, stop=stop)
    df_lc['t0'] = df_lc['time'] - df_lc['time'].min()
    return df_lc
