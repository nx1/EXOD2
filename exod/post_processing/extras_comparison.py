from exod.post_processing.crossmatch import crossmatch_astropy_table_with_regions
from exod.post_processing.results_manager import ResultsManager
from exod.utils.path import data_util

import numpy as np
from astropy.table import Table

if __name__ == "__main__":
    rm = ResultsManager()

    tab_extras = Table.read(data_util / 'ExtraS.fits')
    tab_extras_var = tab_extras[tab_extras['UB_LC500_CO_PVAL'] < 1e-5]

    tabs_extras = {'Full'     : tab_extras,
                   'Variable' : tab_extras_var}

    tabs_exod   = {'3sig'  : rm.df_regions.copy(),
                   '5sig'  : rm.df_regions[rm.df_lc_features['filt_5sig']].copy()}

    for k, df in tabs_exod.items():
        for k2, tab in tabs_extras.items():
            obsid_overlapping = np.intersect1d(df['obsid'], tab['OBS_ID'].astype(str))

            df_regions_overlapping = df[df['obsid'].isin(obsid_overlapping)]
            tab_extras_overlapping = tab[np.isin(tab['OBS_ID'].astype(str), obsid_overlapping.astype('str'))]

            tab_cmatch = crossmatch_astropy_table_with_regions(tab_extras_overlapping, df_regions_overlapping, 'RA', 'DEC')
            MAX_SEP = 20
            mask = tab_cmatch['SEP_ARCSEC'] < MAX_SEP
            tab_cmatch_matches    = tab_cmatch[mask]
            tab_cmatch_no_matches = tab_cmatch[~mask]

            df_regions_matches    = df_regions_overlapping.iloc[tab_cmatch_matches['IDX_ORIGINAL']]
            df_regions_no_matches = df_regions_overlapping.iloc[tab_cmatch_no_matches['IDX_ORIGINAL']]

            N_exod_obsids   = len(np.unique(tab_extras['OBS_ID']))
            N_extras_obsids = df['obsid'].nunique() 

            N_exod_sources_in_overlapping_obsids   = len(np.unique(df_regions_overlapping['cluster_label']))
            N_extras_sources_in_overlapping_obsids = len(np.unique(tab_extras_overlapping['SRCID']))

            N_exod_sources_with_cmatch      = len(np.unique(df_regions_matches['cluster_label']))
            N_exod_sources_without_cmatch   = N_exod_sources_in_overlapping_obsids - N_exod_sources_with_cmatch 

            N_extras_sources_with_cmatch    = len(np.unique(tab_cmatch_matches['SRCID']))
            N_extras_sources_without_cmatch = N_extras_sources_in_overlapping_obsids - N_extras_sources_with_cmatch 
            
            perc_exod_sources_with_cmatch      = (N_exod_sources_with_cmatch      / N_exod_sources_in_overlapping_obsids) * 100
            perc_extras_sources_with_cmatch    = (N_extras_sources_with_cmatch    / N_extras_sources_in_overlapping_obsids) * 100
            perc_exod_sources_without_cmatch   = (N_exod_sources_without_cmatch   / N_exod_sources_in_overlapping_obsids) * 100
            perc_extras_sources_without_cmatch = (N_extras_sources_without_cmatch / N_extras_sources_in_overlapping_obsids) * 100

            print(f'Combination: exod={k} extras={k2}')
            print(f'Number of EXOD observations                       = {N_exod_obsids:,}')
            print(f'Number of ExtraS observations                     = {N_extras_obsids:,}')
            print(f'Number of overlapping obsids                      = {len(obsid_overlapping):,}')
            print(f'Number of exod sources in overlapping obsids      = {N_exod_sources_in_overlapping_obsids:,}')
            print(f'Number of extras sources in overlapping obsids    = {N_extras_sources_in_overlapping_obsids:,}')
            print(f'Number of exod sources with crossmatch            = {N_exod_sources_with_cmatch:,} ({perc_exod_sources_with_cmatch:.2f}%)')
            print(f'Number of exod sources without crossmatch         = {N_exod_sources_without_cmatch:,} ({perc_exod_sources_without_cmatch:.2f}%)')
            print(f'Number of extras sources with crossmatch          = {N_extras_sources_with_cmatch:,} ({perc_extras_sources_with_cmatch:.2f}%)')
            print(f'Number of extras without crossmatch               = {N_extras_sources_without_cmatch:,} ({perc_extras_sources_without_cmatch:.2f}%)')
            print('\n')

