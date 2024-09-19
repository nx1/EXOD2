import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from astropy.table import Table, hstack
from astropy.coordinates import SkyCoord
from astropy import units as u

from exod.post_processing.rotate_regions import rotate_regions_to_detector_coords, hot_regions
from exod.utils.path import savepaths_combined, data_combined, data_catalogue
from exod.post_processing.cluster_regions import ClusterRegions
from exod.post_processing.crossmatch import crossmatch_unique_regions
from exod.post_processing.extract_lc_features import calc_df_lc_feat_filter_flags

from pathlib import Path


def create_exod_srcids(ra_deg, dec_deg):
    """Create Source Identifiers from coordinates."""
    sc = SkyCoord(ra=ra_deg, dec=dec_deg, unit='deg')
    exod_srcids = []
    for s in sc:
        ra  = s.ra.to_string(unit=u.hour, sep='', precision=2, pad=True)
        dec = s.dec.to_string(sep='', precision=2, alwayssign=True, pad=True)
        name = f"EXOD J{ra}{dec}"
        exod_srcids.append(name)
    return exod_srcids


def calc_hot_pixel_flags(df_regions, df_regions_rotated, hot_regions):
    all_masks = []
    for k, v in hot_regions.items():
        mask_t_bin = df_regions_rotated['runid'].str.contains(k)

        for i, (x,y,w,h) in enumerate(v):
            mask = ((df_regions_rotated['X_EPIC'] < x+w)
                  & (df_regions_rotated['X_EPIC'] > x)
                  & (df_regions_rotated['Y_EPIC'] < y+h)
                  & (df_regions_rotated['Y_EPIC'] > y)
                  & mask_t_bin)

            all_masks.append(mask)
    mask_hot_pixels = np.logical_or.reduce(all_masks)
    filt_hot_pixel = np.isin(df_regions['EXOD_DETID'], df_regions_rotated[mask_hot_pixels]['EXOD_DETID'])
    return filt_hot_pixel

def make_exod_catalogue():
    cat_savepath = data_catalogue
    df_lc_feat   = pd.read_csv(savepaths_combined['lc_features'])
    df_regions   = pd.read_csv(savepaths_combined['regions'])

    cr = ClusterRegions(df_regions)
    df_regions_unique = cr.df_regions_unique

    df_regions['EXOD_DETID'] = df_regions['runid'] + '_' + df_regions['label'].astype(str)

    df_lc_feat = calc_df_lc_feat_filter_flags(df_lc_feat)

    dfs_cmatch = crossmatch_unique_regions(df_regions_unique.reset_index(), clobber=False)

    df_regions_unique['region_num'] = df_regions_unique.index.values
    df_regions_unique['EXOD_SRCID'] = create_exod_srcids(ra_deg=df_regions_unique['ra_deg'], dec_deg=df_regions_unique['dec_deg'])

    assert len(df_regions_unique['EXOD_SRCID'].unique()) == len(df_regions_unique)

    df_regions_unique['n_detections']      = df_regions_unique['idxs'].apply(len)
    df_regions_unique['mean_total_counts'] = df_regions.groupby('cluster_label')['intensity_mean'].mean()
    df_regions_unique['mean_area_bbox']    = df_regions.groupby('cluster_label')['area_bbox'].mean()

    dict_exod_id = df_regions_unique['EXOD_SRCID'].to_dict()

    df_regions['unique_reg_id']        = df_regions['cluster_label']
    df_regions['EXOD_SRCID']           = [dict_exod_id[c] for c in df_regions['cluster_label']]
    df_regions['obsid']                = df_regions['runid'].str.extract(r'(\d{10})')
    df_regions['subset']               = df_regions['runid'].str.extract(r'_(\d+)_')[0]
    df_regions['time_bin']             = df_regions['runid'].str.extract(r'_\d+_(\d+)_')[0]
    df_regions['energy_range']         = df_regions['runid'].str.extract(r'_\d+_\d+_([\d\.]+_[\d\.]+)')[0]
    df_regions['mean_pixel_intensity'] = df_regions['intensity_mean'] / df_regions['area_bbox']

    # Append lc columns
    lc_cols = ['n_min', 'n_max', 'n_mean', 'n_std', 'n_sum', 'n_skew', 'n_kurt', 'sigma_max_B_peak', 'sigma_max_B_eclipse', 'bin_min_between_peaks', 'n_peaks', 'ratio_bti', 'ks_pval']
    for col in lc_cols:
        df_regions[f'lc_{col}'] = df_lc_feat[col]

    df_regions['filt_tbin_5_n_l_5']    = df_lc_feat['filt_tbin_5_n_l_5']
    df_regions['filt_exclude_obsid']   = df_lc_feat['filt_exclude_obsid']
    df_regions['filt_5sig']            = df_lc_feat['filt_5sig']
    df_regions['filt_g_20_detections'] = df_lc_feat['filt_g_20_detections']

    df_regions_rotated = rotate_regions_to_detector_coords(clobber=False)
    filt_hot_pixel = calc_hot_pixel_flags(df_regions, df_regions_rotated, hot_regions)

    df_regions['filt_hot_pixel'] = filt_hot_pixel
    print(f'Number of hot pixel regions: {len(df_regions[df_regions["filt_hot_pixel"]])} / {len(df_regions)} ({len(df_regions[df_regions["filt_hot_pixel"]]) / len(df_regions) * 100:.2f}%)')


    cols = {
        'EXOD_DETID'              : ['Unique detection ID', None],
        'EXOD_SRCID'              : ['Identifier for grouped detections within 20".', None],
        'unique_reg_id'           : ['ID for Unique region in table 2', None],
        'ra_deg'                  : ['RA of the detection in degrees (J2000)', 'deg'],
        'dec_deg'                 : ['DEC of the detection in degrees (J2000)', 'deg'],
        'X'                       : ['X in projected sky coordinates', None],
        'Y'                       : ['Y in projected sky coordinates', None],
        'area_bbox'               : ['Area of detection & Extraction region', None],
        'mean_pixel_intensity'    : ['Mean counts in one image pixel', 'count'],
        'lc_n_max'                : ['Maximum counts in 1 bin of LC', 'count'],
        'lc_n_min'                : ['Maximum counts in 1 bin of LC', 'count'],
        'lc_n_sum'                : ['Summed counts in LC', 'count'],
        'lc_n_mean'               : ['Mean counts in LC', 'count'],
        'lc_n_std'                : ['std counts in LC', 'count'],
        'lc_n_skew'               : ['LC skew', 'count'],
        'lc_n_kurt'               : ['LC kurtosis', 'count'],
        'lc_sigma_max_B_peak'     : ['Maximum sigma equivalent of peak in LC', None],
        'lc_sigma_max_B_eclipse'  : ['Maximum Sigma equivalent of eclipse in LC', None],
        'lc_ratio_bti'            : ['Percent of LC that is a bad time interval', None],
        'lc_bin_min_between_peaks': ['Minimum distance used for peak finding', None],
        'lc_n_peaks'              : ['Number of peaks found in LC', None],
        'lc_ks_pval'              : ['Kolmogorov Smirnov test calculated against BG', None],
        'filt_tbin_5_n_l_5'       : ['Flag for lcs with < 5 counts in a bin', None],
        'filt_exclude_obsid'      : ['Flag for exluded observation IDs', None],
        'filt_5sig'               : ['Flag for 5 sigma detection', None],
        'filt_g_20_detections'    : ['Flag for many detections in 1 obsid', None],
        'filt_hot_pixel'          : ['Flag for hot pixel regions', None]
    }

    # Create Table
    tab_exod_cat = Table.from_pandas(df_regions[cols.keys()])

    # Write Units
    for col, v in cols.items():
        unit = v[1]
        tab_exod_cat[col].unit = unit

    # Save Table
    clobber = True
    savepath = cat_savepath / 'EXOD_DR1_cat.fits'
    tab_exod_cat.write(savepath, format='fits', overwrite=clobber)
    # Write descriptions to header
    with fits.open(savepath, mode='update') as hdul:
        header = hdul[1].header

        for i, (col, v) in enumerate(cols.items()):
            comment = v[0]
            header[f'TCOMM{i + 1}'] = comment
            header.comments[f'TTYPE{i + 1}'] = comment

        hdul.flush()
    # Read Table to check
    with fits.open(savepath) as hdul:
        hdul.info()
        hdr = hdul[1].header
        print(repr(hdr))
    tab_exod_cat.read(savepath)
    tab_exod_cat.pprint(max_width=-1)


    #############################
    ### MAKE UNIQUE CATALOGUE ###
    #############################

    # Add SIMBAD Columns
    cols_simbad = ['main_id', 'main_type', 'ra', 'dec', 'SEP_ARCSEC']
    for col in cols_simbad:
        df_regions_unique[f'simbad_{col}'] = dfs_cmatch['SIMBAD'][col].values
    df_regions_unique['simbad_SEP_ARCSEC'] = df_regions_unique['simbad_SEP_ARCSEC'].replace(9999, np.nan)

    # Add GAIA Columns
    cols_gaia = ['DR3Name', 'RAdeg', 'DEdeg', 'SEP_ARCSEC', 'Gmag', 'BPmag', 'RPmag', 'Dist', 'Teff', 'VarFlag']
    for col in cols_gaia:
        df_regions_unique[f'gaia_{col}'] = dfs_cmatch['GAIA DR3'][col].values
    df_regions_unique['gaia_SEP_ARCSEC'] = df_regions_unique['gaia_SEP_ARCSEC'].replace(9999, np.nan)

    # Add OM Columns
    cols_om = ['XMMOM', 'ID', 'RAJ2000', 'DEJ2000', 'SEP_ARCSEC', 'Nobs', 'UVW2mAB', 'UVM2mAB', 'UVW1mAB', 'UmAB', 'BmAB', 'VmAB']
    for col in cols_om:
        df_regions_unique[f'SUSS6_{col}'] = dfs_cmatch['XMM OM'][col].values
    df_regions_unique['SUSS6_SEP_ARCSEC'] = df_regions_unique['SUSS6_SEP_ARCSEC'].replace(9999, np.nan)

    cols = {
        'region_num'        : ['Unique Region Number.', None],
        'EXOD_SRCID'        : ['Identifier for grouped detections within 20".', None],
        'ra_deg'            : ['RA of the detection in degrees (J2000)', 'deg'],
        'dec_deg'           : ['DEC of the detection in degrees (J2000)', 'deg'],
        'n_detections'      : ['Number of times the source was detected.', None],
        'simbad_main_id'    : ['SIMBAD Identifier', None],
        'simbad_main_type'  : ['SIMBAD OTYPE', None],
        'simbad_ra'         : ['SIMBAD RA (J2000)', 'deg'],
        'simbad_dec'        : ['SIMBAD Dec (J2000)', 'deg'],
        'simbad_SEP_ARCSEC' : ['SIMBAD Seperation', 'arcsec'],
        'gaia_DR3Name'      : ['GAIA DR3 ID', None],
        'gaia_RAdeg'        : ['GAIA RA (J2000)', 'deg'],
        'gaia_DEdeg'        : ['GAIA Dec (J2000)', 'deg'],
        'gaia_SEP_ARCSEC'   : ['GAIA Seperation', 'arcsec'],
        'gaia_Gmag'         : ['GAIA G Magnitude', 'mag'],
        'gaia_BPmag'        : ['GAIA BP Magnitude', 'mag'],
        'gaia_RPmag'        : ['GAIA RP Magnitude', 'mag'],
        'gaia_Dist'         : ['GAIA Distance', 'pc'],
        'gaia_Teff'         : ['GAIA Effective Temperature', 'K'],
        'gaia_VarFlag'      : ['GAIA Variablility Flag', None],
        'SUSS6_XMMOM'       : ['SUSS6 XMM OM Source name', None],
        'SUSS6_ID'          : ['SUSS6 Source number', None],
        'SUSS6_RAJ2000'     : ['SUSS6 RA (J2000)', 'deg'],
        'SUSS6_DEJ2000'     : ['SUSS6 Dec (J2000)', 'deg'],
        'SUSS6_SEP_ARCSEC'  : ['SUSS6 Seperation', 'arcsec'],
        'SUSS6_Nobs'        : ['SUSS6 Number of observation IDs', None],
        'SUSS6_UVW2mAB'     : ['SUSS6 Source AB magnitude UVW2', 'mag'],
        'SUSS6_UVM2mAB'     : ['SUSS6 Source AB magnitude UVM2', 'mag'],
        'SUSS6_UVW1mAB'     : ['SUSS6 Source AB magnitude UVW1', 'mag'],
        'SUSS6_UmAB'        : ['SUSS6 Source AB magnitude U', 'mag'],
        'SUSS6_BmAB'        : ['SUSS6 Source AB magnitude B', 'mag'],
        'SUSS6_VmAB'        : ['SUSS6 Source AB magnitude V', 'mag']
    }
    # Create Table
    tab_exod_cat_unique = Table.from_pandas(df_regions_unique[cols.keys()])

    # Add DR14 Columns
    cols_xmm_dr14 = ['IAUNAME', 'SRCID', 'SC_RA', 'SC_DEC', 'SC_POSERR', 'SC_EP_8_FLUX', 'SEP_ARCSEC', 'SC_VAR_FLAG']
    tab_xmm_dr14 = Table.from_pandas(dfs_cmatch['XMM DR14'][cols_xmm_dr14])

    # set entire astropy row that have SEP_ARCSEC > 20" to nan.
    mask = tab_xmm_dr14['SEP_ARCSEC'] > 20
    for col in cols_xmm_dr14:
        if col == 'SC_VAR_FLAG':
            tab_xmm_dr14[col][mask] = False
        elif col == 'SRCID':
            tab_xmm_dr14[col][mask] = 0
        else:
            tab_xmm_dr14[col][mask] = np.nan

    tab_exod_cat_unique = hstack([tab_exod_cat_unique, tab_xmm_dr14])
    

    new_column_names = [f"DR14_{col}" for col in cols_xmm_dr14]
    for old_name, new_name in zip(cols_xmm_dr14, new_column_names):
        tab_exod_cat_unique.rename_column(old_name, new_name)
    # Write Units
    for col, v in cols.items():
        unit = v[1]
        tab_exod_cat_unique[col].unit = unit
    # Save Table
    clobber = True
    savepath = cat_savepath / 'EXOD_DR1_cat_unique.fits'
    tab_exod_cat_unique.write(savepath, format='fits', overwrite=clobber)
    # Write descriptions to header
    with fits.open(savepath, mode='update') as hdul:
        header = hdul[1].header

        for i, (col, v) in enumerate(cols.items()):
            comment = v[0]
            header[f'TCOMM{i + 1}'] = comment
            header.comments[f'TTYPE{i + 1}'] = comment

        hdul.flush()
    # Read Table to check
    with fits.open(savepath) as hdul:
        hdul.info()
        hdr = hdul[1].header
        print(repr(hdr))
    tab_exod_cat_unique = Table.read(savepath)
    tab_exod_cat_unique.pprint(max_width=-1)
    return tab_exod_cat, tab_exod_cat_unique


if __name__ == "__main__":
    make_exod_catalogue()

