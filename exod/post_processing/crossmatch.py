"""
This module contains code for crossmatching the regions with various catalogues.
"""
import io
import time
import warnings
from pathlib import Path

import numpy as np
import requests
from astropy import units as u
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astroquery.simbad.core import BlankResponseWarning
import matplotlib.pyplot as plt
from astroquery.xmatch import XMatch
from matplotlib.colors import LogNorm
from tqdm import tqdm

from exod.utils.logger import logger
from exod.utils.path import data_util, data_results, data_plots, data_combined, savepaths_combined
from exod.utils.simbad_classes import simbad_classifier
from exod.post_processing.cluster_regions import get_unique_regions

warnings.filterwarnings("ignore", category=BlankResponseWarning)

def crossmatch_fits_table(fits_path, df_region, ra_col, dec_col):
    """
    Crossmatch with an arbitrary Fits Table.

    Parameters:
        fits_path (Path): Path to the FITS file.
        df_region (pd.DataFrame): DataFrame containing the regions to crossmatch.
        ra_col (str): Column name for the RA values in FITS file.
        dec_col (str): Column name for the DEC values in FITS file.

    Returns:
        tab_fits_cmatch (astropy.Table): Table containing the crossmatched data.
    """
    tab_fits = Table.read(fits_path)

    skycoord_xmm = SkyCoord(ra=tab_fits[ra_col], dec=tab_fits[dec_col], unit=u.deg, frame='fk5', equinox='J2000')
    skycoord_reg = SkyCoord(ra=df_region['ra_deg'].values, dec=df_region['dec_deg'].values, unit='deg', frame='fk5', equinox='J2000')

    cmatch = skycoord_reg.match_to_catalog_sky(skycoord_xmm)

    tab_cmatch = Table(cmatch)
    tab_cmatch.rename_columns(names=tab_cmatch.colnames, new_names=['key', 'sep2d', 'dist3d'])
    tab_cmatch['sep2d_arcsec'] = tab_cmatch['sep2d'].to(u.arcsec)
    tab_cmatch['idx_orig'] = np.arange(len(tab_cmatch))

    tab_fits_cmatch = tab_fits[tab_cmatch['key']]
    tab_fits_cmatch['SEP_ARCSEC'] = tab_cmatch['sep2d_arcsec']

    matched_reg = df_region.iloc[tab_cmatch['idx_orig']]
    ra_offset   = (matched_reg['ra_deg'] - tab_fits_cmatch[ra_col]) * np.cos(np.radians(matched_reg['dec_deg']))
    dec_offset  = matched_reg['dec_deg'] - tab_fits_cmatch[dec_col]
    tab_fits_cmatch['RA_OFFSET']  = ra_offset
    tab_fits_cmatch['DEC_OFFSET'] = dec_offset
    tab_fits_cmatch['IDX_ORIGINAL'] = tab_cmatch['idx_orig']
    return tab_fits_cmatch


def crossmatch_dr14_slim(df_region):
    """
    Crossmatch the regions with the 4XMM DR13 slim catalogue.
    """
    logger.info('Crossmatching with 4XMM DR14 slim catalogue')
    fits_path = data_util / '4xmmdr14slim_240411.fits'
    tab_xmm_cmatch = crossmatch_fits_table(fits_path, df_region, ra_col='SC_RA', dec_col='SC_DEC')
    return tab_xmm_cmatch


def crossmatch_dr13_slim(df_region):
    """
    Crossmatch the regions with the 4XMM DR13 slim catalogue.
    """
    logger.info('Crossmatching with 4XMM DR13 slim catalogue')
    fits_path = data_util / '4XMM_slim_DR13cat_v1.0.fits'
    tab_xmm_cmatch = crossmatch_fits_table(fits_path, df_region, ra_col='SC_RA', dec_col='SC_DEC')
    return tab_xmm_cmatch


def crossmatch_tranin_dr12(df_region):
    """
    Crossmatch the regions with the CLAXON Hugo Tranin DR12 catalogue.
    """
    logger.info('Crossmatching with CLAXON Hugo Tranin DR12 catalogue')
    fits_path = data_util / 'tranin/classification_DR12_with_input.fits'
    tab_xmm_cmatch = crossmatch_fits_table(fits_path, df_region, ra_col='RA', dec_col='DEC')
    return tab_xmm_cmatch


def crossmatch_cds(df, max_sep_arcsec=20, catalogue='simbad', ra_col='ra_deg', dec_col='dec_deg'):
    """Crossmatch using CDS X-Match Service."""
    print(f'Crossmatching {len(df)} rows with {catalogue} using CDS Xmatch max_sep={max_sep_arcsec}"')
    r = requests.post(
        url='http://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync',
        data={'request': 'xmatch',
              'distMaxArcsec': max_sep_arcsec,
              'RESPONSEFORMAT': 'csv',
              'cat2': catalogue,
              'colRA1': ra_col,
              'colDec1': dec_col},
        files={'cat1': df.to_csv()})
    df_res = pd.read_csv(io.StringIO(r.text), low_memory=False)
    print(f'Returned {len(df_res)} rows')
    return df_res


def calc_sep(df_res, ra_col1, dec_col1, ra_col2, dec_col2):
    print('Calculating Seperations...')
    sc1 = SkyCoord(ra=df_res[ra_col1], dec=df_res[dec_col1], unit='deg')
    sc2 = SkyCoord(ra=df_res[ra_col2], dec=df_res[dec_col2], unit='deg')
    sep = sc1.separation(sc2).to('arcsec')
    sep = [s.value for s in sep]
    return sep


def xmatch_cds(tab1, catalogue, max_sep, ra_col='ra_deg', dec_col='dec_deg', **kwargs):
    table = XMatch.query(cat1=tab1, cat2=catalogue, max_distance=max_sep, colRA1=ra_col, colDec1=dec_col, **kwargs)
    return table


def keep_closest_match(df):
    print('Keeping Closest Match...')
    print(f'Length pre  = {len(df)}')
    idx = df.groupby('cluster_label')['SEP_ARCSEC'].idxmin()
    df = df.loc[idx]
    print(f'Length post = {len(df)}')
    return df


def add_no_match_sources(df1, df2):
    print('Adding in Sources with No Matches...')
    cluster_labels_no_match = np.setdiff1d(df1['cluster_label'], df2['cluster_label'])
    df_no_match = df1.set_index('cluster_label').loc[cluster_labels_no_match].reset_index()
    df_no_match['SEP_ARCSEC'] = 9999
    df_res = pd.concat([df2, df_no_match], axis=0).sort_values('cluster_label')
    return df_res


def xmatch(df, catalogue='simbad', max_sep_arcsec=20, ra_col='ra', dec_col='dec'):
    df_res = crossmatch_cds(df, max_sep_arcsec=max_sep_arcsec, catalogue=catalogue, ra_col='ra_deg', dec_col='dec_deg')
    df_res['SEP_ARCSEC'] = calc_sep(df_res, ra_col1='ra_deg', dec_col1='dec_deg', ra_col2=ra_col, dec_col2=dec_col)
    df_res = keep_closest_match(df_res)
    df_res = add_no_match_sources(df, df_res)
    assert len(df_res) == len(df)
    return df_res


def crossmatch_unique_regions(df_regions_unique, max_sep_arcsec=20, clobber=True):
    savepaths_cmatch = {'SIMBAD'  : data_combined / 'cmatch_simbad.csv',
                        'GAIA DR3': data_combined / 'cmatch_gaia_dr3.csv',
                        'XMM OM'  : data_combined / 'cmatch_xmm_om.csv'}
    dfs_cmatch = {}
    dfs_cmatch['XMM DR14'] = crossmatch_dr14_slim(df_regions_unique)

    if not clobber:
        if all([savepath.exists() for savepath in savepaths_cmatch.values()]):
            logger.info('Crossmatch files already exist and clobber=False, loading from files')
            for k, savepath in savepaths_cmatch.items():
                logger.info(f'Loading {k} crossmatch from {savepath}')
                dfs_cmatch[k] = pd.read_csv(savepath)
            return dfs_cmatch
        else:
            logger.info('Some crossmatch files are missing. Recreating...')


    catalogs = {'SIMBAD'  : 'simbad',
                'GAIA DR3': 'vizier:I/355/gaiadr3',
                'XMM OM'  : 'vizier:II/378/xmmom6s'}
    # 'GLADE'    : 'vizier:VII/291/gladep'}
    # 'CHIME FRB': 'vizier:J/ApJS/257/59/table2',}

    catalogs_coord_cols = {'SIMBAD'   : {'ra': 'ra', 'dec': 'dec'},
                           'GAIA DR3' : {'ra': 'RAJ2000', 'dec': 'DEJ2000'},
                           'XMM OM'   : {'ra': 'RAJ2000', 'dec': 'DEJ2000'},
                           'GLADE'    : {'ra': 'RAJ2000', 'dec': 'DEJ2000'},
                           'CHIME FRB': {'ra': 'RAJ2000', 'dec': 'DEJ2000'}}


    for k, cat in catalogs.items():
        dfs_cmatch[k] = xmatch(df_regions_unique, cat, max_sep_arcsec=max_sep_arcsec,
                               ra_col=catalogs_coord_cols[k]['ra'], dec_col=catalogs_coord_cols[k]['dec'])

        print(f'Saving {k} crossmatch to {savepaths_cmatch[k]}')
        dfs_cmatch[k].to_csv(savepaths_cmatch[k], index=False)

    return dfs_cmatch








if __name__ == "__main__":
    crossmatch_unique_regions()

    # df_region = pd.read_csv(data_combined / 'merged_with_dr14' / 'df_regions.csv')
    # df_region = df_region[df_region['runid'].str.contains('50_0.2_12.0')]
    # df_region = df_region.iloc[100:200]
    # # df_region = df_region.sample(1000)
    # crossmatch = CrossMatch(df_region)
    # crossmatch.run()
    # cmatch_info = crossmatch.info
    # crossmatch.plot_pie_chart()
    # crossmatch.plot_seperations()
    # plt.show()
