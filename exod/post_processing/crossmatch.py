"""
This module contains code for crossmatching the regions with various catalogues.
"""
import time
import warnings
from pathlib import Path

import numpy as np
from astropy import units as u
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astroquery.simbad.core import BlankResponseWarning
import matplotlib.pyplot as plt
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
    ra_offset   = matched_reg['ra_deg'] - tab_fits_cmatch[ra_col]
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


def crossmatch_simbad(df_region, radius):
    """
    Crossmatch with a df_region table containing ['ra'] and ['dec']
    columns in degrees and 'icrs' coordinates with SIMBAD to get
    objects within a specified radius.

    This function queries SIMBAD with all the coordinates at once which saves time
    as you don't need to perform an individual query for each set of coordinates.

    Parameters:
        df_region (pd.DataFrame): containing the regions to crossmatch. 'ra' and 'dec' in degrees.
        radius (astropy.units): Radius to search around the coordinates.

    Returns:
        tab_res (astropy.Table): Result from SIMBAD query.
    """
    n_reg = len(df_region)
    logger.info(f'Crossmatching df_region n_reg={n_reg} with SIMBAD, radius={radius}')
    skycoord_reg = SkyCoord(ra=df_region['ra_deg'].values, dec=df_region['dec_deg'].values, unit='deg', frame='fk5', equinox='J2000')

    simbad = Simbad()
    simbad.TIMEOUT = 1000
    # Additional fields can be checked with simbad.list_votable_fields()
    simbad.add_votable_fields('otype', 'distance')

    logger.info('Querying Region (This can take a while...)')
    tab_res = simbad.query_region(coordinates=skycoord_reg, radius=radius)

    if not tab_res:
        logger.info('No Results Found! Just returning error table.')
        err_idx = np.arange(0, n_reg, 1)
        err_sep = [9999 * u.arcsec] * len(err_idx)
        ra_reg  = [skycoord_reg[i].ra for i in err_idx]
        dec_reg = [skycoord_reg[i].dec for i in err_idx]
        tab_err = Table({'SCRIPT_NUMBER_ID' : err_idx,
                         'RA_REGION_DEG'    : ra_reg,
                         'DEC_REGION_DEG'   : dec_reg,
                         'SEP_ARCSEC'       : err_sep})
        return tab_err
    tab_res['SCRIPT_NUMBER_ID'] = tab_res['SCRIPT_NUMBER_ID'] - 1  # Use 0 Indexing
    logger.info(f'Found {len(tab_res)} results')

    logger.info('Appending region coordinates to table...')
    tab_res['RA_REGION_DEG'] = [skycoord_reg[i].ra for i in tab_res['SCRIPT_NUMBER_ID']]
    tab_res['DEC_REGION_DEG'] = [skycoord_reg[i].dec for i in tab_res['SCRIPT_NUMBER_ID']]

    logger.info('Calculating separations...')
    sc1 = SkyCoord(ra=tab_res['RA_REGION_DEG'], dec=tab_res['DEC_REGION_DEG'])
    sc2 = SkyCoord(ra=tab_res['RA'], dec=tab_res['DEC'], unit=(u.hourangle, u.deg))
    sep = sc1.separation(sc2).to(u.arcsec)
    tab_res['SEP_ARCSEC'] = sep

    logger.info('Keeping Only closest match for each region')
    rows = []
    for tab in tqdm(tab_res.group_by('SCRIPT_NUMBER_ID').groups):
        min_idx = tab['SEP_ARCSEC'].argmin()
        row = tab[min_idx]
        rows.append(row)
    tab_res_closest = vstack(rows)

    logger.info('Appending regions with no match to table...')
    err_idx = np.setdiff1d(np.arange(0, n_reg, 1), tab_res_closest['SCRIPT_NUMBER_ID'])
    err_sep = [9999 * u.arcsec] * len(err_idx)
    ra_reg  = [skycoord_reg[i].ra for i in err_idx]
    dec_reg = [skycoord_reg[i].dec for i in err_idx]
    tab_err = Table({'SCRIPT_NUMBER_ID' : err_idx,
                     'RA_REGION_DEG'    : ra_reg,
                     'DEC_REGION_DEG'   : dec_reg,
                     'SEP_ARCSEC'       : err_sep})
    tab_res_closest = vstack([tab_res_closest, tab_err])
    tab_res_closest.sort('SCRIPT_NUMBER_ID')
    return tab_res_closest


def crossmatch_chunk(query_func, df_region, savepath, radius, chunk_size=100, clobber=False):
    """
    Crossmatch in chunks to avoid timeouts.

    Parameters:
        query_func (function): Function to query the catalogue.
        df_region (pd.DataFrame): containing the regions to crossmatch. 'ra' and 'dec' in degrees.
        radius (astropy.units): Radius to search around the coordinates. eg. 2*u.arcmin.
        chunk_size (int): Number of regions to query at once.
        clobber (bool): Overwrite existing files.

    Returns:
        tab_cmatch (astropy.Table): Result from the query.
    """
    savepath = Path(savepath)
    if not clobber and savepath.exists():
        logger.info(f'{savepath} exists and clobber=False | loading table...')
        tab_cmatch = Table.read(savepath, format='csv')
        return tab_cmatch

    start_time = time.time()
    n_rows = len(df_region)
    all_tabs = []

    for i in range(0, n_rows, chunk_size):
        logger.info(f'{i} / {n_rows}')
        chunk_start_time = time.time()

        start = i
        end = min(i + chunk_size, n_rows)
        df_sub = df_region.iloc[start:end]

        tab = query_func(df_region=df_sub, radius=radius)
        idxs = np.arange(start, end, 1)
        tab['SCRIPT_NUMBER_ID'] = idxs
        all_tabs.append(tab)

        chunk_elapsed_time = time.time() - chunk_start_time
        total_elapsed_time = time.time() - start_time
        estimated_total_time = (total_elapsed_time / end) * n_rows
        estimated_remaining_time = estimated_total_time - total_elapsed_time
        logger.info(f'Time | elapsed: {chunk_elapsed_time:.2f} remaining: {estimated_remaining_time:.2f} total={total_elapsed_time:.2f}')    
    tab_cmatch = vstack(all_tabs)
    tab_cmatch.write(savepath, format='csv', overwrite=True)
    return tab_cmatch


def crossmatch_vizier(catalog, df_region, radius):
    """
    Crossmatch with a Vizier catalogue.

    Parameters:
        catalog (str): Name of the Vizier catalogue to query.
        df_region (pd.DataFrame): containing the regions to crossmatch. 'ra' and 'dec' in degrees.
        radius (astropy.units): Radius to search around the coordinates. eg. 2*u.arcmin.

    Returns:
        tab_res (astropy.Table) result from Vizier query.
        skycoord_reg (astropy.coordinates.SkyCoord): SkyCoord object of the region coordinates.
    """
    logger.info(f'Crossmatching df_region len={len(df_region)} with Vizier, radius={radius} catalog={catalog}')
    skycoord_reg = SkyCoord(ra=df_region['ra_deg'].values, dec=df_region['dec_deg'].values, unit='deg', frame='fk5', equinox='J2000')
    v = Vizier()
    v.TIMEOUT = 1000
    tab_list = v.query_region(skycoord_reg, radius=radius, frame='icrs', catalog=catalog)
    tab_res = tab_list[0]
    tab_res['_q'] = tab_res['_q'] - 1
    return tab_res, skycoord_reg


def crossmatch_gaia(df_region, radius):
    """Crossmatch with GAIA DR3 Catalogue."""
    catalog = 'I/355/gaiadr3'
    n_reg = len(df_region)
    tab_res, skycoord_reg = crossmatch_vizier(catalog, df_region, radius)

    # Append Separation To table
    coords1 = SkyCoord([skycoord_reg[i] for i in tab_res['_q']])
    coords2 = SkyCoord(ra=tab_res['RA_ICRS'], dec=tab_res['DE_ICRS'], unit='deg', frame='icrs')
    sep = coords1.separation(coords2).to(u.arcsec)
    tab_res['SEP_ARCSEC'] = sep
    tab_res['RA_REGION_DEG'] = [skycoord_reg[i].ra for i in tab_res['_q']]
    tab_res['DEC_REGION_DEG'] = [skycoord_reg[i].dec for i in tab_res['_q']]

    # Only Keep the closest Match for each _q
    rows = []
    for tab in tqdm(tab_res.group_by('_q').groups):
        min_idx = tab['SEP_ARCSEC'].argmin()
        row = tab[min_idx]
        rows.append(row)
    tab_res_closest = vstack(rows)
    
    logger.info('Appending regions with no match to table...')
    err_idx = np.setdiff1d(np.arange(0, n_reg, 1), tab_res_closest['_q'])
    err_sep = [9999 * u.arcsec] * len(err_idx)
    ra_reg  = [skycoord_reg[i].ra for i in err_idx]
    dec_reg = [skycoord_reg[i].dec for i in err_idx]
    tab_err = Table({'_q' : err_idx,
                     'RA_REGION_DEG'    : ra_reg,
                     'DEC_REGION_DEG'   : dec_reg,
                     'SEP_ARCSEC'       : err_sep})
    tab_res_closest = vstack([tab_res_closest, tab_err])
    tab_res_closest.sort('_q')
    return tab_res_closest


def crossmatch_xmm_om(df_region, radius):
    """Crossmatch with XMM Optical Monitor Catalogue v6 (Page+ 2023) (Thanks Matt!)."""
    catalog = '	II/378/xmmom6s'
    n_reg = len(df_region)
    tab_res, skycoord_reg = crossmatch_vizier(catalog, df_region, radius)

    # Append Separation To table
    coords1 = SkyCoord([skycoord_reg[i] for i in tab_res['_q']])
    coords2 = SkyCoord(ra=tab_res['RAJ2000'], dec=tab_res['DEJ2000'], unit='deg', frame='fk5', equinox='J2000')
    sep = coords1.separation(coords2).to(u.arcsec)
    tab_res['SEP_ARCSEC'] = sep
    tab_res['RA_REGION_DEG'] = [skycoord_reg[i].ra for i in tab_res['_q']]
    tab_res['DEC_REGION_DEG'] = [skycoord_reg[i].dec for i in tab_res['_q']]

    # Only Keep the closest Match for each _q
    rows = []
    for tab in tqdm(tab_res.group_by('_q').groups):
        min_idx = tab['SEP_ARCSEC'].argmin()
        row = tab[min_idx]
        rows.append(row)
    tab_res_closest = vstack(rows)

    logger.info('Appending regions with no match to table...')
    err_idx = np.setdiff1d(np.arange(0, n_reg, 1), tab_res_closest['_q'])
    err_sep = [9999 * u.arcsec] * len(err_idx)
    ra_reg  = [skycoord_reg[i].ra for i in err_idx]
    dec_reg = [skycoord_reg[i].dec for i in err_idx]
    tab_err = Table({'_q'             : err_idx,
                     'RA_REGION_DEG'  : ra_reg,
                     'DEC_REGION_DEG' : dec_reg,
                     'SEP_ARCSEC'     : err_sep})
    tab_res_closest = vstack([tab_res_closest, tab_err])
    tab_res_closest.sort('_q')
    return tab_res_closest


def classify_simbad_otype(tab_res):
    """
    Sub-classify the SIMBAD sources based on the OTYPE column.

    This was done in the original version of EXOD, however many of the SIMBAD OTYPES
    appear to be unaccounted for, additionally there seem to be some OTYPES that are
    not in the simbad_classifier dictionary which causes errors. My Guess is that
    we will do something more sophisticated than this, however it is a good start.

    Parameters:
        tab_res (astropy.Table): Table containing the SIMBAD results with an OTYPE column.

    Returns:
        tab_res (astropy.Table): Table with the CLASSIFICATION column added.
    """
    classification = [simbad_classifier[t] for t in tab_res['OTYPE']]
    tab_res['CLASSIFICATION'] = classification
    return tab_res

def crossmatch_unique_regions():
    clobber            = False
    clustering_radius  = 20*u.arcsec # Clustering radius for unique regions
    cmatch_max_radius  = 30*u.arcsec # Max search radius for external crossmatching
    cmatch_chunk_size  = 100         # Chunksize for external crossmatching

    df_regions        = pd.read_csv(savepaths_combined['regions'])
    df_regions_unique = get_unique_regions(df_regions, clustering_radius=clustering_radius)
    tab_cmatch_xmm    = crossmatch_dr14_slim(df_regions_unique)

    tab_cmatch_simbad = crossmatch_chunk(query_func=crossmatch_simbad, df_region=df_regions_unique,
                                         savepath=(savepaths_combined['cmatch_simbad']), radius=cmatch_max_radius,
                                         chunk_size=cmatch_chunk_size, clobber=clobber)

    tab_cmatch_gaia = crossmatch_chunk(query_func=crossmatch_gaia, df_region=df_regions_unique,
                                       savepath=(savepaths_combined['cmatch_gaia']), radius=cmatch_max_radius,
                                       chunk_size=cmatch_chunk_size, clobber=clobber)

    tab_cmatch_om = crossmatch_chunk(query_func=crossmatch_xmm_om, df_region=df_regions_unique,
                                     savepath=(savepaths_combined['cmatch_om']), radius=cmatch_max_radius,
                                     chunk_size=cmatch_chunk_size, clobber=clobber)

    tables = {'tab_cmatch_xmm'    : tab_cmatch_xmm,
              'tab_cmatch_simbad' : tab_cmatch_simbad,
              'tab_cmatch_gaia'   : tab_cmatch_gaia,
              'tab_cmatch_om'     : tab_cmatch_om}
    return tables



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
