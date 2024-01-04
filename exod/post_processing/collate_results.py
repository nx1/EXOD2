import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad

from astroquery.exceptions import BlankResponseWarning
import warnings
warnings.simplefilter(action='ignore', category=BlankResponseWarning) # Ignore Non-results.
from exod.utils.path import data_results
from exod.utils.logger import logger
from exod.utils.simbad_classes import simbad_classifier


def combine_all_region_files():
    """
    Combine all df_region .csv files across all observations.

    Returns
    -------
    df_all_regions : pd.DataFrame : The combined dataframe of all the df_region files
    """
    logger.info('Combining all region files')
    csv_files = list(data_results.glob('*/*regions.csv'))
    logger.info(f'Found {len(csv_files)} csv files')

    dfs = []
    for csv in csv_files:
        df = pd.read_csv(csv, dtype={'obsid':str})
        logger.info(f'{csv} rows: {len(df)}')
        dfs.append(df)
    df_all_regions = pd.concat(dfs, axis=0).reset_index(drop=True)
    logger.info(f'df_all_regions:\n{df_all_regions}')
    return df_all_regions



def crossmatch_simbad(df_region, radius):
    """
    Crossmatch with a df_region table containing ['ra'] and ['dec']
    columns in degrees and 'icrs' coordinates with SIMBAD to get
    objects within a specified radius.

    This function queries SIMBAD with all the coordinates at once which saves time
    as you don't need to perform an individual query for each set of coordinates.

    Parameters
    ----------
    df_region : astropy.Table of pd.DataFrame containing 'ra' and 'dec' in degrees
    radius : astropy.units eg. 2*u.arcmin

    Returns
    -------
    tab_res : astropy.Table result from SIMBAD query.
    """
    logger.info(f'Crossmatching df_region len={len(df_region)} with SIMBAD, radius={radius}')

    logger.info('Creating SkyCoord Object')
    coords = SkyCoord(ra=df_region['ra'],
                      dec=df_region['dec'],
                      unit='deg',
                      frame='icrs')

    simbad = Simbad()
    # Additional fields can be checked with simbad.list_votable_fields()
    simbad.add_votable_fields('otype', 'distance')


    logger.info('Querying Region (This can take a while)')
    tab_res = simbad.query_region(coordinates=coords, radius=radius)
    tab_res['SCRIPT_NUMBER_ID'] = tab_res['SCRIPT_NUMBER_ID'] - 1  # Use 0 Indexing
    logger.info(f'Found {len(tab_res)} results')

    logger.info('Appending region coordinates to table...')
    tab_res['RA_REGION_DEG'] = [coords[i].ra for i in tab_res['SCRIPT_NUMBER_ID']]
    tab_res['DEC_REGION_DEG'] = [coords[i].dec for i in tab_res['SCRIPT_NUMBER_ID']]

    logger.info('Calculating seperations...')
    sc1 = SkyCoord(ra=tab_res['RA_REGION_DEG'], dec=tab_res['DEC_REGION_DEG'])
    sc2 = SkyCoord(ra=tab_res['RA'], dec=tab_res['DEC'], unit=(u.hourangle, u.deg))
    sep = sc1.separation(sc2)
    tab_res['SEP'] = sep
    return tab_res


def get_df_regions_no_crossmatch(df_regions, tab_res):
    """
    Get the rows in df_regions containing the regions that did not provide a crossmatch
    with SIMBAD.

    This function finds all the indexs in the df_regions dataframe that are not in the
    tab_res provided by astropy, it does this by looking at the dataframe index and the
    SCRIPT_NUMBER_ID (which has been adjusted to use 0 indexing).

    Parameters
    ----------
    df_regions : pd.DataFrame that was used to query SIMBAD.
    tab_res : astropy.Table that was returned by the SIMBAD query.

    Returns
    -------
    df_regions_no_crossmatch : pd.DataFrame All the regions that did not have a successful crossmatch.
    """
    l1 = df_regions.index
    l2 = tab_res['SCRIPT_NUMBER_ID']
    no_crossmatch_indexes = np.setdiff1d(l1, l2)
    logger.info(f'{len(no_crossmatch_indexes)} / {len(l1)} regions have no counterparts with SIMBAD')
    df_regions_no_crossmatch = df_regions.iloc[no_crossmatch_indexes]
    return df_regions_no_crossmatch


def get_df_regions_with_crossmatch(df_regions, tab_res):
    """
    Get the rows in df_regions containing the regions that did not provide a crossmatch
    with SIMBAD.

    This function finds all the indexs in the df_regions dataframe that are not in the
    tab_res provided by astropy, it does this by looking at the dataframe index and the
    SCRIPT_NUMBER_ID (which has been adjusted to use 0 indexing).

    Parameters
    ----------
    df_regions : pd.DataFrame that was used to query SIMBAD.
    tab_res : astropy.Table that was returned by the SIMBAD query.

    Returns
    -------
    df_regions_no_crossmatch : pd.DataFrame All the regions that did not have a successful crossmatch.
    """
    l1 = np.array(df_regions.index)
    l2 = np.array(tab_res['SCRIPT_NUMBER_ID'])
    common_idx = np.intersect1d(l1, l2)
    logger.info(f'{len(common_idx)} / {len(l1)} regions have counterparts with SIMBAD')
    df_regions_with_crossmatch = df_regions.iloc[common_idx]
    return df_regions_with_crossmatch

def classify_simbad_otype(tab_res):
    """
    Sub-classify the SIMBAD sources based on the OTYPE column.

    This was done in the original version of EXOD, however many of the SIMBAD OTYPES
    appear to be unaccounted for, additionally there seem to be some OTYPES that are
    not in the simbad_classifier dictionary which causes errors. My Guess is that
    we will do something more sophisticated than this, however it is a good start.

    Parameters
    ----------
    tab_res : astropy.Table with OTYPE column

    Returns
    -------
    tab_res : astropy.Table with the CLASSIFICATION column added.
    """
    classification = [simbad_classifier[t] for t in tab_res['OTYPE']]
    tab_res['CLASSIFICATION'] = classification
    return tab_res


if __name__ == '__main__':
    df_all_regions = combine_all_region_files()
    tab_res = crossmatch_simbad(df_all_regions, radius=1*u.arcmin)
    tab_res = classify_simbad_otype(tab_res=tab_res)
    df_all_regions_no_crossmatch = get_df_regions_no_crossmatch(df_regions=df_all_regions, tab_res=tab_res)
