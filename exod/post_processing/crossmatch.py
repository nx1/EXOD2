import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from exod.pre_processing.read_events import get_PN_image_file
from exod.utils.logger import logger
from exod.utils.path import data_util, data_results
from exod.utils.simbad_classes import simbad_classifier


def crossmatch_dr13_slim(df_regions):
    tab_xmm = Table.read(data_util / '4XMM_slim_DR13cat_v1.0.fits')
    skycoord_xmm = SkyCoord(ra=tab_xmm['SC_RA'], dec=tab_xmm['SC_DEC'], unit=u.deg)
    sky_coords = SkyCoord(ra=df_regions['ra_deg'].values, dec=df_regions['dec_deg'].values, unit='deg', frame='icrs')

    cmatch = sky_coords.match_to_catalog_sky(skycoord_xmm)
    tab_cmatch = Table(cmatch)
    tab_cmatch.rename_columns(names=tab_cmatch.colnames, new_names=['idx', 'sep2d', 'dist3d'])
    tab_cmatch['sep2d_arcsec'] = tab_cmatch['sep2d'].to(u.arcsec)
    tab_cmatch['idx_orig'] = np.arange(len(tab_cmatch))

    tab_xmm_cmatch = tab_xmm[tab_cmatch['idx']]
    tab_xmm_cmatch['SEP'] = tab_cmatch['sep2d']
    tab_xmm_cmatch['SEP_ARCSEC'] = tab_xmm_cmatch['SEP'] * 3600
    return tab_xmm_cmatch


def crossmatch_tranin_dr12(df_regions):
    tab_xmm = Table.read(data_util / 'tranin/classification_DR12_with_input.fits')
    skycoord_xmm = SkyCoord(ra=tab_xmm['RA'], dec=tab_xmm['DEC'], unit=u.deg)
    sky_coords = SkyCoord(ra=df_regions['ra_deg'].values, dec=df_regions['dec_deg'].values, unit='deg', frame='icrs')

    cmatch = sky_coords.match_to_catalog_sky(skycoord_xmm)
    tab_cmatch = Table(cmatch)
    tab_cmatch.rename_columns(names=tab_cmatch.colnames, new_names=['idx', 'sep2d', 'dist3d'])
    tab_cmatch['sep2d_arcsec'] = tab_cmatch['sep2d'].to(u.arcsec)
    tab_cmatch['idx_orig'] = np.arange(len(tab_cmatch))

    tab_xmm_cmatch = tab_xmm[tab_cmatch['idx']]
    tab_xmm_cmatch['SEP'] = tab_cmatch['sep2d']
    tab_xmm_cmatch['SEP_ARCSEC'] = tab_xmm_cmatch['SEP'] * 3600
    return tab_xmm_cmatch



def crossmatch_simbad(df_region, radius):
    """
    Crossmatch with a df_region table containing ['ra'] and ['dec']
    columns in degrees and 'icrs' coordinates with SIMBAD to get
    objects within a specified radius.

    This function queries SIMBAD with all the coordinates at once which saves time
    as you don't need to perform an individual query for each set of coordinates.

    Parameters
    ----------
    df_region : astropy.Table or pd.DataFrame containing 'ra' and 'dec' in degrees
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
    tab_res['SEP_DEG'] = sep
    tab_res['SEP_ARCMIN'] = tab_res['SEP_DEG'].to(u.arcmin)
    tab_res['SEP_ARCSEC'] = tab_res['SEP_DEG'].to(u.arcsec)
    return tab_res


def crossmatch_vizier(df_region, radius, catalog):
    """
    Crossmatch Vizier catalogue.

    Parameters
    ----------
    df_region : astropy.Table or pd.DataFrame containing 'ra' and 'dec' in degrees
    radius : astropy.units eg. 2*u.arcmin
    catalog : Vizier Catalogue to query.

    Returns
    -------
    tab_res : astropy.table.Table
    """
    v = Vizier
    coords = SkyCoord(ra=df_region['ra'],
                      dec=df_region['dec'],
                      unit='deg',
                      frame='icrs')
    tab_list = v.query_region(coords, radius=radius, frame='icrs', catalog=catalog)
    tab_res = tab_list[0]
    tab_res['_q'] = tab_res['_q'] - 1
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
    col = _get_table_id_col(tab_res)
    l1 = df_regions.index
    l2 = tab_res[col]
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
    SCRIPT_NUMBER_ID or _q (which has been adjusted to use 0 indexing).

    Parameters
    ----------
    df_regions : pd.DataFrame that was used to query SIMBAD.
    tab_res : astropy.Table that was returned by the SIMBAD query.

    Returns
    -------
    df_regions_no_crossmatch : pd.DataFrame All the regions that did not have a successful crossmatch.
    """
    col = _get_table_id_col(tab_res)
    l1 = np.array(df_regions.index)
    l2 = np.array(tab_res[col])
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


def plot_simbad_crossmatch_image(obsid,
                                 df_all_regions_no_crossmatch,
                                 df_all_regions_with_crossmatch,
                                 tab_res):
    img_file = get_PN_image_file(obsid=obsid)

    hdul = fits.open(img_file)
    header = hdul[0].header
    img_data = hdul[0].data
    wcs = WCS(header=header)

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': wcs}, facecolor='grey')
    cmap = plt.cm.hot
    cmap.set_bad('black')

    m1 = ax.imshow(img_data,
                   norm=LogNorm(),
                   interpolation='none',
                   origin='lower',
                   cmap=cmap)

    # Plot Detections without a crossmatch
    df_all_regions_no_crossmatch_obsid = df_all_regions_no_crossmatch[df_all_regions_no_crossmatch['obsid'] == obsid]
    if len(df_all_regions_no_crossmatch_obsid) > 0:
        ra = df_all_regions_no_crossmatch_obsid['ra']
        dec = df_all_regions_no_crossmatch_obsid['dec']
        ax.scatter(ra, dec, transform=ax.get_transform('world'), color='cyan', marker='o', label='No Crossmatch',
                   alpha=0.5)

    # Plot Detections with Crossmatch
    df_all_regions_with_crossmatch_obsid = df_all_regions_with_crossmatch[
        df_all_regions_with_crossmatch['obsid'] == obsid]
    if len(df_all_regions_with_crossmatch_obsid) > 0:
        ra = df_all_regions_with_crossmatch_obsid['ra']
        dec = df_all_regions_with_crossmatch_obsid['dec']
        ax.scatter(ra, dec, transform=ax.get_transform('world'), color='yellow', marker='o', label='With Crossmatch',
                   alpha=0.5)

    # Get Crossmatched Sources for specific observation
    l1 = np.array(tab_res['SCRIPT_NUMBER_ID'])
    l2 = np.array(df_all_regions_with_crossmatch_obsid.index)
    common_idx = np.intersect1d(l1, l2)
    tab_res_with_crossmatch = tab_res[np.isin(tab_res['SCRIPT_NUMBER_ID'], common_idx)]

    # Plot Crossmatched Sources
    if len(tab_res_with_crossmatch) > 0:
        ra = tab_res_with_crossmatch['RA']
        dec = tab_res_with_crossmatch['DEC']
        coords = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
        ax.scatter(coords.ra.deg, coords.dec.deg, transform=ax.get_transform('world'), color='white', marker='+',
                   label='SIMBAD Sources', alpha=1.0)

        # Add text labels for each source
        for i, source in enumerate(tab_res_with_crossmatch):
            source_ra = coords[i].ra.deg
            source_dec = coords[i].dec.deg
            source_name = source['MAIN_ID']
            otype = source['OTYPE']
            sep_arcsec = source['SEP_ARCSEC']
            txt = f'{source_name} | {otype} | {sep_arcsec:.2f}'

            ax.text(source_ra, source_dec, txt, color='white', ha='left', va='bottom',
                    transform=ax.get_transform('world'))

    # Set the x-axis and y-axis limits to exclude surrounding 0 values
    ax.set_xlim(np.min(np.nonzero(img_data)[1]), np.max(np.nonzero(img_data)[1]))
    ax.set_ylim(np.min(np.nonzero(img_data)[0]), np.max(np.nonzero(img_data)[0]))
    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')
    ax.set_title(f'SIMBAD Crossmatch Plot | obsid={obsid}')
    ax.legend()

    savepath = data_results / f'{obsid}' / 'SIMBAD_crossmatch_plot.png'
    logger.info(f'Saving figure to: {savepath}')
    plt.savefig(savepath)
    plt.show()


def _get_table_id_col(tab_res):
    """Get the correct ID column to use for crossmatching from the astropy table."""
    logger.info('Getting table ID column')
    if '_q' in tab_res.columns:
        return '_q'
    elif 'SCRIPT_NUMBER_ID' in tab_res.columns:
        return 'SCRIPT_NUMBER_ID'
    else:
        raise KeyError('_q or SCRIPT_NUMBER_ID not found in table columns!')
