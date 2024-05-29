"""
This module contains code for crossmatching the regions with various catalogues.
"""
import time
import warnings

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astroquery.simbad.core import BlankResponseWarning
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm

from exod.utils.logger import logger
from exod.utils.path import data_util, data_results, data_plots
from exod.utils.plotting import cmap_image
from exod.utils.simbad_classes import simbad_classifier
from exod.xmm.observation import Observation
import exod.post_processing.crossmatch_simulation as crossmatch_simulation
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
    tab_cmatch.rename_columns(names=tab_cmatch.colnames, new_names=['idx', 'sep2d', 'dist3d'])
    tab_cmatch['sep2d_arcsec'] = tab_cmatch['sep2d'].to(u.arcsec)
    tab_cmatch['idx_orig'] = np.arange(len(tab_cmatch))

    tab_fits_cmatch = tab_fits[tab_cmatch['idx']]
    tab_fits_cmatch['SEP_ARCSEC'] = tab_cmatch['sep2d_arcsec']
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

def crossmatch_simbad_chunk(df_region, radius=5 * u.arcsec, chunk_size=1000):
    """Query Simbad in chunks to avoid timeouts."""
    start_time = time.time()
    n_rows = len(df_region)
    all_tabs = []

    for i in range(0, n_rows, chunk_size):
        logger.info(f'{i} / {n_rows}')
        chunk_start_time = time.time()

        start = i
        end = min(i + chunk_size, n_rows)
        df_sub = df_region.iloc[start:end]

        tab = crossmatch_simbad(df_region=df_sub, radius=radius)
        idxs = np.arange(start, end, 1)
        tab['SCRIPT_NUMBER_ID'] = idxs
        all_tabs.append(tab)

        chunk_elapsed_time = time.time() - chunk_start_time
        total_elapsed_time = time.time() - start_time
        estimated_total_time = (total_elapsed_time / end) * n_rows
        estimated_remaining_time = estimated_total_time - total_elapsed_time
        logger.info(f'Time | elapsed: {chunk_elapsed_time:.2f} remaining: {estimated_remaining_time:.2f} total={total_elapsed_time:.2f}')    
    tab_res = vstack(all_tabs)
    return tab_res

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
    logger.info(tab_res)
    # Only Keep the closest Match for each _q
    rows = []
    for tab in tqdm(tab_res.group_by('_q').groups):
        min_idx = tab['SEP_ARCSEC'].argmin()
        row = tab[min_idx]
        rows.append(row)
    tab_res_closest = vstack(rows)
    logger.info(tab_res_closest)
    
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

def crossmatch_regions_subsets():
    """
    Crossmatch regions between the different simulation subsets.

    """
    return ''


def plot_simbad_crossmatch_image(obsid, df_all_regions_no_crossmatch, df_all_regions_with_crossmatch, tab_res):
    """
    Plot the SIMBAD crossmatch results on the image for a specific observation.

    Parameters:
        obsid (str): Observation ID to plot.
        df_all_regions_no_crossmatch (pd.DataFrame): DataFrame containing all regions with no crossmatch.
        df_all_regions_with_crossmatch (pd.DataFrame): DataFrame containing all regions with a crossmatch.
        tab_res (astropy.Table): Table containing the SIMBAD results.
    """
    observation = Observation(obsid)
    observation.get_files()
    img = observation.images[0]
    img.read()
    img_data = img.data
    wcs = img.wcs

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': wcs}, facecolor='grey')
    m1 = ax.imshow(img_data,
                   norm=LogNorm(),
                   interpolation='none',
                   origin='lower',
                   cmap=cmap_image)

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
    # plt.show()


class CrossMatch:
    def __init__(self, df_region):
        self.df_region = df_region
        self.skycoord_regions = SkyCoord(ra=df_region['ra_deg'].values, dec=df_region['dec_deg'].values,
                                         unit='deg', frame='fk5', equinox='J2000')
        self.radius_simbad = 28*u.arcsec
        self.radius_gaia   = 28*u.arcsec
        self.radius_xmm_om = 28*u.arcsec
        self.radius_self_cmatch = 0.25*u.arcsec # Radius for clustering sources across simulation subsets
        self.max_sep = 15*u.arcsec # Maximum separation for crossmatch

    def run(self):
        self.crossmatch_dr14_slim()
        self.crossmatch_tranin_dr12()
        self.crossmatch_simbad()
        self.crossmatch_gaia()
        self.crossmatch_om()
        self.crossmatch_unique_regions()

        crossmatch_simulation.get_unique_sources(df_regions=self.df_region, clustering_radius=self.radius_self_cmatch)
        crossmatch_simulation.main()

    def split_by_max_seperation(self, tab, colname='SEP_ARCSEC'):
        mask = tab[colname] < self.max_sep
        tab_cmatch = tab[mask]
        tab_no_cmatch = tab[~mask]
        return tab_cmatch, tab_no_cmatch
    def crossmatch_dr14_slim(self):
        self.tab_dr14 = crossmatch_dr14_slim(self.df_region)
        self.tab_dr14_cmatch, self.tab_dr14_no_cmatch = self.split_by_max_seperation(self.tab_dr14)

    def crossmatch_tranin_dr12(self):
        self.tab_dr12 = crossmatch_tranin_dr12(self.df_region)
        self.tab_dr12_cmatch, self.tab_dr12_no_cmatch = self.split_by_max_seperation(self.tab_dr12)

    def crossmatch_simbad(self):
        self.tab_simbad = crossmatch_simbad(self.df_region, radius=self.radius_simbad)
        self.tab_simbad_cmatch, self.tab_simbad_no_cmatch = self.split_by_max_seperation(self.tab_simbad)

    def crossmatch_gaia(self):
        self.tab_gaia = crossmatch_gaia(self.df_region, radius=self.radius_gaia)
        self.tab_gaia_cmatch, self.tab_gaia_no_cmatch = self.split_by_max_seperation(self.tab_gaia)

    def crossmatch_om(self):
        self.tab_om = crossmatch_xmm_om(self.df_region, radius=self.radius_xmm_om)
        self.tab_om_cmatch, self.tab_om_no_cmatch = self.split_by_max_seperation(self.tab_om)

    def crossmatch_unique_regions(self):
        self.df_unique_region = crossmatch_simulation.get_unique_sources(df_regions=self.df_region, clustering_radius=self.radius_self_cmatch)

    def plot_pie_chart(self):
        fig, ax = plt.subplots(3,2, figsize=(8,8))
        info = self.info

        def autopct_format(values):
            def my_format(pct):
                total = sum(values)
                val = int(round(pct * total / 100.0))
                return f'{pct:.1f}%\n({val:d})'
            return my_format

        pie_kwargs = {'labels' : ['Crossmatch', 'No Crossmatch'],
                      'colors' : ['lime', 'red']}

        fig.suptitle(f'Crossmatch Information | {info["n_regions"]} Regions | max_sep={info["max_sep"]}')

        ax[0,0].set_title('4XMM DR13')
        data = [info['n_dr13_slim_cmatch'], info['n_dr13_slim_no_cmatch']]
        ax[0,0].pie(data, autopct=autopct_format(data), **pie_kwargs)

        ax[0,1].set_title('Tranin DR12')
        data = [info['n_tranin_dr12_cmatch'], info['n_tranin_dr12_no_cmatch']]
        ax[0,1].pie(data, autopct=autopct_format(data), **pie_kwargs)

        ax[1,0].set_title(f'GAIA ({info['radius_gaia']})')
        data = [info['n_gaia_dr3_cmatch'], info['n_gaia_dr3_no_cmatch']]
        ax[1,0].pie(data, autopct=autopct_format(data), **pie_kwargs)

        ax[1,1].set_title(f'SIMBAD ({info["radius_simbad"]})')
        data = [info['n_simbad_cmatch'], info['n_simbad_no_cmatch']]
        ax[1,1].pie(data, autopct=autopct_format(data), **pie_kwargs)

        ax[2,0].set_title(f'OM ({info["radius_xmm_om"]})')
        data = [info['n_om_cmatch'], info['n_om_no_cmatch']]
        ax[2,0].pie(data, autopct=autopct_format(data), **pie_kwargs)

        ax[2,1].axis('off')
        plt.savefig(data_plots / 'crossmatch_pie_chart.png')
        plt.savefig(data_plots / 'crossmatch_pie_chart.pdf')

        # plt.show()

    def plot_seperations(self):
        fig, ax = plt.subplots(2,2, figsize=(8,8))
        bins = np.linspace(start=0, stop=100, num=100)

        ax[0,0].set_title('4XMM DR14')
        ax[0,0].hist(self.tab_dr14['SEP_ARCSEC'], bins=bins, histtype='step', label='Total', color='black')
        ax[0,0].hist(self.tab_dr14_cmatch['SEP_ARCSEC'], bins=bins, histtype='step', label=f'<{self.max_sep}', color='lime')
        ax[0,0].hist(self.tab_dr14_no_cmatch['SEP_ARCSEC'], bins=bins, histtype='step', label=f'>={self.max_sep}', color='red')
        ax[0,0].set_xlabel('Seperation (arcsec)')

        ax[0,1].set_title('4XMM DR13')
        ax[0,1].hist(self.tab_dr12['SEP_ARCSEC'], bins=bins, histtype='step', label='Total', color='black')
        ax[0,1].hist(self.tab_dr12_cmatch['SEP_ARCSEC'], bins=bins, histtype='step', label=f'<{self.max_sep}', color='lime')
        ax[0,1].hist(self.tab_dr12_no_cmatch['SEP_ARCSEC'], bins=bins, histtype='step', label=f'>={self.max_sep}', color='red')
        ax[0,1].set_xlabel('Seperation (arcsec)')

        ax[1,0].set_title('GAIA')
        ax[1,0].hist(self.tab_gaia['SEP_ARCSEC'], bins=bins, histtype='step', label='Total', color='black')
        ax[1,0].hist(self.tab_gaia_cmatch['SEP_ARCSEC'], bins=bins, histtype='step', label=f'<{self.max_sep}', color='lime')
        ax[1,0].hist(self.tab_gaia_no_cmatch['SEP_ARCSEC'], bins=bins, histtype='step', label=f'>={self.max_sep}', color='red')
        ax[1,0].set_xlabel('Seperation (arcsec)')

        ax[1,1].set_title('SIMBAD')
        ax[1,1].hist(self.tab_simbad['SEP_ARCSEC'], bins=bins, histtype='step', label='Total', color='black')
        ax[1,1].hist(self.tab_simbad_cmatch['SEP_ARCSEC'], bins=bins, histtype='step', label=f'<{self.max_sep}', color='lime')
        ax[1,1].hist(self.tab_simbad_no_cmatch['SEP_ARCSEC'], bins=bins, histtype='step', label=f'>={self.max_sep}', color='red')
        ax[1,1].set_xlabel('Seperation (arcsec)')

        for a in ax.flatten():
            a.legend()
        plt.savefig(data_plots / 'crossmatch_seperations.png')
        plt.savefig(data_plots / 'crossmatch_seperations.pdf')
        # plt.show()

    @property
    def info(self):
        info = {'radius_simbad'           : self.radius_simbad,
                'radius_gaia'             : self.radius_gaia,
                'radius_xmm_om'           : self.radius_xmm_om,
                'max_sep'                 : self.max_sep,
                'n_regions'               : len(self.df_region),
                'n_unique_regions'        : len(self.df_unique_region),
                'n_dr13_slim'             : len(self.tab_dr14),
                'n_dr13_slim_cmatch'      : len(self.tab_dr14_cmatch),
                'n_dr13_slim_no_cmatch'   : len(self.tab_dr14_no_cmatch),
                'n_tranin_dr12'           : len(self.tab_dr12),
                'n_tranin_dr12_cmatch'    : len(self.tab_dr12_cmatch),
                'n_tranin_dr12_no_cmatch' : len(self.tab_dr12_no_cmatch),
                'n_simbad'                : len(self.tab_simbad),
                'n_simbad_cmatch'         : len(self.tab_simbad_cmatch),
                'n_simbad_no_cmatch'      : len(self.tab_simbad_no_cmatch),
                'n_gaia_dr3'              : len(self.tab_gaia),
                'n_gaia_dr3_cmatch'       : len(self.tab_gaia_cmatch),
                'n_gaia_dr3_no_cmatch'    : len(self.tab_gaia_no_cmatch),
                'n_om'                    : len(self.tab_om),
                'n_om_cmatch'             : len(self.tab_om_cmatch),
                'n_om_no_cmatch'          : len(self.tab_om_no_cmatch),}
        for k, v in info.items():
            logger.info(f'{k:>25} : {v}')
        return info


if __name__ == "__main__":
    from exod.utils.path import data_combined
    import pandas as pd
    df_region = pd.read_csv(data_combined / 'merged_with_dr14' / 'df_regions.csv')
    df_region = df_region[df_region['runid'].str.contains('50_0.2_12.0')]
    df_region = df_region.iloc[100:200]
    # df_region = df_region.sample(1000)
    crossmatch = CrossMatch(df_region)
    crossmatch.run()
    cmatch_info = crossmatch.info
    crossmatch.plot_pie_chart()
    crossmatch.plot_seperations()
    plt.show()
