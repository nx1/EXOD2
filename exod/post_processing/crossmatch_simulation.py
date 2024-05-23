"""This module is used for various the simulation subsets with each other
In order to determine the crossmatch fraction and various other metrics."""

from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import pandas as pd
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord

from exod.utils.path import data_combined, data_plots
from exod.utils.logger import logger


def get_subset_keys():
    """Get the keys for the simulation subsets."""
    subsets = ['5_0.2_2.0',
               '5_2.0_12.0',
               '5_0.2_12.0',
               '50_0.2_2.0',
               '50_2.0_12.0',
               '50_0.2_12.0',
               '200_0.2_2.0',
               '200_2.0_12.0',
               '200_0.2_12.0']
    return subsets


def get_unique_sources(df_regions, clustering_radius=0.25 * u.arcsec):
    """
    Find unique sources in a DataFrame of regions by clustering sources within a certain radius.

    Parameters:
        df_regions (pd.DataFrame): DataFrame containing the region information.
        clustering_radius (astropy.units.Quantity): Clustering radius in arcseconds.

    Returns:
        df_sources_unique (pd.DataFrame): DataFrame containing the unique sources.
    """
    # Convert RA and DEC to points on the unit sphere.
    ra_deg  = df_regions['ra_deg'].values * u.degree
    dec_deg = df_regions['dec_deg'].values * u.degree
    ra_rad  = ra_deg.to(u.rad).value
    dec_rad = dec_deg.to(u.rad).value
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)

    # See https://en.wikipedia.org/wiki/K-d_tree
    xyz = np.vstack((x, y, z)).T
    tree = KDTree(xyz)

    # Find all pairs of points between self and other whose distance is at most r.
    clusters = tree.query_ball_tree(tree, clustering_radius.to(u.rad).value)
    assert len(clusters) == len(ra_deg)

    # Count sources in each cluster
    counts = [len(c) for c in clusters]
    counts = np.array(counts)

    # Create mask to only keep first value of the cluster
    mask_unique = np.full(len(ra_deg), True)
    for cluster in clusters:
        for i in cluster[1:]:  # Mask False out all associated indexs
            mask_unique[i] = False

    unique_ra = ra_deg[mask_unique]
    unique_dec = dec_deg[mask_unique]
    unique_counts = counts[mask_unique]
    idxs = [clusters[i] for i in range(len(clusters)) if mask_unique[i]]

    # Create a DataFrame for unique sources with counts
    df_sources_unique = pd.DataFrame({'ra_deg': unique_ra, 'dec_deg': unique_dec, 'idxs': idxs, 'count': unique_counts})
    logger.info(f'A total of {len(df_sources_unique)} unique sources were found from {len(ra_deg)} sources within a clustering radius of {clustering_radius}')
    return df_sources_unique

def split_subsets(df_regions):
    """
    Split the regions into subsets based on the t_bin, E_lo, E_hi values in the runid.

    Parameters:
        df_regions (pd.DataFrame): The DataFrame containing the region information.

    Returns:
        dfs_subsets (dict): A dictionary containing the DataFrames for each subset.
    """
    logger.info('Splitting df_regions into simulations subsets...')
    subsets = get_subset_keys()
    dfs_subsets = {}
    for s in subsets:
        dfs_subsets[s] = df_regions[df_regions['runid'].str.contains(s)]
    return dfs_subsets


def crossmatch_simulation_subsets(dfs_subsets):
    """
    Crossmatch the subsets of simulations to each other.

    Parameters:
        dfs_subsets (dict): A dictionary containing the DataFrames for each subset.

    Returns:
        dfs_subset_crossmatch (dict): A dictionary containing the crossmatch results for each subset.
            Each DataFrame contains the indices of the crossmatched sources in the other subsets.
            A placeholder value of -1 is denoted for sources without a crossmatch.
    """
    max_sep = 15 * u.arcsec  # Maximum separation for considering a crossmatch
    no_cmatch_id = -1  # Placeholder value for sources without a crossmatch
    dfs_subset_crossmatch = {}
    for k1, df1 in dfs_subsets.items():
        sc1 = SkyCoord(ra=df1['ra_deg'], dec=df1['dec_deg'], unit='deg', frame='fk5', equinox='J2000')  #
        res = {}
        res[k1] = np.arange(len(sc1))
        for k2, df2 in dfs_subsets.items():
            if k1 == k2:
                continue

            print(f'Matching {k1} ({len(df1)}) With {k2:<12} ({len(df2)})')
            sc2 = SkyCoord(ra=df2['ra_deg'], dec=df2['dec_deg'], unit='deg', frame='fk5', equinox='J2000')
            cmatch = sc1.match_to_catalog_sky(sc2)

            tab_cmatch = Table(cmatch)
            tab_cmatch.rename_columns(names=tab_cmatch.colnames, new_names=['idx', 'sep2d', 'dist3d'])
            tab_cmatch['sep2d'] = tab_cmatch['sep2d'].to(u.arcsec)

            is_match = np.where(tab_cmatch['sep2d'] < max_sep, tab_cmatch['idx'], no_cmatch_id)  # Replace
            res[k2] = is_match

        df = pd.DataFrame(res)
        dfs_subset_crossmatch[k1] = df
    return dfs_subset_crossmatch

def calc_subset_stats(dfs_subsets):
    all_res = []
    for k, df in dfs_subsets.items():
        t_bin, E_lo, E_hi = k.split('_')
        res = {'subset'      : k,
               't_bin'       : t_bin,
               'E_lo'        : E_lo,
               'E_hi'        : E_hi,
               'n_regions'   : len(df),
               'n_obsids'    : len(df['runid'].value_counts()),
               'reg/obs'     : len(df) / len(df['runid'].value_counts()),
               'mean counts' : df['intensity_mean'].mean(),
               'std counts' : df['intensity_mean'].std(),
               }
        all_res.append(res)
    
    df_region_subset = pd.DataFrame(all_res)
    return df_region_subset


def calc_subset_n_regions(dfs_subset_crossmatch):
    """Calculate the number of regions in each subset."""
    n_regions_sim = {}
    for k, df in dfs_subset_crossmatch.items():
        n_regions_sim[k] = len(df)
    return n_regions_sim


def calc_subset_cmatch_fraction(dfs_subset_crossmatch):
    """
    Calculate the fraction of successfully crossmatched regions.

    Parameters:
        dfs_subset_crossmatch (dict): A dictionary containing the crossmatch results for each subset.

    Returns:
        df_subset_cmatch_fraction (pd.DataFrame): A DataFrame containing the fraction of successfully crossmatched regions.
    """
    all_res = []
    for k, df in dfs_subset_crossmatch.items():
        res = {}
        for col in df.columns:
            count = (df[col] > -1).sum()
            perc = count / len(df)
            res[col] = perc
        all_res.append(res)
    df_subset_cmatch_fraction = pd.DataFrame(all_res)
    df_subset_cmatch_fraction.index = dfs_subset_crossmatch.keys()
    return df_subset_cmatch_fraction


def plot_mean_count_histogram(dfs_subsets):
    """Plot the mean count histograms for each subset."""
    linestyles = {'5_': 'solid',
                  '50_': 'dashed',
                  '200_': 'dotted'}

    colors = {'0.2_2.0': 'red',
              '2.0_12.0': 'blue',
              '0.2_12.0': 'black'}

    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax[0].set_title('Mean Count Histograms')
    for k, df in dfs_subsets.items():
        ls = [linestyles[key] for key in linestyles.keys() if key in k]
        c = [colors[key] for key in colors.keys() if key in k]

        hist_kwargs = {'bins'     : np.linspace(start=0, stop=500, num=50),
                       'histtype' : 'step',
                       'lw'       : 1.5,
                       'ls'       : ls[0],
                       'color'    : c[0],
                       'label'    : k}

        ax[0].hist(df['intensity_mean'], **hist_kwargs)
        ax[1].hist(df['intensity_mean'], density=True, **hist_kwargs)
    ax[0].set_ylabel('Number')
    ax[1].set_ylabel('Fraction')
    ax[1].set_xlabel('Intensity Mean (Counts)')
    for a in ax:
        a.legend()
        a.grid()
    plt.subplots_adjust(hspace=0)
    plt.savefig(data_plots / 'mean_count_histogram.png')
    plt.savefig(data_plots / 'mean_count_histogram.pdf')
    plt.show()


def plot_crossmatch_confusion_matrix(df_subset_cmatch_fraction, n_regions_sim):
    """Plot the confusion matrix for the crossmatch fractions."""
    labels_readable = {'5_0.2_2.0'    : r'$t_{\mathrm{bin}} = 5  ;E=0.2-2.0$',
                       '5_2.0_12.0'   : r'$t_{\mathrm{bin}} = 5  ;E=2.0-12.0$',
                       '5_0.2_12.0'   : r'$t_{\mathrm{bin}} = 5  ;E=0.2-12.0$',
                       '50_0.2_2.0'   : r'$t_{\mathrm{bin}} = 50 ;E=0.2-2.0$',
                       '50_2.0_12.0'  : r'$t_{\mathrm{bin}} = 50 ;E=2.0-12.0$',
                       '50_0.2_12.0'  : r'$t_{\mathrm{bin}} = 50 ;E=0.2-12.0$',
                       '200_0.2_2.0'  : r'$t_{\mathrm{bin}} = 200;E=0.2-2.0$',
                       '200_2.0_12.0' : r'$t_{\mathrm{bin}} = 200;E=2.0-12.0$',
                       '200_0.2_12.0' : r'$t_{\mathrm{bin}} = 200;E=0.2-12.0$'}
    lab1 = [labels_readable[i] for i in df_subset_cmatch_fraction.index]
    lab2 = [fr'$N_{{\mathrm{{reg}}}}$ = {n}' for n in n_regions_sim.values()]

    X = df_subset_cmatch_fraction.values
    fig, ax = plt.subplots(figsize=(10, 9.2))
    ax.set_title('Fraction of successfully crossmatched Regions')
    ax.imshow(X, cmap='Blues', interpolation='none')
    ax.set_xticks(range(X.shape[0]), lab1, rotation=90)
    ax.set_yticks(range(X.shape[0]), lab1, rotation=0)
    ax2 = ax.twinx()
    ax2.set_yticks(range(X.shape[0]), lab2, rotation=0)
    ax2.set_ylim(ax.get_ylim())
    for (j, i), label in np.ndenumerate(X):
        label = f'{label:.2f}'
        ax.text(i, j, label, ha='center', va='center')
    ax.set_xlabel('Crossmatch A')
    ax.set_ylabel('Crossmatch B')
    plt.tight_layout()
    logger.info('Saving Crossmatch Confusion Matrix...')
    plt.savefig(data_plots / 'crossmatch_confusion_matrix.png')
    plt.savefig(data_plots / 'crossmatch_confusion_matrix.pdf')
    plt.show()


def print_crossmatch_fraction(dfs_subset_crossmatch):
    for k, df in dfs_subset_crossmatch.items():
        print(f'{k} Results:')
        for col in df.columns[1:]:
            count = (df[col] > -1).sum()
            print(f'{col:<12} : {count:<5} / {len(df)} ({count / len(df):.2f})')
        print('=' * 40)


def main():
    df_region_path = data_combined / '30_4_2024/df_regions.csv'
    df_regions = pd.read_csv(df_region_path)

    dfs_subsets = split_subsets(df_regions=df_regions)

    df_subset_stats = calc_subset_stats(dfs_subsets=dfs_subsets)
    plot_mean_count_histogram(dfs_subsets=dfs_subsets)
    dfs_subset_crossmatch = crossmatch_simulation_subsets(dfs_subsets=dfs_subsets)

    df_subset_cmatch_fraction = calc_subset_cmatch_fraction(dfs_subset_crossmatch=dfs_subset_crossmatch)
    n_regions_sim = calc_subset_n_regions(dfs_subset_crossmatch=dfs_subset_crossmatch)
    plot_crossmatch_confusion_matrix(df_subset_cmatch_fraction=df_subset_cmatch_fraction, n_regions_sim=n_regions_sim)
    print_crossmatch_fraction(dfs_subset_crossmatch=dfs_subset_crossmatch)
    print(df_subset_stats)
    print(df_subset_cmatch_fraction)


if __name__ == "__main__":
    main()
