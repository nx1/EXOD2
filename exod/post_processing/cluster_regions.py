from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
from astropy import units as u
from scipy.spatial import KDTree, distance
from astropy.table import Table

from exod.utils.logger import logger
from exod.utils.path import savepaths_combined


def ra_dec_to_xyz(ra_deg, dec_deg):
    """
    Convert RA and DEC to points on the unit sphere.

    Parameters:
        dec_deg (astropy.units.Quantity): Declination in degrees.
        ra_deg (astropy.units.Quantity): Right Ascension in degrees.

    Returns:
        xyz (np.ndarray): The xyz positions of the points as a (N, 3) array.
    """
    ra_rad  = ra_deg.to(u.rad).value
    dec_rad = dec_deg.to(u.rad).value

    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    xyz = np.vstack((x, y, z)).T
    return xyz


def number_clusters(clusters):
    """
    Create dictionary that maps each unique cluster to its unique cluster number.

    Parameters:
        clusters (list): List of clusters.

    Returns:
        cluster_to_cluster_num (OrderedDict): Dictionary containing the unique regions.
    """
    logger.info(f'Numbering unique clusters...')
    cluster_to_cluster_num = OrderedDict()
    i = 0
    for j, cluster in enumerate(clusters):
        c_t = tuple(cluster)
        if c_t not in cluster_to_cluster_num:
            cluster_to_cluster_num[c_t] = i
            i += 1
    logger.info(f'Found {len(cluster_to_cluster_num)} unique region clusters...')
    return cluster_to_cluster_num


def label_region_num_to_cluster_num(cluster_xyz_means, region_to_clusters, unique_regions, xyz):
    """
    Label each region with the cluster number it is closest.
    Parameters:
        cluster_xyz_means (dict): Dictionary containing the cluster means.
        region_to_clusters (dict): Dictionary containing the regions associated with each cluster.
        unique_regions (OrderedDict): Dictionary containing the unique regions.
        xyz (np.ndarray): The xyz positions of the points as a (N, 3) array.

    Returns:
        cluster_labels (list): List of cluster labels for each unique source.
    """
    logger.info(f'Associating regions to clusters...')

    cluster_labels = []
    for i in range(len(xyz)):
        clusters = region_to_clusters[i]
        if len(clusters) == 1:
            c_t = tuple(clusters[0])
            unique_region_num = unique_regions[c_t]
            cluster_labels.append(unique_region_num)

        elif len(clusters) > 1:
            # More than one cluster associated with region i
            cluster_distances = {}
            for cluster in region_to_clusters[i]:
                c_t = tuple(cluster)
                d = distance.euclidean(xyz[i], cluster_xyz_means[c_t])
                cluster_distances[c_t] = d

            # Find the closest cluster
            closest_cluster = min(cluster_distances, key=cluster_distances.get)
            cluster_labels.append(unique_regions[closest_cluster])
    return cluster_labels


def map_region_num_to_clusters(clusters):
    """
    Create dictionary that maps the region number to the cluster.

    Parameters:
        clusters (list): List of clusters.

    Returns:
        region_to_clusters (dict): Dictionary containing the regions associated with each cluster.
    """
    logger.info(f'Mapping Regions to each cluster...')
    region_to_clusters = defaultdict(list)
    for c in clusters:
        for region_num in c:
            region_to_clusters[region_num].append(c)
    return region_to_clusters


def calc_cluster_xyz_means(clusters, xyz):
    """
    Calculate the mean position of each cluster.

    Parameters:
        clusters (list): List of clusters.
        xyz (np.ndarray): The xyz positions of the points as a (N, 3) array.

    Returns:
        cluster_means (dict): Dictionary containing the cluster means.
    """
    logger.info(f'Precomputing cluster means...')
    cluster_means = {tuple(cluster): np.mean([xyz[region] for region in cluster], axis=0) for cluster in clusters}
    return cluster_means


def cluster_regions(df_regions, clustering_radius=20 * u.arcsec):
    """
    Find unique regions in a DataFrame of regions by clustering sources within a certain radius using a K-D tree.

    Input Regions (cartesian coords)
    xyz = [[0,0,0], # 0
           [0,0,1], # 1
           [0,1,0], # 2
           [5,0,0], # 3
           [4,0,0], # 4
           [0,2,0], # 5
           [0,3,0], # 6

    cluster[0] = [0,1,3] --> cluster_num = 0 | mean position = 0,0,0 + 0,0,1 + 0,1,0 = 0.00, 0.33, 0.33
    cluster[1] = [0,1,3] --> cluster_num = 0 | mean position = 0,0,0 + 0,0,1 + 0,1,0 = 0.00, 0.33, 0.33
    cluster[2] = [1,3,6] --> cluster_num = 1 | mean position = 0,0,1 + 5,0,0 + 0,3,0 = 1.66, 1.00, 0.33
    cluster[3] = [1,3,6] --> cluster_num = 1 | mean position = 0,0,1 + 5,0,0 + 0,3,0 = 1.66, 1.00, 0.33

    Row 1 is both in cluster_num=1 and cluster_num=2, which one should we label it to?
    The closest mean position of course!

    xyz[1] = [0,0,1]

    distance.euclidean([0,0,1], [0.00, 0.33, 0.33]) = 0.75 (lowest!)
    distance.euclidean([0,0,1], 1.66, 1.00, 0.33)   = 2.05

    # Row 1 therefore gets associated with cluster num 0

    unique_regions = OrderedDict([((0,)                     : 0),
                                   ((1, 7)                  : 1),
                                   ((2, 8)                  : 2),
                                   ((3, 9)                  : 3),
                                   ((4, 10, 13, 16, 18)     : 4),
                                   ((5, 11, 14, 17, 19, 27) : 5),
                                   ((6,)                    : 6),
                                   ((12,)                   : 7),
                                   ((15,)                   : 8),
                                   ((20, 23)                : 9),

    region_to_clusters = defaultdict(list,
                                    {0: [[0]],
                                     1: [[1, 7], [1, 7]],
                                     7: [[1, 7], [1, 7]],
                                     2: [[2, 8], [2, 8]],
                                     8: [[2, 8], [2, 8]],
                                     3: [[3, 9], [3, 9]],
                                     9: [[3, 9], [3, 9]],
                                     4: [[4, 10, 13, 16, 18],
                                         [4, 10, 13, 16, 18],
                                         [4, 10, 13, 16, 18],
                                         [4, 10, 13, 16, 18],
                                         [4, 10, 13, 16, 18]],
                                     ... }

    cluster_xyz_means = {(0,)                : array([ 0.71092614,  0.12281395, -0.69245994]),
                     (1, 7)              : array([ 0.71165566,  0.12214711, -0.69182823]),
                     (2, 8)              : array([ 0.71478684,  0.1221836 , -0.68858618]),
                     (3, 9)              : array([ 0.71486105,  0.11933079, -0.68900932]),
                     (4, 10, 13, 16, 18) : array([ 0.71399846,  0.11878942, -0.68999657]),

    cluster_labels = [0, 1, 2, 3, 4, 5, 6, 1, 2, ...]

    Parameters:
        df_regions (pd.DataFrame): DataFrame containing ra_deg and dec_deg columns.
        clustering_radius (astropy.units.Quantity): Clustering radius in arcseconds.

    Returns:
        cluster_labels (list): List of cluster labels for each unique source.
    """
    xyz = ra_dec_to_xyz(df_regions['ra_deg'].values * u.deg, df_regions['dec_deg'].values * u.deg)

    kd_tree = KDTree(data=xyz, leafsize=10, compact_nodes=True, copy_data=False, balanced_tree=True, boxsize=None)
    clusters = kd_tree.query_ball_tree(other=kd_tree, r=clustering_radius.to('rad').value, p=2.0, eps=0)

    # Loop over all clusters and map unique clusters to an incrementally increasing number
    cluster_to_cluster_num = number_clusters(clusters)
    cluster_num_to_cluster = dict((v, k) for k, v in cluster_to_cluster_num.items())

    region_to_clusters = map_region_num_to_clusters(clusters)

    cluster_xyz_means = calc_cluster_xyz_means(clusters, xyz)

    cluster_labels = label_region_num_to_cluster_num(cluster_xyz_means, region_to_clusters, cluster_to_cluster_num, xyz)

    logger.info(f'Final number of unique regions = {len(set(cluster_labels))}')
    return cluster_labels


def get_unique_regions(df_regions, clustering_radius=20 * u.arcsec):
    xyz = ra_dec_to_xyz(df_regions['ra_deg'].values * u.deg, df_regions['dec_deg'].values * u.deg)

    kd_tree = KDTree(data=xyz, leafsize=10, compact_nodes=True, copy_data=False, balanced_tree=True, boxsize=None)
    clusters = kd_tree.query_ball_tree(other=kd_tree, r=clustering_radius.to('rad').value, p=2.0, eps=0)

    # Loop over all clusters and map unique clusters to an incrementally increasing number
    cluster_to_cluster_num = number_clusters(clusters)
    cluster_num_to_cluster = dict((v, k) for k, v in cluster_to_cluster_num.items())

    cluster_labels = cluster_regions(df_regions, clustering_radius=clustering_radius)
    df_regions['cluster_label'] = cluster_labels

    df_regions_unique = df_regions.groupby(['cluster_label'])[['ra_deg', 'dec_deg']].agg('mean')
    df_regions_unique['idxs'] = [list(cluster_num_to_cluster[c_num]) for c_num in df_regions_unique.index]

    tab_regions_unique = Table.from_pandas(df_regions_unique[['ra_deg', 'dec_deg']])
    tab_regions_unique.write(savepaths_combined['regions_unique'], overwrite=True)
    return df_regions_unique
