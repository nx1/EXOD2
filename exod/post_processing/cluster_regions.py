"""
Cluster regions in a DataFrame of regions by clustering sources within a certain radius using a K-D tree.

An example of how this works is provided below:

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
"""
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
from astropy import units as u
from scipy.spatial import KDTree, distance
from astropy.table import Table

from exod.processing.coordinates import ra_dec_to_xyz
from exod.utils.logger import logger
from exod.utils.path import savepaths_combined


class ClusterRegions:
    """
    Cluster regions by clustering sources within a certain radius using a K-D tree.

    Some definitions:
        Region Number: The index of the region in the DataFrame.
        Cluster: A collection of regions, e.g (0,) or (1,7)
        Cluster Number: The index of the cluster in the list of clusters.

    Attributes:
        df_regions (pd.DataFrame): DataFrame containing ra_deg and dec_deg columns.
        clustering_radius (astropy.units.Quantity): Clustering radius in arcseconds.
        xyz (np.ndarray): Cartesian coordinates of the regions.
        clusters (list): List of lists containing the regions in each cluster.
        cluster_xyz_means (dict): Maps the cluster number to the mean cartesian position of the cluster on the unit sphere.
        region_to_clusters (dict): Maps the region number to the containing the regions associated with each cluster.
        cluster_num_to_cluster (dict): Maps the cluster number to the cluster.
        cluster_to_cluster_num (dict): Maps the cluster to the cluster number. (reverse of cluster_num_to_cluster)
        region_num_to_cluster_num (dict): Maps the region number to the cluster number (unique region number).
        df_regions_unique (pd.DataFrame): DataFrame containing the unique regions.
        n_clusters (int): Number of unique regions.
    """
    def __init__(self, df_regions, clustering_radius=20 * u.arcsec):
        self.df_regions = df_regions
        self.clustering_radius = clustering_radius

        self.xyz = []                           # Cartesian coordinates of the regions
        self.clusters = []                      # List of lists containing the regions in each cluster
        self.cluster_xyz_means = []             # Maps the cluster number to the mean cartesian position of the cluster on the unit sphere
        self.region_to_clusters = {}            # Maps the region number to the containing the regions associated with each cluster
        self.cluster_num_to_cluster = {}        # Maps the cluster number to the cluster
        self.cluster_to_cluster_num = {}        # Maps the cluster to the cluster number (reverse of cluster_num_to_cluster)
        self.region_num_to_cluster_num = {}     # Maps the region number to the cluster number (unique region number)
        self.df_regions_unique = pd.DataFrame() # DataFrame containing the unique regions
        self.n_clusters = 0                     # Number of unique regions

        self.run()

    def number_clusters(self):
        """Create dictionary that maps each unique cluster to its unique cluster number."""
        cluster_to_cluster_num = OrderedDict()
        i = 0
        for j, cluster in enumerate(self.clusters):
            c_t = tuple(cluster)
            if c_t not in cluster_to_cluster_num:
                cluster_to_cluster_num[c_t] = i
                i += 1
        logger.info(f'Initially found {len(cluster_to_cluster_num)} unique region clusters...')
        self.cluster_to_cluster_num = cluster_to_cluster_num
        # Create reverse mapping (cluster number to cluster)
        self.cluster_num_to_cluster = dict((v, k) for k, v in self.cluster_to_cluster_num.items())

    def map_region_num_to_clusters(self):
        """Create dictionary that maps the region number to the cluster."""
        self.region_to_clusters = defaultdict(list)
        for c in self.clusters:
            for region_num in c:
                self.region_to_clusters[region_num].append(c)

    def calc_cluster_xyz_means(self):
        """ Calculate the mean cartesian positions of each cluster."""
        self.cluster_xyz_means = {tuple(cluster): np.mean([self.xyz[region] for region in cluster], axis=0) for cluster in self.clusters}

    def cluster_regions(self):
        """Find unique regions by clustering sources within a certain radius using a K-D tree."""
        self.xyz = ra_dec_to_xyz(self.df_regions['ra_deg'].values * u.deg, self.df_regions['dec_deg'].values * u.deg)
        kd_tree = KDTree(data=self.xyz, leafsize=10, compact_nodes=True, copy_data=False, balanced_tree=True, boxsize=None)
        clustering_radius_rad = self.clustering_radius.to('rad').value
        self.clusters = kd_tree.query_ball_tree(other=kd_tree, r=clustering_radius_rad, p=2.0, eps=0)

    def label_region_num_to_cluster_num(self):
        """Loop over all regions and label them to the closest cluster."""
        cluster_xyz_means = self.cluster_xyz_means
        region_to_clusters = self.region_to_clusters
        cluster_to_cluster_num = self.cluster_to_cluster_num
        xyz = self.xyz

        cluster_labels = []
        for i in range(len(xyz)):
            clusters = region_to_clusters[i]
            if len(clusters) == 1:
                c_t = tuple(clusters[0])
                unique_region_num = cluster_to_cluster_num[c_t]
                cluster_labels.append(unique_region_num)

            elif len(clusters) > 1: # More than one cluster associated with region i
                cluster_distances = {}
                for cluster in region_to_clusters[i]:
                    c_t = tuple(cluster)
                    d = distance.euclidean(xyz[i], cluster_xyz_means[c_t])
                    cluster_distances[c_t] = d

                # Find the closest cluster
                closest_cluster = min(cluster_distances, key=cluster_distances.get)
                cluster_labels.append(cluster_to_cluster_num[closest_cluster])

        self.cluster_labels = cluster_labels
        self.df_regions['cluster_label'] = cluster_labels
        self.n_clusters = len(set(self.cluster_labels))
        logger.info(f'Final number of unique regions = {self.n_clusters}')

    def calc_unique_regions_table(self):
        df_regions_unique = self.df_regions.groupby(['cluster_label'])[['ra_deg', 'dec_deg']].agg('mean')
        df_regions_unique['idxs'] = [list(self.cluster_num_to_cluster[c_num]) for c_num in df_regions_unique.index]
        self.df_regions_unique = df_regions_unique

    def renumber_clusters(self):
        """Renumber the clusters so that they go from 0 to n_clusters."""
        logger.info('Renumbering clusters...')
        old2new = {old:new for new, old in zip(range(self.n_clusters), self.df_regions_unique.index)}
        new2old = {new:old for old, new in old2new.items()}

        self.df_regions_unique.reset_index(drop=True, inplace=True)
        self.df_regions_unique.index.name = 'cluster_label'
        self.df_regions['cluster_label'] = self.df_regions['cluster_label'].map(old2new)
        self.cluster_labels = self.df_regions['cluster_label'].values
        self.cluster_num_to_cluster = {c_num : self.cluster_num_to_cluster[new2old[c_num]] for c_num in range(self.n_clusters)}
        self.cluster_to_cluster_num = {v:k for k,v in self.cluster_num_to_cluster.items()}
        self.region_num_to_cluster_num = {reg:cluster for reg, cluster in zip(self.df_regions.index, self.cluster_labels)}

    def save_unique_regions_table(self):
        logger.info(f'Saving unique regions table to {savepaths_combined["regions_unique"]}')
        tab_regions_unique = Table.from_pandas(self.df_regions_unique[['ra_deg', 'dec_deg']])
        tab_regions_unique.write(savepaths_combined['regions_unique'], overwrite=True)

    def run(self):
        self.cluster_regions()
        self.number_clusters()
        self.map_region_num_to_clusters()
        self.calc_cluster_xyz_means()
        self.label_region_num_to_cluster_num()
        self.calc_unique_regions_table()
        self.renumber_clusters()
        self.save_unique_regions_table()

if __name__ == "__main__":
    df_regions = pd.read_csv(savepaths_combined['regions'], index_col=0)
    cluster_regions = ClusterRegions(df_regions)
    #print(cluster_regions.region_to_clusters)
    #print(cluster_regions.cluster_num_to_cluster)
    #print(cluster_regions.cluster_to_cluster_num)

    print(cluster_regions.df_regions.tail())
    print(cluster_regions.df_regions_unique.tail())
