import base64
import io

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from exod.utils.path import savepaths_combined
from exod.post_processing.cluster_regions import ClusterRegions
from exod.post_processing.filter_subsets import SubsetManager, get_filters, generate_valid_combinations, Subset
import pandas as pd

class RegionIdentifier:
    def make_region_identifier_from_runid_label(self, runid, label):
        key = str((runid, str(label)))
        return key

    def decode_region_identifier(self, region_identifier):
        pass

    def decode_runid(self, runid):
        obsid, label, t_bin, E_lo, E_hi = runid.split('_')
        return {'obsid':obsid, 'label':label, 't_bin':t_bin, 'E_lo':E_lo, 'E_hi':E_hi}


class ResultsManager:
    def __init__(self):
        self.load_results()
        self.cluster_regions()
        self.calc_subsets()

    def load_results(self):
        self.df_dl            = pd.read_csv(savepaths_combined['dl_info'])
        self.df_cmatch_simbad = pd.read_csv(savepaths_combined['cmatch_simbad'])
        self.df_cmatch_om     = pd.read_csv(savepaths_combined['cmatch_om'])
        self.df_cmatch_gaia   = pd.read_csv(savepaths_combined['cmatch_gaia'])
        self.df_cmatch_xmm    = pd.read_csv(savepaths_combined['cmatch_dr14'])
        self.df_dc            = pd.read_csv(savepaths_combined['dc_info'])
        self.df_evt           = pd.read_csv(savepaths_combined['evt_info'], index_col='obsid', dtype={'obsid':str})
        self.df_obs           = pd.read_csv(savepaths_combined['obs_info'], dtype={'obsid':str}, low_memory=False)
        self.df_run           = pd.read_csv(savepaths_combined['run_info'], index_col='obsid', dtype={'obsid':str})
        self.df_sim           = pd.read_csv('/home/nkhan/EXOD2/data/results_combined/merged_with_dr14/EXOD_simlist.csv', dtype={'obsid':str})
        self.df_lc_idx        = pd.read_csv(savepaths_combined['lc_idx'], index_col='Unnamed: 0')
        self.df_lc_features   = pd.read_csv(savepaths_combined['lc_features'], dtype={'obsid':str})
        self.df_regions       = pd.read_csv(savepaths_combined['regions'])
        self.df_otype_stats   = pd.read_csv('/home/nkhan/EXOD2/data/results_combined/simbad_stats/EXOD FULL_otype_stats.csv')
        #self.df_lc            = pd.read_csv(savepaths_combined['lc'], dtype={'obsid':str})

    def cluster_regions(self):
        cr = ClusterRegions(self.df_regions)
        cr.cluster_regions()
        self.df_regions_unique = cr.df_regions_unique
        self.cr = cr

    def calc_subsets(self):
        self.df_regions['sigma_max_B_peak']    = self.df_lc_features['sigma_max_B_peak']
        self.df_regions['sigma_max_B_eclipse'] = self.df_lc_features['sigma_max_B_eclipse']
        self.df_regions['DR14_SEP_ARCSEC']     = self.df_regions['cluster_label'].map(self.df_cmatch_xmm['SEP_ARCSEC'])
        self.df_regions['SIMBAD_SEP_ARCSEC']   = self.df_regions['cluster_label'].map(self.df_cmatch_simbad['SEP_ARCSEC'])
        filters = get_filters()
        valid_combinations = generate_valid_combinations(*filters)
        sm = SubsetManager()
        sm.add_subsets([Subset(f, self.df_regions) for f in valid_combinations])
        sm.calc_all()
        self.subset_manager = sm
           
    def get_results_for_runid_label(self, runid, label):
        pass
    
    def get_results_for_unique_region(self, unique_region_id):
        region_idxs = self.df_regions_unique.iloc[unique_region_id]['idxs'] 
        reg_info    = self.df_regions.iloc[region_idxs[0]] # This is bad.
        obsid, obsid_subset, t_bin, E_lo, E_hi = reg_info['runid'].split('_')
        evt_info           = self.df_evt.loc[obsid].iloc[0]
        cmatch_simbad_info = self.df_cmatch_simbad.iloc[unique_region_id]
        cmatch_gaia_info   = self.df_cmatch_gaia.iloc[unique_region_id]
        cmatch_om_info     = self.df_cmatch_om.iloc[unique_region_id]
        xmm_info           = self.df_cmatch_xmm.iloc[unique_region_id]
        run_info           = self.df_run.loc[obsid]

    def get_results_for_obsid(self, obsid):
        pass

    def get_region_summary(self, region_id):
        region_id = int(region_id)
        next_id = region_id + 1
        prev_id = region_id - 1

        region_idxs = self.df_regions_unique.iloc[region_id]['idxs']  # Indices of unique region
        reg_info = self.df_regions.iloc[region_idxs[0]]  # Get first index of the region
        obsid, obs_subset_num, t_bin, E_lo, E_hi = reg_info['runid'].split('_')

        evt_info = self.df_evt.loc[obsid].iloc[0]
        cmatch_simbad_info = self.df_cmatch_simbad.iloc[region_id]
        cmatch_gaia_info = self.df_cmatch_gaia.iloc[region_id]
        cmatch_om_info = self.df_cmatch_om.iloc[region_id]
        xmm_info = self.df_cmatch_xmm.iloc[region_id]
        run_info = self.df_run.loc[obsid]

        lightcurves = []
        for idx in region_idxs:
            df_region = self.df_regions.iloc[idx]
            runid = df_region['runid']
            label = df_region['label']
            key = str((runid, str(label)))

            # Use helper function to get light curve data
            df_lc = self.get_lc_by_idx(idx)
            label = f'reg_id={idx} key={key}'
            lightcurve_data_url = plot_lc(df_lc, label=label)
            lightcurves.append({
                'data_url': lightcurve_data_url,
                'region_id': idx,
                'runid': runid,
                'label': label,
                'ra': df_region['ra'],
                'dec': df_region['dec'],
                'ra_deg': df_region['ra_deg'],
                'dec_deg': df_region['dec_deg']
            })

        # Bundle the content as a dictionary for Flask
        content = {
            'region_id': region_id,
            'region_idxs': region_idxs,
            'next_id': next_id,
            'prev_id': prev_id,
            'reg_info': reg_info,
            'evt_info': evt_info,
            'run_info': run_info,
            'cmatch_info': cmatch_simbad_info,
            'cmatch_gaia_info': cmatch_gaia_info,
            'cmatch_om_info': cmatch_om_info,
            'xmm_info': xmm_info,
            'obsid': obsid,
            'label': label,
            't_bin': t_bin,
            'E_lo': E_lo,
            'E_hi': E_hi,
            'lightcurves': lightcurves
        }
        return content

    def get_otype_summary(self, otype):
        df_cmatch_simbad_otype = self.df_cmatch_simbad[self.df_cmatch_simbad['main_type'] == otype]
        df_cmatch_gaia_otype   = self.df_cmatch_gaia.loc[df_cmatch_simbad_otype.index]
        df_cmatch_om_otype     = self.df_cmatch_om.loc[df_cmatch_simbad_otype.index]
        df_cmatch_xmm_otype    = self.df_cmatch_xmm.loc[df_cmatch_simbad_otype.index]

        idxs = [self.df_regions_unique.iloc[i]['idxs'] for i in df_cmatch_simbad_otype.index]
        idxs = [idx for sublist in idxs for idx in sublist]

        df_regions_to_plot = self.df_regions.iloc[idxs]
        print(f'Found {len(idxs)} lightcurves in {len(df_cmatch_simbad_otype)} unique regions for {otype}')

        lightcurves = []
        for idx in tqdm(idxs):
            df_lc = self.get_lc_by_idx(idx)
            reg = self.cr.df_regions.iloc[idx]  # The 'cluster_label' is calculated in clusterregions.
            ra = reg['ra_deg']
            dec = reg['dec_deg']
            unique_region_id = reg['cluster_label']
            label = f'reg_id={idx} unique_id={unique_region_id} ra={ra:.2f} dec={dec:.2f}'
            lightcurve_data_url = plot_lc(df_lc, label)
            lightcurves.append({'data_url': lightcurve_data_url, 'region_id': unique_region_id})

        content = {'otype': otype,
                   'df_otype_stats': self.df_otype_stats,
                   'df_cmatch_simbad_otype': df_cmatch_simbad_otype,
                   'lightcurves': lightcurves,
                   'df_regions_to_plot': df_regions_to_plot}
        return content


    def get_lc_by_idx(self, idx):
        lc_idxs = self.df_lc_idx.iloc[idx]
        start, stop = lc_idxs['start'], lc_idxs['stop']
        df_lc = pd.read_hdf(savepaths_combined['lc'], start=start, stop=stop)
        return df_lc


    def get_observation_summary(self, obsid):
        evt_info = self.df_evt.loc[obsid].iloc[0]
        run_info = self.df_run.loc[obsid]
        # Get regions
        df_regions_obs = self.df_regions[self.df_regions['runid'].str.contains(obsid)]
        tab_regions_obs = df_regions_obs[['runid', 'label', 'ra', 'dec', 'ra_deg', 'dec_deg']].to_html(table_id='myTable',
                                                                                                       classes='display compact')
        # Get Lightcurves
        lightcurves = []
        mask = rm.df_lc_idx.index.str.contains(obsid)
        idxs = np.where(mask)[0]
        df_lc_idx_obs = self.df_lc_idx[mask]
        i = 0
        for key, row in tqdm(df_lc_idx_obs.iterrows()):
            df_lc = pd.read_hdf(savepaths_combined['lc'], start=row['start'], stop=row['stop'])
            region_id = idxs[i]
            df_region = self.df_regions.iloc[region_id]
            unique_reg_id = self.cr.region_num_to_cluster_num[region_id]
            label = f'key={key} reg_id={region_id} unique_reg_id={unique_reg_id}'
            lightcurve_data_url = plot_lc(df_lc, label)
            lightcurves.append({'data_url': lightcurve_data_url,
                                'region_id': unique_reg_id,
                                'ra_deg': df_region['ra_deg'],
                                'dec_deg': df_region['dec_deg'],
                                'ra': df_region['ra'],
                                'dec': df_region['dec']})
            i += 1
        content = {'obsid': obsid,
                   'tab_regions_obs': tab_regions_obs,
                   'evt_info': evt_info,
                   'run_info': run_info,
                   'lightcurves': lightcurves}
        return content

    def get_subset_summary(self, subset_num):
        subset = self.subset_manager.get_subset_by_index(int(subset_num))
        df_lc_features_subset = self.df_lc_features.loc[subset.df.index]
        df_lc_features_subset = df_lc_features_subset.sort_values('n_max', ascending=False)
        lightcurves = []
        for idx in tqdm(df_lc_features_subset.index):
            df_lc = self.get_lc_by_idx(idx)
            unique_reg_id = self.cr.region_num_to_cluster_num[idx]
            label = f'reg_id={idx} unique_reg_id={unique_reg_id}'
            lightcurve_data_url = plot_lc(df_lc, label)
            lightcurves.append({'data_url': lightcurve_data_url, 'region_id': unique_reg_id})
        content = {'lightcurves': lightcurves,
                   'subset': subset}
        return content

if __name__ == "__main__":
    rm = ResultsManager()


def plot_lc(df_lc, label):
    df_lc['t0'] = df_lc['time'] - df_lc['time'].min()
    fig = plt.figure(figsize=(15, 2))
    ax = fig.subplots()
    ax.step(df_lc['t0'], df_lc['n'], lw=1.0, color='black', label=label)
    ax.step(df_lc['t0'], df_lc['mu'], color='red', lw=1.0)
    ax.set_ylabel('Counts')
    ax.set_xlim(0, df_lc['t0'].max())
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.subplots_adjust(hspace=0, wspace=0)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    lightcurve_data_url = base64.b64encode(buf.read()).decode('ascii')
    return lightcurve_data_url


