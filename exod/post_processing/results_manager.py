import base64
import io

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt

from exod.post_processing.extract_lc_features import calc_df_lc_feat_filter_flags
from exod.post_processing.hot_regions import rotate_regions_to_detector_coords, calc_hot_region_flags, hot_regions
from exod.post_processing.util import calc_detid_column
from exod.utils.path import savepaths_combined
from exod.post_processing.cluster_regions import ClusterRegions
from exod.post_processing.filter_subsets import SubsetManager, get_filters, generate_valid_combinations, Subset
import pandas as pd


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
    plt.close(fig)
    return lightcurve_data_url


def create_iau_srcids(ra_deg, dec_deg):
    """Create Source Identifiers from coordinates in degrees (IAU name)."""
    sc = SkyCoord(ra=ra_deg, dec=dec_deg, unit='deg')
    srcids = []
    for s in sc:
        ra   = s.ra.to_string(unit=u.hour, sep='', precision=2, pad=True)
        dec  = s.dec.to_string(sep='', precision=2, alwayssign=True, pad=True)
        name = f"EXOD J{ra}{dec}"
        srcids.append(name)
    return srcids


class RegionIdentifier:
    def make_region_identifier_from_runid_label(self, runid, label):
        key = str((runid, str(label)))
        return key

    def decode_region_identifier(self, region_identifier):
        pass

    def decode_runid(self, runid):
        obsid, label, t_bin, E_lo, E_hi = runid.split('_')
        return {'obsid' : obsid, 'label' : label, 't_bin' : t_bin, 'E_lo' : E_lo, 'E_hi' : E_hi}


class ResultsManager:
    def __init__(self):
        self.load_results()
        self.cluster_regions()
        self.calc_subsets()
        self.calc_flags()

    def load_results(self):
        self.df_dl            = pd.read_csv(savepaths_combined['dl_info'])
        self.df_cmatch_simbad = pd.read_csv(savepaths_combined['cmatch_simbad'])
        self.df_cmatch_om     = pd.read_csv(savepaths_combined['cmatch_om'])
        self.df_cmatch_gaia   = pd.read_csv(savepaths_combined['cmatch_gaia'])
        self.df_cmatch_glade  = pd.read_csv(savepaths_combined['cmatch_glade'])
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
        self.df_regions_rotated = rotate_regions_to_detector_coords(self.df_regions, clobber=False)

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

    def calc_flags(self):
        self.df_regions = calc_detid_column(self.df_regions)
        self.df_regions['filt_hot_pixel'] = calc_hot_region_flags(self.df_regions, self.df_regions_rotated, hot_regions)
        self.df_lc_features = calc_df_lc_feat_filter_flags(self.df_lc_features)

    def get_flags(self, region_id):
        lc_feat = self.df_lc_features.iloc[region_id]
        reg = self.df_regions.iloc[region_id]
        
        flags = {'region_id'            : region_id,
                 'n_max_isolated_flare' : lc_feat['n_max_isolated_flare'],
                 'filt_exclude_obsid'   : lc_feat['filt_exclude_obsid'],
                 'filt_hot_pixel'       : reg['filt_hot_pixel']}
        print(flags)
        return flags

    def get_all_flags(self, region_ids):
        return [self.get_flags(region_id) for region_id in region_ids]

    def get_results_for_obsid(self, obsid):
        pass

    def get_unique_region_summary(self, unique_region_id):
        unique_region_id   = int(unique_region_id)
        region_ids = self.df_regions_unique.iloc[unique_region_id]['idxs']  # Indices of unique region
        reg_info   = self.df_regions.iloc[region_ids[0]]  # Get first index of the region
        obsid, obs_subset_num, t_bin, E_lo, E_hi = reg_info['runid'].split('_')

        evt_info           = self.df_evt.loc[obsid].iloc[0]
        cmatch_simbad_info = self.df_cmatch_simbad.iloc[unique_region_id]
        cmatch_gaia_info   = self.df_cmatch_gaia.iloc[unique_region_id]
        cmatch_om_info     = self.df_cmatch_om.iloc[unique_region_id]
        cmatch_xmm_info    = self.df_cmatch_xmm.iloc[unique_region_id]
        cmatch_glade_info  = self.df_cmatch_glade.iloc[unique_region_id]
        run_info           = self.df_run.loc[obsid]

        lightcurves = self.get_and_plot_lcs_by_idxs(region_ids)
        lc_one_plot = self.plot_lcs_one_plot(region_ids)
        flags       = self.get_all_flags(region_ids)

        content = {
            'region_id'         : unique_region_id,
            'region_ids'        : region_ids,
            'next_id'           : unique_region_id + 1,
            'prev_id'           : unique_region_id - 1,
            'N_regions'         : len(region_ids),
            'reg_info'          : reg_info,
            'evt_info'          : evt_info,
            'run_info'          : run_info,
            'cmatch_info'       : cmatch_simbad_info,
            'cmatch_gaia_info'  : cmatch_gaia_info,
            'cmatch_om_info'    : cmatch_om_info,
            'cmatch_xmm_info'   : cmatch_xmm_info,
            'cmatch_glade_info' : cmatch_glade_info,
            'obsid'             : obsid,
            't_bin'             : t_bin,
            'E_lo'              : E_lo,
            'E_hi'              : E_hi,
            'lightcurves'       : lightcurves,
            'lc_one_plot'       : lc_one_plot,
            'flags'             : flags
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

        lightcurves = self.get_and_plot_lcs_by_idxs(idxs)

        content = {'otype'                  : otype,
                   'df_otype_stats'         : self.df_otype_stats,
                   'df_cmatch_simbad_otype' : df_cmatch_simbad_otype,
                   'lightcurves'            : lightcurves,
                   'df_regions_to_plot'     : df_regions_to_plot}
        return content


    def get_lc_by_idx(self, idx):
        lc_idxs = self.df_lc_idx.iloc[idx]
        start, stop = lc_idxs['start'], lc_idxs['stop']
        df_lc = pd.read_hdf(savepaths_combined['lc'], start=start, stop=stop)
        return df_lc

    def get_lcs_by_idxs(self, idxs):
        return [self.get_lc_by_idx(idx) for idx in idxs]

    def get_and_plot_lcs_by_idxs(self, idxs):
        df_regions = self.df_regions.iloc[idxs]
        lightcurves = []
        plt_labels = []
        for region_id, row in df_regions.iterrows():
            runid   = row['runid']
            label   = row['label']
            ra      = row['ra']
            dec     = row['dec']
            ra_deg  = row['ra_deg']
            dec_deg = row['dec_deg']

            unique_region_id = self.cr.region_num_to_cluster_num[region_id]
            plt_labels.append(f'runid={runid} reg_id={region_id} unique_id={unique_region_id} ra={ra_deg:.2f} dec={dec_deg:.2f}')

            res = {'region_id'        : region_id,
                   'unique_region_id' : unique_region_id,
                   'runid'            : runid,
                   'label'            : label,
                   'ra_deg'           : round(ra_deg,4),
                   'dec_deg'          : round(dec_deg,4),
                   'ra'               : ra,
                   'dec'              : dec}
            lightcurves.append(res)

        dfs = self.get_lcs_by_idxs(idxs)
        data_urls = [plot_lc(df_lc, label=plt_labels[i]) for i, df_lc in enumerate(dfs)]

        for i, r in enumerate(lightcurves):
            r['data_url'] = data_urls[i]

        return lightcurves

    def plot_lcs_one_plot(self, idxs):
        df_lcs = self.get_lcs_by_idxs(idxs)
        regions = self.df_regions.iloc[idxs]
        reg_tbins = {'5s'   : regions[regions['runid'].str.contains('_5_')],
                     '50s'  : regions[regions['runid'].str.contains('_50_')],
                     '200s' : regions[regions['runid'].str.contains('_200_')]}

        fig = plt.figure(figsize=(15, 2))
        ax = fig.subplots()
        for df in df_lcs:
            ax.step(df['time'], df['n'], lw=1.0)

        fig.tight_layout()
        fig.subplots_adjust(hspace=0, wspace=0)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        lightcurve_data_url = base64.b64encode(buf.read()).decode('ascii')
        plt.close(fig)
        return lightcurve_data_url


    def get_observation_summary(self, obsid):
        evt_info = self.df_evt.loc[obsid].iloc[0]
        run_info = self.df_run.loc[obsid]
        # Get regions
        df_regions_obs = self.df_regions[self.df_regions['runid'].str.contains(obsid)]
        tab_regions_obs = df_regions_obs[['runid', 'label', 'ra', 'dec', 'ra_deg', 'dec_deg']].to_html(table_id='myTable',
                                                                                                       classes='display compact')
        # Get Lightcurves
        mask = self.df_lc_idx.index.str.contains(obsid)
        idxs = np.where(mask)[0]
        lightcurves = self.get_and_plot_lcs_by_idxs(idxs)

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

        lightcurves = self.get_and_plot_lcs_by_idxs(df_lc_features_subset.index)

        content = {'lightcurves': lightcurves, 'subset': subset}
        return content


if __name__ == "__main__":
    rm = ResultsManager()


