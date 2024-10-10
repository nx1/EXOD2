import base64
import io

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt

from exod.post_processing.crossmatch_runs import get_run_subset_keys
from exod.post_processing.extract_lc_features import calc_df_lc_feat_filter_flags
from exod.post_processing.hot_regions import rotate_regions_to_detector_coords, calc_hot_region_flags, hot_regions
from exod.post_processing.util import calc_detid_column
from exod.utils.path import savepaths_combined
from exod.post_processing.cluster_regions import ClusterRegions
from exod.post_processing.filter_subsets import SubsetManager, get_filters, generate_valid_combinations, Subset
import pandas as pd


def plot_lc(df_lc, label):
    fig = plt.figure(figsize=(11, 2))
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
        self.df_regions = self.decode_runids(self.df_regions)

    def load_results(self):
        self.df_dl            = pd.read_csv(savepaths_combined['dl_info'])
        self.df_cmatch_simbad = pd.read_csv(savepaths_combined['cmatch_simbad'])
        self.df_cmatch_om     = pd.read_csv(savepaths_combined['cmatch_om'])
        self.df_cmatch_gaia   = pd.read_csv(savepaths_combined['cmatch_gaia'])
        self.df_cmatch_glade  = pd.read_csv(savepaths_combined['cmatch_glade'])
        self.df_cmatch_xmm    = pd.read_csv(savepaths_combined['cmatch_dr14'])
        self.df_cmatch_chime  = pd.read_csv('/home/nkhan/EXOD2/data/results_combined/merged_with_dr14/df_regions_unique_cmatch_chime.csv')
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

    def decode_runids(self, df):
        df[['obsid', 'obsid_subset', 't_bin', 'E_lo', 'E_hi']] = df['runid'].str.split('_', expand=True)
        df = df.astype({'t_bin': 'float', 'E_lo': 'float', 'E_hi': 'float'})
        return df

    def get_flags(self, region_id):
        lc_feat = self.df_lc_features.iloc[region_id]
        reg = self.df_regions.iloc[region_id]
        
        flags = {'region_id'            : region_id,
                 'n_max_isolated_flare' : lc_feat['n_max_isolated_flare'],
                 'filt_exclude_obsid'   : lc_feat['filt_exclude_obsid'],
                 'filt_hot_pixel'       : reg['filt_hot_pixel']}
        return flags

    def get_all_flags(self, region_ids):
        return [self.get_flags(region_id) for region_id in region_ids]

    def get_unique_region_summary(self, unique_region_id):
        unique_region_id = int(unique_region_id)
        df_region_unique = self.df_regions_unique.iloc[unique_region_id]
        region_ids = df_region_unique['idxs']  # Region indexs for the unique region
        df_regions = self.df_regions.iloc[region_ids]

        cmatch_simbad_info = self.df_cmatch_simbad.iloc[unique_region_id]
        cmatch_gaia_info   = self.df_cmatch_gaia.iloc[unique_region_id]
        cmatch_om_info     = self.df_cmatch_om.iloc[unique_region_id]
        cmatch_xmm_info    = self.df_cmatch_xmm.iloc[unique_region_id]
        cmatch_glade_info  = self.df_cmatch_glade.iloc[unique_region_id]

        df_obs_info = self.get_observation_info_from_df_regions(df_regions)
        lightcurves = self.get_and_plot_lcs_by_idxs(region_ids)
        lc_one_plot = self.plot_lcs_one_plot(region_ids)

        content = {
            'region_id'         : unique_region_id,
            'df_region_unique'  : df_region_unique,
            'region_ids'        : region_ids,
            'next_id'           : unique_region_id + 1,
            'prev_id'           : unique_region_id - 1,
            'n_regions'         : len(region_ids),
            'df_obs_info'      : df_obs_info,
            'cmatch_info'       : cmatch_simbad_info,
            'cmatch_gaia_info'  : cmatch_gaia_info,
            'cmatch_om_info'    : cmatch_om_info,
            'cmatch_xmm_info'   : cmatch_xmm_info,
            'cmatch_glade_info' : cmatch_glade_info,
            'obsids'            : df_regions['obsid'].unique(),
            'n_obsids'          : df_regions['obsid'].nunique(),
            'lightcurves'       : lightcurves,
            'lc_one_plot'       : lc_one_plot,
        }
        return content

    def get_observation_info_from_df_regions(self, df_regions):
        cols_to_return = ['date', 'object', 'exposure', 'N_events', 'mean_rate']

        unique_obsids = df_regions['obsid'].unique()
        df_evt_filt = self.df_evt.loc[unique_obsids]
        df_run_filt = self.df_run.loc[unique_obsids]
        merged_df = df_evt_filt.merge(df_run_filt, left_index=True, right_index=True)

        df_obs_info = merged_df[cols_to_return].groupby(level=0).first().reset_index()

        df_obs_info['date'] = pd.to_datetime(df_obs_info['date']).dt.strftime('%Y-%m-%d')
        df_obs_info['mean_rate'] = df_obs_info['mean_rate'].round(1)
        df_obs_info['exposure'] = (df_obs_info['exposure'] / 1000).round().astype(int).astype(str) + 'k'
        df_obs_info['N_events'] = (df_obs_info['N_events'] / 1000).round().astype(int).astype(str) + 'k'
        return df_obs_info

    def get_otype_summary(self, otype):
        df_cmatch_simbad_otype = self.df_cmatch_simbad[self.df_cmatch_simbad['main_type'] == otype]
        df_cmatch_gaia_otype   = self.df_cmatch_gaia.loc[df_cmatch_simbad_otype.index]
        df_cmatch_om_otype     = self.df_cmatch_om.loc[df_cmatch_simbad_otype.index]
        df_cmatch_xmm_otype    = self.df_cmatch_xmm.loc[df_cmatch_simbad_otype.index]

        region_ids = [self.df_regions_unique.iloc[i]['idxs'] for i in df_cmatch_simbad_otype.index]
        lcs_one_plot = [self.plot_lcs_one_plot(reg_ids) for reg_ids in region_ids]

        # Unpack list of lists.
        # region_ids = [idx for sublist in region_ids for idx in sublist]
        # df_regions_to_plot = self.df_regions.iloc[region_ids]
        # print(f'Found {len(region_ids)} lightcurves in {len(df_cmatch_simbad_otype)} unique regions for {otype}')

        # lightcurves = self.get_and_plot_lcs_by_idxs(region_ids)

        content = {'otype'                  : otype,
                   'df_otype_stats'         : self.df_otype_stats,
                   'df_cmatch_simbad_otype' : df_cmatch_simbad_otype,
                   # 'lightcurves'            : lightcurves,
                   # 'df_regions_to_plot'     : df_regions_to_plot,
                   'lcs_one_plot'           : lcs_one_plot}
        return content

    def get_chime_summary(self):
        # Calculate seperation in arcsec, since TOPCAT seperation is some weird thing between 0 and 1.
        sc1 = SkyCoord(self.df_cmatch_chime['ra_deg'], self.df_cmatch_chime['dec_deg'], unit='deg')
        sc2 = SkyCoord(self.df_cmatch_chime['ra'], self.df_cmatch_chime['dec'], unit='deg')
        self.df_cmatch_chime['SEP_ARCSEC'] = sc1.separation(sc2).to('arcsec').value
        
        # region_num is the unique_region_id, because I crossmatched with the final catalogue.
        unique_region_ids = self.df_cmatch_chime.sort_values('SEP_ARCSEC')['region_num'].unique()
        region_ids = [self.df_regions_unique.iloc[i]['idxs'] for i in unique_region_ids]

        unique_regions = []
        for i, unique_region_id in enumerate(unique_region_ids):
            chime_matches = []
            df_cmatch_chime_sub = self.df_cmatch_chime[self.df_cmatch_chime['region_num'] == unique_region_id]
            for j, row in df_cmatch_chime_sub.iterrows():
                res2 = {'tns_name'   : row['tns_name'],
                        'ra'         : row['ra'],
                        'dec'        : row['dec'],
                        'ra_err'     : row['ra_err'],
                        'dec_err'    : row['dec_err'],
                        'ra_deg'     : round(row['ra_deg'], 4),   # EXOD region positions
                        'dec_deg'    : round(row['dec_deg'], 4),  # EXOD region positions
                        'Separation' : round(row['SEP_ARCSEC'], 2)}
                chime_matches.append(res2)

            res = {'unique_region_id' : unique_region_id,
                   'lc'               : self.plot_lcs_one_plot(region_ids[i]),
                   'n_chime_matches'  : len(chime_matches),
                   'chime_matches'    : chime_matches,
                   }
            unique_regions.append(res)
        content = {'unique_regions' : unique_regions}
        return content


    def get_lc_by_idx(self, idx):
        lc_idxs = self.df_lc_idx.iloc[idx]
        start, stop = lc_idxs['start'], lc_idxs['stop']
        df_lc = pd.read_hdf(savepaths_combined['lc'], start=start, stop=stop)
        df_lc['t0'] = df_lc['time'] - df_lc['time'].min()
        return df_lc

    def get_lcs_by_idxs(self, idxs):
        return [self.get_lc_by_idx(idx) for idx in idxs]

    def get_and_plot_lcs_by_idxs(self, region_ids):
        df_regions = self.df_regions.iloc[region_ids]
        flags      = self.get_all_flags(region_ids)
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

        dfs = self.get_lcs_by_idxs(region_ids)
        data_urls = [plot_lc(df_lc, label=plt_labels[i]) for i, df_lc in enumerate(dfs)]

        for i, r in enumerate(lightcurves):
            r['data_url'] = data_urls[i]
            r['flags']    = flags[i]

        return lightcurves

    def plot_lcs_one_plot(self, region_ids):
        df_lcs = self.get_lcs_by_idxs(region_ids)
        df_lc = pd.concat(df_lcs)
        df_lc = self.decode_runids(df_lc)
        df_lc = df_lc.sort_values(by='time')

        # get the starting times for each obsid. Add these to t0 to get the new shifted times.
        obsid_start_times = df_lc.groupby('obsid')['t0'].agg('max').cumsum().shift(1, fill_value=0)
        df_lc['t0_shifted'] = df_lc['t0'] + df_lc['obsid'].map(obsid_start_times)

        colors = {'0.2_2.0': 'r', '2.0_12.0': 'b', '0.2_12.0': 'k'}  # Color by energy

        unique_t_bins = np.sort(df_lc['t_bin'].unique())
        num_subplots = len(unique_t_bins)
        fig, ax = plt.subplots(num_subplots, 1, figsize=(11, 2 * num_subplots), sharex=True)

        if num_subplots == 1:
            ax = [ax]

        for i, t_bin in enumerate(unique_t_bins):
            df_lc_tbin = df_lc[df_lc['t_bin'] == t_bin]
            unique_E_combinations = df_lc[['E_lo', 'E_hi']].drop_duplicates()

            for obsid, start_time in obsid_start_times.items():
                ax[i].axvline(x=start_time, color='black', linestyle='-', lw=1.0)
                ax[i].text(start_time - 50,0, obsid, rotation=90, verticalalignment='bottom', fontsize=8, color='k',
                           bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

            for _, comb in unique_E_combinations.iterrows():
                E_lo = comb['E_lo']
                E_hi = comb['E_hi']
                df_lc_sub = df_lc_tbin[(df_lc_tbin['E_lo'] == E_lo) & (df_lc_tbin['E_hi'] == E_hi)]

                if len(df_lc_sub) == 0:
                    continue

                ax[i].step(df_lc_sub['t0_shifted'], df_lc_sub['n'], lw=1.0,
                           label=fr'$t_{{bin}}$={t_bin}s E = {E_lo}-{E_hi} keV', color=colors.get(f'{E_lo}_{E_hi}'))

            ax[i].legend()

        plt.tight_layout()
        fig.subplots_adjust(hspace=0.0, wspace=0)

        ax[0].set_xlim(0, df_lc['t0_shifted'].max())
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