from exod.post_processing.cluster_regions import ClusterRegions
from exod.post_processing.filter_subsets import SubsetManager, Subset, generate_valid_combinations, get_filters, generate_valid_combinations
from exod.utils.path import data, savepaths_combined

import base64
import io
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, url_for, request, redirect
import astropy.units as u



dtype = {'obsid' : str}
csv_kwargs = {'dtype':dtype}
#             'nrows': 1000}

df_dl            = pd.read_csv(savepaths_combined['dl_info'], **csv_kwargs)
df_cmatch_simbad = pd.read_csv(savepaths_combined['cmatch_simbad'], **csv_kwargs)
df_cmatch_om     = pd.read_csv(savepaths_combined['cmatch_om'], **csv_kwargs)
df_cmatch_gaia   = pd.read_csv(savepaths_combined['cmatch_gaia'], **csv_kwargs)
df_cmatch_xmm    = pd.read_csv(savepaths_combined['cmatch_dr14'], **csv_kwargs)
df_dc            = pd.read_csv(savepaths_combined['dc_info'], **csv_kwargs)
df_evt           = pd.read_csv(savepaths_combined['evt_info'], index_col='obsid', **csv_kwargs)
df_obs           = pd.read_csv(savepaths_combined['obs_info'], **csv_kwargs)
df_run           = pd.read_csv(savepaths_combined['run_info'], index_col='obsid', **csv_kwargs)
df_sim           = pd.read_csv('/home/nkhan/EXOD2/data/results_combined/merged_with_dr14/EXOD_simlist.csv', **csv_kwargs)
df_lc_idx        = pd.read_csv(savepaths_combined['lc_idx'], index_col='Unnamed: 0', **csv_kwargs)
df_lc_features   = pd.read_csv(savepaths_combined['lc_features'], **csv_kwargs)
df_regions       = pd.read_csv(savepaths_combined['regions'], **csv_kwargs)
df_otype_stats   = pd.read_csv('/home/nkhan/EXOD2/data/results_combined/simbad_stats/EXOD FULL_otype_stats.csv')
#df_lc            = pd.read_csv(savepaths_combined['lc'], **csv_kwargs)

cr = ClusterRegions(df_regions)
df_sources_unique = cr.df_regions_unique
assert len(df_cmatch_simbad) == len(df_sources_unique)

df_regions['sigma_max_B_peak']    = df_lc_features['sigma_max_B_peak']      
df_regions['sigma_max_B_eclipse'] = df_lc_features['sigma_max_B_eclipse']   
df_regions['DR14_SEP_ARCSEC']     = df_regions['cluster_label'].map(df_cmatch_xmm['SEP_ARCSEC'])
df_regions['SIMBAD_SEP_ARCSEC']   = df_regions['cluster_label'].map(df_cmatch_simbad['SEP_ARCSEC'])
                                                                        
filters = get_filters()                                                 
valid_combinations = generate_valid_combinations(*filters)              
sm = SubsetManager()                                                    
sm.add_subsets([Subset(f, df_regions) for f in valid_combinations])     
sm.calc_all()                                                           
                                                                         

app = Flask(__name__, static_folder=data, static_url_path='/static')



@app.route('/')
def main_page():
    content = {}
    return render_template("obs_list.html", content=content)

def get_lc_by_idx(idx):
    lc_idxs = df_lc_idx.iloc[idx]
    start, stop = lc_idxs['start'], lc_idxs['stop']
    df_lc = pd.read_hdf(savepaths_combined['lc'], start=start, stop=stop)
    return df_lc

def plot_lc(df_lc, label):
    df_lc['t0'] = df_lc['time'] - df_lc['time'].min()
    fig = plt.Figure(figsize=(15, 2))
    ax = fig.subplots()
    ax.step(df_lc['t0'], df_lc['n'], lw=1.0, color='black', label=label)
    ax.step(df_lc['t0'], df_lc['mu'], color='red', lw=1.0)
    ax.set_ylabel('Counts')
    ax.set_xlim(0, df_lc['t0'].max())
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    lightcurve_data_url = base64.b64encode(buf.read()).decode('ascii')
    return lightcurve_data_url

@app.route('/region/<region_id>')
def region_summary(region_id):
    region_id = int(region_id)
    next_id = region_id + 1
    prev_id = region_id - 1

    region_idxs = df_sources_unique.iloc[region_id]['idxs'] # Indexs that the unique region is associated with
    reg_info    = df_regions.iloc[region_idxs[0]]              # Get the first index
    obsid, label, t_bin, E_lo, E_hi = reg_info['runid'].split('_')

    evt_info           = df_evt.loc[obsid].iloc[0]
    cmatch_simbad_info = df_cmatch_simbad.iloc[region_id]
    cmatch_gaia_info   = df_cmatch_gaia.iloc[region_id]
    cmatch_om_info     = df_cmatch_om.iloc[region_id]
    xmm_info           = df_cmatch_xmm.iloc[region_id]
    run_info           = df_run.loc[obsid]

    lightcurves = []
    for idx in region_idxs:
        runid = df_regions.iloc[idx]['runid']
        label = df_regions.iloc[idx]['label']
        key = str((runid, str(label)))

        df_lc = get_lc_by_idx(idx)
        label = f'reg_id={idx} key={key}'
        lightcurve_data_url = plot_lc(df_lc, label=label)
        lightcurves.append({'data_url' : lightcurve_data_url})

    content = {'region_id'        : region_id,
               'region_idxs'      : region_idxs,
               'next_id'          : next_id,
               'prev_id'          : prev_id,
               'reg_info'         : reg_info,
               'evt_info'         : evt_info,
               'run_info'         : run_info,
               'cmatch_info'      : cmatch_simbad_info,
               'cmatch_gaia_info' : cmatch_gaia_info,
               'cmatch_om_info'   : cmatch_om_info,
               'xmm_info'         : xmm_info,
               'obsid'            : obsid,
               'label'            : label,
               't_bin'            : t_bin,
               'E_lo'             : E_lo,
               'E_hi'             : E_hi,
               'lightcurves'      : lightcurves}
    return render_template('region_summary.html', content=content)

@app.route('/otype/<otype>')
def otype(otype):
    df_cmatch_simbad_otype = df_cmatch_simbad[df_cmatch_simbad['main_type'] == otype]
    df_cmatch_gaia_otype   = df_cmatch_gaia.loc[df_cmatch_simbad_otype.index]
    df_cmatch_om_otype     = df_cmatch_om.loc[df_cmatch_simbad_otype.index]
    df_cmatch_xmm_otype    = df_cmatch_xmm.loc[df_cmatch_simbad_otype.index]

    idxs = [df_sources_unique.iloc[i]['idxs'] for i in df_cmatch_simbad_otype.index]
    idxs = [idx for sublist in idxs for idx in sublist]

    df_regions_to_plot = df_regions.iloc[idxs]
    print(f'Found {len(idxs)} lightcurves in {len(df_cmatch_simbad_otype)} unique regions for {otype}')

    lightcurves = []
    for idx in tqdm(idxs):
        df_lc = get_lc_by_idx(idx)
        reg = df_regions.iloc[idx]
        ra         = reg['ra_deg']
        dec        = reg['dec_deg']
        region_id  = reg['cluster_label']
        label = f'reg_id={idx} unique_id={region_id} ra={ra:.2f} dec={dec:.2f}'
        lightcurve_data_url = plot_lc(df_lc, label)
        lightcurves.append({'data_url': lightcurve_data_url, 'region_id': region_id})

    content = {'otype'                  : otype,
               'df_otype_stats'         : df_otype_stats,
               'df_cmatch_simbad_otype' : df_cmatch_simbad_otype,
               'lightcurves'            : lightcurves,
               'df_regions_to_plot'     : df_regions_to_plot}

    return render_template('otype.html', content=content)

@app.route('/obs/<obsid>')
def observation_page(obsid):
    evt_info = df_evt.loc[obsid].iloc[0]
    run_info = df_run.loc[obsid]

    # Get regions
    df_regions_obs = df_regions[df_regions['runid'].str.contains(obsid)]
    tab_regions_obs = df_regions_obs[['runid', 'label', 'ra', 'dec', 'ra_deg', 'dec_deg']].to_html(table_id='myTable', classes='display compact')

    # Get Lightcurves
    lightcurves = []
    mask = df_lc_idx.index.str.contains(obsid)
    idxs = np.where(mask)[0]
    df_lc_idx_obs = df_lc_idx[mask]

    i = 0
    for key, row in tqdm(df_lc_idx_obs.iterrows()):
        df_lc = pd.read_hdf(savepaths_combined['lc'], start=row['start'], stop=row['stop'])
        unique_reg_id = cr.region_num_to_cluster_num[idxs[i]]
        label = f'key={key} reg_id={idxs[i]} unique_reg_id={unique_reg_id}'
        lightcurve_data_url = plot_lc(df_lc, label)
        lightcurves.append({'data_url'  : lightcurve_data_url,
                            'region_id' : unique_reg_id})
        i+=1

    content = {'obsid'           : obsid,
               'tab_regions_obs' : tab_regions_obs,
               'evt_info'        : evt_info,
               'run_info'        : run_info,
               'lightcurves'     : lightcurves}
    return render_template('observation_page.html', content=content)

@app.route('/subsets')
def subsets():
    return render_template('subsets.html', subsets=sm.subsets)

@app.route('/subsets/<subset_num>')
def show_subset(subset_num):
    subset = sm.get_subset_by_index(int(subset_num))
    # sort by maximum lightcurve count.
    df_lc_features_subset = df_lc_features.loc[subset.df.index]
    df_lc_features_subset = df_lc_features_subset.sort_values('n_max', ascending=False)

    lightcurves = []
    for idx in tqdm(df_lc_features_subset.index):
        df_lc = get_lc_by_idx(idx)
        unique_reg_id = cr.region_num_to_cluster_num[idx]
        label = f'reg_id={idx} unique_reg_id={unique_reg_id}'
        lightcurve_data_url = plot_lc(df_lc, label)
        lightcurves.append({'data_url' : lightcurve_data_url, 'region_id' : unique_reg_id})

    content = {'lightcurves' : lightcurves,
               'subset'      : subset}

    return render_template('subset_page.html', content=content)

if __name__ == "__main__":
    app.run(debug=False)
