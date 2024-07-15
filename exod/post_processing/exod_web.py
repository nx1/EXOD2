from exod.post_processing.crossmatch import crossmatch_dr14_slim
from exod.post_processing.cluster_regions import cluster_regions, get_unique_regions
from exod.utils.path import data, data_results, savepaths_combined

import base64
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io
from flask import Flask, render_template, url_for, request, redirect
import astropy.units as u


dtype = {'obsid' : str}
csv_kwargs = {'dtype':dtype}
#              'nrows': 1000}

df_dl            = pd.read_csv(savepaths_combined['dl_info'], **csv_kwargs)
df_cmatch_simbad = pd.read_csv(savepaths_combined['cmatch_simbad'], **csv_kwargs)
df_cmatch_om     = pd.read_csv(savepaths_combined['cmatch_om'], **csv_kwargs)
df_cmatch_gaia   = pd.read_csv(savepaths_combined['cmatch_gaia'], **csv_kwargs)
df_dc            = pd.read_csv(savepaths_combined['dc_info'], **csv_kwargs)
df_evt           = pd.read_csv(savepaths_combined['evt_info'], index_col='obsid', **csv_kwargs)
df_obs           = pd.read_csv(savepaths_combined['obs_info'], **csv_kwargs)
df_run           = pd.read_csv(savepaths_combined['run_info'], index_col='obsid', **csv_kwargs)
df_sim           = pd.read_csv('/home/nkhan/EXOD2/data/results_combined/merged_with_dr14/EXOD_simlist.csv', **csv_kwargs)
#df_lc            = pd.read_csv(savepaths_combined['lc'], **csv_kwargs)
df_lc_idx        = pd.read_csv(savepaths_combined['lc_idx'], index_col='Unnamed: 0', **csv_kwargs)
df_regions       = pd.read_csv(savepaths_combined['regions'], **csv_kwargs)
df_sources_unique = get_unique_regions(df_regions, 20 * u.arcsec)
print(df_sources_unique)


tab_cmatch_xmm    = crossmatch_dr14_slim(df_sources_unique)
df_cmatch_xmm     = tab_cmatch_xmm.to_pandas()

df_class = pd.DataFrame()
df_class['key'] = df_sources_unique.index
df_class['class'] = None
df_class['comments'] = None
df_class = df_class.set_index('key')

app = Flask(__name__, static_folder=data, static_url_path='/static')

@app.route('/')
def main_page():
    content = {}
    return render_template("obs_list.html", content=content)


@app.route('/regions')
def regions():
    tab_reg = df_regions.to_html(table_id='myTable', classes='display compact')
    content   = {'tab' : tab_reg}
    return render_template('table_template.html', content=content)


@app.route('/unique_regions')
def unique_regions():
    tab_reg = df_sources_unique.to_html(table_id='myTable', classes='display compact')
    content   = {'tab' : tab_reg}
    return render_template('table_template.html', content=content)


@app.route('/lightcurve_idx')
def lightcurve_idx():
    tab_lc_idx = df_lc_idx.to_html(table_id='myTable', classes='display compact')
    content = {'tab':tab_lc_idx}
    return render_template("table_template.html", content=content)


@app.route('/simlist')
def simlist():
    tab_sim = df_sim.to_html(table_id='myTable', classes='display compact')
    content = {'tab':tab_sim}
    return render_template("table_template.html", content=content)


@app.route('/run_info')
def run_info():
    tab_run = df_run.to_html(table_id='myTable', classes='display compact')
    content = {'tab':tab_run}
    return render_template("table_template.html", content=content)


@app.route('/obs_info')
def obs_info():
    tab_obs = df_obs.to_html(table_id='myTable', classes='display compact')
    content = {'tab':tab_obs}
    return render_template("table_template.html", content=content)


@app.route('/evt_info')
def evt_info():
    tab_evt = df_evt.to_html(table_id='myTable', classes='display compact')
    content = {'tab':tab_evt}
    return render_template("table_template.html", content=content)


@app.route('/dl_info')
def dl_info():
    tab_dl = df_dl.to_html(table_id='myTable', classes='display compact')
    content = {'tab':tab_dl}
    return render_template("table_template.html", content=content)


@app.route('/dc_info')
def dc_info():
    tab_dc = df_dc.to_html(table_id='myTable', classes='display compact')
    content = {'tab':tab_dc}
    return render_template("table_template.html", content=content)


@app.route('/crossmatch_simbad')
def crossmatch_simbad():
    tab_cmatch = df_cmatch_simbad.to_html(table_id='myTable', classes='display compact')
    content = {'tab':tab_cmatch}
    return render_template("table_template.html", content=content)


@app.route('/crossmatch_xmm')
def crossmatch_xmm():
    tab_cmatch_xmm = df_cmatch_xmm.to_html(table_id='myTable', classes='display compact')
    content = {'tab':tab_cmatch_xmm}
    return render_template("table_template.html", content=content)


@app.route('/crossmatch_om')
def crossmatch_om():
    tab_cmatch_xmm = df_cmatch_om.to_html(table_id='myTable', classes='display compact')
    content = {'tab':tab_cmatch_xmm}
    return render_template("table_template.html", content=content)


@app.route('/crossmatch_gaia')
def crossmatch_gaia():
    tab_cmatch_xmm = df_cmatch_gaia.to_html(table_id='myTable', classes='display compact')
    content = {'tab':tab_cmatch_xmm}
    return render_template("table_template.html", content=content)


@app.route('/region/<region_id>')
def region_summary(region_id):
    region_id = int(region_id)
    next_id = region_id + 1
    prev_id = region_id - 1

    region_idxs = df_sources_unique.iloc[region_id]['idxs'] # Indexs that the unique region is associated with
    reg_info = df_regions.iloc[region_idxs[0]]              # Get the first index
    obsid, label, t_bin, E_lo, E_hi = reg_info['runid'].split('_')

    evt_info    = df_evt.loc[obsid].iloc[0]
    cmatch_simbad_info = df_cmatch_simbad.iloc[region_id]
    cmatch_gaia_info = df_cmatch_gaia.iloc[region_id]
    cmatch_om_info = df_cmatch_om.iloc[region_id]
    xmm_info = df_cmatch_xmm.iloc[region_id]
    run_info = df_run.loc[obsid]

    lightcurves = []
    for i in range(len(region_idxs)):
        idx = region_idxs[i]
        runid = df_regions.iloc[idx]['runid']
        label = df_regions.iloc[idx]['label']
        key   = str((runid, str(label)))
        lc_idxs = df_lc_idx.loc[key]
        start, stop = lc_idxs['start'], lc_idxs['stop']
        df_lc = pd.read_hdf(savepaths_combined['lc'], start=start, stop=stop)

        lightcurve_data_url = plot_lc(df_lc, key)
        lightcurves.append({'data_url':lightcurve_data_url})
        class_info = df_class.loc[region_id]

    content = {'region_id'   : region_id,
               'region_idxs' : region_idxs,
               'next_id'     : next_id,
               'prev_id'     : prev_id,
               'reg_info'    : reg_info,
               'evt_info'    : evt_info,
               'run_info'    : run_info,
               'cmatch_info' : cmatch_simbad_info,
               'cmatch_gaia_info' : cmatch_gaia_info,
               'cmatch_om_info' : cmatch_om_info,
               'xmm_info'    : xmm_info,
               'obsid'       : obsid,
               'label'       : label,
               't_bin'       : t_bin,
               'E_lo'        : E_lo,
               'E_hi'        : E_hi,
               'lightcurves' : lightcurves,
               'class_info'  : class_info}
    return render_template('region_summary.html', content=content)

def plot_lc(df_lc, key):
    fig = plt.Figure(figsize=(15, 2))
    ax = fig.subplots()
    ax.plot(df_lc['time'], df_lc['n'], lw=1.0, color='black', label=key)
    ax.plot(df_lc['time'], df_lc['mu'], color='red', lw=1.0)
    ax.set_ylabel('Counts')
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    lightcurve_data_url = base64.b64encode(buf.read()).decode('ascii')
    return lightcurve_data_url


@app.route('/handle_data', methods=['POST'])
def handle_data():
    button = request.form['button']
    region_id = int(request.form['region_id'])
    classifications = ["unsure", "interesting", "agn", "burst", "star", "xrb", "cv", "junk"]
    if button in classifications:
        df_class.at[region_id, 'class'] = button

    if button == 'save_class':
        savepath = '/home/nkhan/EXOD2/data/results_combined/merged_with_dr14/df_class.csv'
        print(f'Saving to {savepath}')
        df_class.to_csv(savepath)
    return redirect(url_for('region_summary', region_id=region_id+1))


@app.route('/obs/<obsid>')
def observation_page(obsid):

    # Get Event Information
    evt_info = df_evt.loc[obsid].iloc[0]
    run_info = df_run.loc[obsid]

    # Get regions
    df_regions_obs = df_regions[df_regions['runid'].str.contains(obsid)]
    tab_regions_obs = df_regions_obs[['runid', 'label', 'ra', 'dec', 'ra_deg', 'dec_deg']].to_html(table_id='myTable', classes='display compact')

    # Get Lightcurves
    lightcurves = []
    df_lc_idx_obs = df_lc_idx[df_lc_idx.index.str.contains(obsid)]
    for key, row in df_lc_idx_obs.iterrows():
        df_lc = pd.read_hdf(savepaths_combined['lc'], start=row['start'], stop=row['stop'])
        lightcurve_data_url = plot_lc(df_lc, key)
        lightcurves.append({'data_url' : lightcurve_data_url})

    content = {'obsid'           : obsid,
               'tab_regions_obs' : tab_regions_obs,
               'evt_info'        : evt_info,
               'run_info'        : run_info,
               'lightcurves'     : lightcurves}
    print(content)
    return render_template('observation_page.html', content=content)


if __name__ == "__main__":
    app.run(debug=True)
