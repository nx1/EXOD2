from exod.post_processing.plot_detection_images import plot_images
from exod.post_processing.results_manager import ResultsManager
from exod.utils.path import data
from exod.post_processing.ds9 import view_obs_images

import base64
import io

import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template

rm = ResultsManager()
rm.load_results()
sm = rm.subset_manager
cr = rm.cr

app = Flask(__name__, static_folder=data, static_url_path='/static')

@app.route('/')
def main_page():
    content = {}
    return render_template("obs_list.html", content=content)


@app.route('/region/<region_id>')
def region_summary(region_id):
    content = rm.get_region_summary(region_id)
    return render_template('region_summary.html', content=content)


@app.route('/otype/<otype>')
def otype(otype):
    content = rm.get_otype_summary(otype)
    return render_template('otype.html', content=content)


@app.route('/obs/<obsid>')
def observation_page(obsid):
    content = rm.get_observation_summary(obsid)
    return render_template('observation_page.html', content=content)


@app.route('/subsets')
def subsets():
    return render_template('subsets.html', subsets=sm.subsets)


@app.route('/subsets/<subset_num>')
def show_subset(subset_num):
    content = rm.get_subset_summary(subset_num)
    return render_template('subset_page.html', content=content)

@app.route('/img/<obsid>')
def plot_detection_images(obsid):
    figs = plot_images(obsid, rm.df_regions)
    images = []
    for fig in figs:
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        data_url = base64.b64encode(buf.read()).decode('ascii')
        images.append(data_url)
    content = {'images' : images}
    return render_template('plot_detection_images.html', content=content)

@app.route('/ds9/<obsid>/<region_id>')
def show_ds9(obsid, region_id):
    region_id = int(region_id)
    ra = rm.df_regions.iloc[region_id]['ra_deg']
    dec = rm.df_regions.iloc[region_id]['dec_deg']
    view_obs_images(obsid=obsid, ra=ra, dec=dec)

if __name__ == "__main__":
    app.run(debug=False)
