from exod.post_processing.results_manager import ResultsManager
from exod.utils.path import data
from exod.post_processing.ds9 import view_obs_images

import base64
import io

import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template

rm = ResultsManager()

app = Flask(__name__, static_folder=data, static_url_path='/static')

@app.route('/')
def main_page():
    content = rm.get_homepage_summary()
    return render_template("homepage.html", content=content)

@app.route('/region/<region_id>')
def region_summary(region_id):
    content = rm.get_unique_region_summary(region_id)
    return render_template('region_summary.html', content=content)

@app.route('/otype/<otype>')
def otype(otype):
    content = rm.get_otype_summary(otype)
    return render_template('otype.html', content=content)

@app.route('/chime')
def chime():
    content = rm.get_chime_summary()
    return render_template('chime.html', content=content)

@app.route('/obs/<obsid>')
def observation_page(obsid):
    content = rm.get_observation_summary(obsid)
    return render_template('observation_page.html', content=content)

@app.route('/subsets')
def subsets():
    return render_template('subsets.html', subsets=rm.subset_manager.subsets)

@app.route('/subsets/<subset_num>')
def show_subset(subset_num):
    content = rm.get_subset_summary(subset_num)
    return render_template('subset_page.html', content=content)

@app.route('/img/<obsid>')
def plot_detection_images(obsid):
    content = rm.get_observation_image_summary(obsid)
    return render_template('plot_detection_images.html', content=content)

@app.route('/ds9/<obsid>/<region_id>')
def show_ds9(obsid, region_id):
    region_id = int(region_id)
    ra = rm.df_regions.iloc[region_id]['ra_deg']
    dec = rm.df_regions.iloc[region_id]['dec_deg']
    view_obs_images(obsid=obsid, ra=ra, dec=dec)

if __name__ == "__main__":
    app.run(debug=False)
