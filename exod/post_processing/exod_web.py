import re
import pandas as pd
from flask import Flask, render_template, url_for
from exod.utils import path
from exod.post_processing.collate_results import combine_all_region_files
from exod.utils.path import data, data_results

app = Flask(__name__, static_folder=data, static_url_path='/static')


@app.route('/')
def main_page():
    sim_lists = list(path.data_results.glob('*simlist*csv'))
    print(sim_lists)
    sim_list = sim_lists[0]
    df = pd.read_csv(sim_list, dtype={'obsid':str})
    table = df.to_html(classes='table table-striped table-hover')

    content = {'sim_list' : sim_list,
               'table'    : table}

    return render_template("obs_list.html",
            content=content)


@app.route('/obs/<obsid>')
def observation_page(obsid):
    obs_path = data_results / obsid

    # Get Region File
    try:
        region_file = list(obs_path.glob('*region*'))[0]
        df_region = pd.read_csv(region_file)
        tab_reg = df_region.to_html(classes='table table-striped table-hover')
    except IndexError:
        print('No Region File Found!')
        tab_reg = None



    def get_img_path(glob_pattern):
        """Get the /static/ image path."""
        try:
            img = list(obs_path.glob(glob_pattern))[0]
            img = img.relative_to(data)
            img = str(img)
            img = url_for('static', filename=img)
        except IndexError:
            print(f'Could not find {glob_pattern}')
            return None
        return img

    # Get Variability Image
    var_img_file    = get_img_path('*var_img*')
    cmatch_img_file = get_img_path('*SIMBAD*')
    bti_plot        = get_img_path('*bti_plot*')

    content   = {'obsid'   : obsid,
                 'tab_reg' : tab_reg,
                 'var_img' : var_img_file,
                 'cmatch_img' : cmatch_img_file,
                 'bti_plot' : bti_plot}

    return render_template('observation_page.html',
                           content=content)


@app.route('/all_detected_regions')
def all_detected_regions():
    df_all_regions = combine_all_region_files()
    tab_reg = df_all_regions.to_html(classes='table table-striped table-hover')

    content   = {'tab_reg' : tab_reg}

    return render_template('all_detected_regions.html',
                           content=content)


@app.route('/all_var_img')
def all_var_img_page():
    images = list(path.data_results.glob('*/*var_img.png'))
    images = [f.relative_to(data) for f in images]
    images = [str(i) for i in images]
    images = [url_for('static', filename=f) for f in images]

    pattern = re.compile(r'\d{10}')
    obsids = [re.search(pattern, s).group(0) for s in images]
    n = len(images)

    print(images)
    content = {'images':images,
               'obsids':obsids,
               'n':n}
    return render_template("all_var_img.html", content=content)


@app.route('/all_simbad_crossmatch')
def all_simbad_crossmatch():
    images = list(path.data_results.glob('*/*SIMBAD*.png'))
    images = [f.relative_to(data) for f in images]
    images = [str(i) for i in images]
    images = [url_for('static', filename=f) for f in images]

    pattern = re.compile(r'\d{10}')
    obsids = [re.search(pattern, s).group(0) for s in images]
    n = len(images)

    print(images)
    content = {'images':images,
               'obsids':obsids,
               'n':n}
    return render_template("all_simbad_crossmatch.html", content=content)


if __name__ == "__main__":
    app.run(debug=True)
