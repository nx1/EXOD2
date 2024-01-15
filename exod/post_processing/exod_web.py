import re
import pandas as pd
from flask import Flask, render_template
from exod.utils import path
from exod.post_processing.collate_results import combine_all_region_files

app = Flask(__name__)


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
    df_all_regions = combine_all_region_files()
    tab_reg = df_all_regions.to_html(classes='table table-striped table-hover')

    content   = {'obsid' : obsid,
                 'tab_reg' : tab_reg}

    return render_template('observation_page.html',
                           content=content)

@app.route('/all_detected_regions')
def all_detected_regions():
    df_all_regions = combine_all_region_files()
    tab_reg = df_all_regions.to_html(classes='table table-striped table-hover')

    content   = {'tab_reg' : tab_reg}

    return render_template('all_detected_regions.html',
                           content=content)


    print(images)
    content = {'images':images}
    return render_template("all_var_img.html", content=content)


@app.route('/all_var_img')
def all_var_img_page():
    images = list(path.data_results.glob('*/*var_img.png'))
    images = [str(i) for i in images]


    pattern = re.compile(r'\d{10}')
    obsids = [re.search(pattern, s).group(0) for s in images]
    print(obsids)


    print(images)
    content = {'images':images}
    return render_template("all_var_img.html", content=content)


if __name__ == "__main__":
    app.run(debug=True)
