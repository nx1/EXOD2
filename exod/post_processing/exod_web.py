import pandas as pd
from flask import Flask, render_template
from exod.utils import path
from exod.post_processing.collate_results import combine_all_region_files

app = Flask(__name__)


@app.route('/')
def main_page():
    sim_list = path.data_results / 'EXOD_simlist_03_01_24-210941.csv'
    df = pd.read_csv(sim_list, dtype={'obsid':str})
    return render_template("obs_list.html",
            table=df.to_html(classes='table table-striped table-hover'))


@app.route('/obs/<obsid>')
def observation_page(obsid):
    df_all_regions = combine_all_region_files()
    return render_template('observation_page.html',
                           obsid=obsid)

if __name__ == "__main__":
    app.run(debug=True)
