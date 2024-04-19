from exod.post_processing.crossmatch import crossmatch_simbad, get_df_regions_no_crossmatch, \
    get_df_regions_with_crossmatch
from exod.utils.path import data_results
from exod.utils.logger import logger

import tarfile
from glob import glob
import re
from tqdm import tqdm
import pandas as pd
import astropy.units as u

from astroquery.exceptions import BlankResponseWarning
import warnings
warnings.simplefilter(action='ignore', category=BlankResponseWarning) # Ignore Non-results.


def read_all_csv_1d(glob_pattern):
    """Read all 1-d csvs, aka only 1 column of data."""
    dfs = []
    for f in tqdm(glob(glob_pattern)):
        df = pd.read_csv(f, index_col=0).T
        dfs.append(df)

    df_combined = pd.concat(dfs, ignore_index=True)
    return df_combined


def read_all_csv_regions(glob_pattern):
    regex = re.compile(r'\d{10}')
    dfs = []
    for f in tqdm(glob(glob_pattern)):
        obsid = regex.findall(f)[0]
        df = pd.read_csv(f)
        df['obsid'] = obsid
        df['subset'] = int(f.split('/')[-2][-1])
        dfs.append(df)

    df_regions = pd.concat(dfs, ignore_index=True)
    return df_regions


def combine_all_region_files():
    """
    Combine all df_region .csv files across all observations.

    Returns
    -------
    df_all_regions : pd.DataFrame : The combined dataframe of all the df_region files
    """
    logger.info('Combining all region files')
    csv_files = list(data_results.glob('*/*regions.csv'))
    logger.info(f'Found {len(csv_files)} csv files')

    dfs = []
    for csv in csv_files:
        df = pd.read_csv(csv, dtype={'obsid':str})
        logger.info(f'{csv} rows: {len(df)}')
        dfs.append(df)
    df_all_regions = pd.concat(dfs, axis=0).reset_index(drop=True)
    logger.info(f'df_all_regions:\n{df_all_regions}')
    return df_all_regions


def make_results_tarfile(output_filename):
    """
    Create results tarfile of exod results.

    Parameters
    ----------
    output_filename : filename without .tar.gz extension
    """
    csv_files   = list(data_results.glob('*/*/*.csv'))
    image_files = list(data_results.glob('*/*/*.png'))
    simlists    = list(data_results.glob('*simlist*'))
    print(f'Found #_csv={len(csv_files)} #_image_var={len(image_files)}')

    tarfile_savepath = data_combined / f'{output_filename}.tar.gz'
    with tarfile.open(str(tarfile_savepath), mode='w|gz') as tf:
        for f in tqdm([*csv_files, *image_files, *simlists]):
            arcname = f.relative_to(data_results)
            print(f'{f} --> {arcname}')
            tf.add(f, arcname=arcname)
    print(f'tarfile save to: {tarfile_savepath}')

if __name__ == '__main__':
    from exod.utils.path import data_combined
    df_all_regions = combine_all_region_files()
    tab_res = crossmatch_simbad(df_region=df_all_regions, radius=1 * u.arcmin)
    df_all_regions_no_crossmatch = get_df_regions_no_crossmatch(df_regions=df_all_regions, tab_res=tab_res)
    df_all_regions_with_crossmatch = get_df_regions_with_crossmatch(df_regions=df_all_regions, tab_res=tab_res)

    # Count and plot OTYPES
    df_otypes = tab_res.to_pandas().value_counts('OTYPE')
    logger.info(f'{df_otypes}\ndf_otypes')
    df_otypes.sort_values().plot(figsize=(5,10.0), kind='barh', title='SIMBAD Crossmatch')
