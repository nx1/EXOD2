import os
from pathlib import Path

base              = Path(os.environ['EXOD'])
data              = base / 'data'
utils             = base / 'exod' / 'utils'
data_raw          = data / 'raw'
data_processed    = data / 'processed'
data_results      = data / 'results'
data_combined     = data / 'results_combined'
data_catalogue    = data_combined / 'exod_catalogue'
data_simbad_stats = data_combined / 'simbad_stats'
data_util         = data / 'util'
data_plots        = data / 'plots'
logs              = base / 'logs'
docs              = base / 'docs'

all_paths = {
    'base'           : base,
    'data'           : data,
    'utils'          : utils,
    'data_raw'       : data_raw,
    'data_processed' : data_processed,
    'data_results'   : data_results,
    'data_combined'  : data_combined,
    'data_catalogue' : data_catalogue,
    'data_util'      : data_util,
    'data_plots'     : data_plots,
    'logs'           : logs,
    'docs'           : docs
}

savepaths_util = {'4xmm_dr14_slim'    : data_util / '4xmmdr14slim_240411.fits',
                  '4xmm_dr14_cat'     : data_util / '4XMM_DR14cat_v1.0.fits',
                  'GLADE+'            : data_util / 'GLADEP.fits',
                  'ExtraS'            : data_util / 'ExtraS.fits',
                  '4xmm_dr14_obslist' : data_util / '4xmmdr14_obslist.fits',
                  'CHIME_FRB'         : data_util / 'chimefrbcat1.fits'}

# TODO Remove merged_with_dr14 from savepaths_combined when finalized!
savepaths_combined = {'bti'             : data_combined / 'merged_with_dr14' / 'df_bti.csv',
                      'regions'         : data_combined / 'merged_with_dr14' / 'df_regions.csv',
                      'alerts'          : data_combined / 'merged_with_dr14' / 'df_alerts.csv',
                      'regions_unique'  : data_combined / 'merged_with_dr14' / 'df_regions_unique.csv',
                      'lc'              : data_combined / 'merged_with_dr14' / 'df_lc.h5',
                      'lc_idx'          : data_combined / 'merged_with_dr14' / 'df_lc_idx.csv',
                      'lc_features'     : data_combined / 'merged_with_dr14' / 'df_lc_features.csv',
                      'run_info'        : data_combined / 'merged_with_dr14' / 'df_run_info.csv',
                      'obs_info'        : data_combined / 'merged_with_dr14' / 'df_obs_info.csv',
                      'dl_info'         : data_combined / 'merged_with_dr14' / 'df_dl_info.csv',
                      'dc_info'         : data_combined / 'merged_with_dr14' / 'df_dc_info.csv',
                      'evt_info'        : data_combined / 'merged_with_dr14' / 'df_evt_info.csv',
                      'cmatch_simbad'   : data_combined / 'merged_with_dr14' / 'df_regions_unique_cmatch_simbad.csv',
                      'cmatch_gaia'     : data_combined / 'merged_with_dr14' / 'df_regions_unique_cmatch_gaia.csv',
                      'cmatch_om'       : data_combined / 'merged_with_dr14' / 'df_regions_unique_cmatch_om.csv',
                      'cmatch_dr14'     : data_combined / 'merged_with_dr14' / 'df_regions_unique_cmatch_dr14.csv',
                      'cmatch_glade'    : data_combined / 'merged_with_dr14' / 'df_regions_unique_cmatch_gladep.csv',
                      'cmatch_chime'    : data_combined / 'merged_with_dr14' / 'df_regions_unique_cmatch_chime.csv',
                      'exod_cat'        : data_catalogue / 'EXOD_DR1_cat.fits',
                      'exod_cat_unique' : data_catalogue / 'EXOD_DR1_cat_unique.fits'}

def create_all_paths():
    """Create all paths if they don't exist."""
    for name, path in all_paths.items():
        os.makedirs(path, exist_ok=True)


def check_file_exists(file_path, clobber=True):
    """
    Check if a file exists and raise FileExistsError if clobber is False.

    Parameters:
    - file_path (str or Path): The path to the file.
    - clobber (bool): If True, overwrite the file if it exists.

    Raises:
    - FileExistsError: If the file exists and clobber is False.
    """
    if not clobber and Path(file_path).exists():
        raise FileExistsError(f'File {file_path} exists and clobber={clobber}!')


def read_observation_ids(file_path):
    """
    Read observation IDs from file.
    Each line should be a single observation.
    """
    with open(file_path, 'r') as file:
        obs_ids = [line.strip() for line in file.readlines()]
    return obs_ids


if __name__ == "__main__":
    create_all_paths()
    for name, path in all_paths.items():
        exists = "exists" if path.exists() else "does not exist"
        print(f"{name:<15} : {path} : {exists}")



