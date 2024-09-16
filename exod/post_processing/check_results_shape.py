from numpy import save
import pandas as pd

from exod.utils.path import savepaths_combined

def check_results_shape():
    """Check if the length of the results are consistent."""
    shapes = {}
    for name, path in savepaths_combined.items():
        if name == 'lc':
            continue
        df = pd.read_csv(path)
        shape = df.shape
        shapes[name] = shape
        print(f'{name:<15}: {shape}')

    assert shapes['regions_unique'][0] == shapes['cmatch_simbad'][0]
    assert shapes['regions_unique'][0] == shapes['cmatch_gaia'][0]
    assert shapes['regions_unique'][0] == shapes['cmatch_om'][0]
    assert shapes['regions_unique'][0] == shapes['cmatch_dr14'][0]

check_results_shape()
