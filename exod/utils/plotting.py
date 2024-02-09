import matplotlib
from matplotlib import pyplot as plt


def set_latex_font():
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'


def cmap_image():
    cmap = matplotlib.colormaps['hot']
    cmap.set_bad('black')
    return cmap
