from exod.xmm.observation import Observation
from exod.utils.plotting import cmap_image, get_image_limits

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.coordinates import SkyCoord
from exod.utils.plotting import cmap_image
from exod.utils.path import savepaths_combined
from exod.utils.plotting import get_image_limits

def plot_image(img, sc):
    x, y = skycoord_to_pixel(sc, img.wcs)

    fig = plt.figure(figsize=(10,10))
    ax = plt.subplot(projection=img.wcs)
    ax.set_title(img.filename)
    ax.imshow(img.data, norm=LogNorm(), cmap=cmap_image())
    ax.scatter(x, y, s=80, marker='o', facecolors='none', edgecolors='lime')
    xrange, yrange = get_image_limits(img.data)
    ax.set_xlim(xrange[0], xrange[1])
    ax.set_ylim(yrange[0], yrange[1])
    return fig

def plot_images(obsid, df_regions):
    obs = Observation(obsid)
    obs.get_files()

    sub = df_regions[df_regions['runid'].str.contains(obsid)]
    sc = SkyCoord(sub['ra_deg'], sub['dec_deg'], unit='deg')

    figs = []
    for img in obs.images:
        img.read()
        fig = plot_image(img, sc)
        figs.append(fig)
    return figs
        
