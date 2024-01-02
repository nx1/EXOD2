import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os
from exod.pre_processing.read_events_files import read_EPIC_events_file
from exod.utils.path import data_processed
from exod.utils.synthetic_data import create_fake_burst
from cv2 import inpaint, INPAINT_NS, filter2D
from scipy.stats import poisson
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

def BTIs_vs_GTIs_energy_bands(tab_energy_low, tab_energy_high):
    size_arcsec, time_interval = 15, 100
    fig, axes = plt.subplots(3,2)
    fig.set_figheight(15)
    fig.set_figwidth(10)
    for ind, emin, emax in zip(range(len(tab_energy_high)),tab_energy_low,tab_energy_high):
        cube, coordinates_XY, rejected = read_EPIC_events_file('0831790701', size_arcsec, time_interval,
                                                       gti_only=False, emin=emin, emax=emax)
        image_total = np.sum(cube, axis=2)
        threshold=np.nanpercentile(image_total.flatten(), 95) #This or from detected sources
        condition = np.where(image_total>threshold)
        mask = np.zeros(image_total.shape)
        mask[condition]=1
        mask=np.uint8(mask[:,:,np.newaxis])

        kept = [ind for ind in range(cube.shape[2]) if ind not in rejected]
        image_GTI = np.sum(cube[:,:,kept], axis=2)
        no_source_image_GTI = inpaint(image_GTI.astype(np.float32)[:,:,np.newaxis], mask, 5, flags=INPAINT_NS)
        axes[ind][0].imshow(image_GTI, interpolation='none',norm=LogNorm())
        axes[ind][0].set_title(f'GTI {emin}-{emax} keV')

        image_BTI = np.sum(cube[:,:,rejected], axis=2)
        no_source_image_BTI = inpaint(image_BTI.astype(np.float32)[:,:,np.newaxis], mask, 5, flags=INPAINT_NS)
        axes[ind][1].imshow(image_BTI, interpolation='none',norm=LogNorm())
        axes[ind][1].set_title(f'BTI {emin}-{emax} keV')
    plt.savefig(os.path.join(data_processed, '0831790701', f"TestSpectralDependence.png"))

if __name__=='__main__':
    BTIs_vs_GTIs_energy_bands([0.2,1,5], [1,5,12])
