import numpy as np
import matplotlib
from matplotlib import pyplot as plt, pyplot
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm, ListedColormap
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.stats import gaussian_kde
import cmasher

from exod.utils.logger import logger


def use_scienceplots():
    import scienceplots
    plt.style.use('science')

def set_latex_font():
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'


def cmap_image():
    cmap = matplotlib.colormaps['hot']
    cmap.set_bad('black')
    return cmap


def plot_image(image_arr, title, log=False):
    plt.figure(figsize=(7,7))
    plt.title(title)
    norm = None
    if log:
        norm = LogNorm()

    plt.imshow(image_arr, norm=norm, cmap=cmap_image(), interpolation='none')
    plt.colorbar(shrink=0.75)
    plt.tight_layout()
    plt.show()


def compare_images(images, titles, log=False, plot=False):
    if not plot:
        return None

    N_images = len(images)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title(f'images[0]={titles[0]}')
    norm = None
    if log:
        norm = LogNorm()

    im = ax.imshow(images[0], norm=norm, cmap=cmap_image(), interpolation='none')
    plt.colorbar(im, ax=ax, shrink=0.75)
    plt.tight_layout()

    def update(frame):
        i = frame % N_images
        im.set_array(images[i])
        ax.set_title(f'images[{i}]={titles[i]}')

    ani = FuncAnimation(fig, update, frames=range(N_images), interval=1000)  # Adjust frames and interval as needed
    plt.show()


def plot_frame_masks(instrum, masks, labels, plot=False):
    if not plot:
        return None
    cmap = ListedColormap([[1, 0, 0], [0, 1, 0]])
    mask_stack = np.vstack(masks)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_title(f'{instrum} | Frame Masks | Green=True, Red=False')
    ax.imshow(mask_stack, cmap=cmap, aspect='auto', interpolation='none')
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Frame Number')
    plt.tight_layout()
    plt.show()


def plot_3d_image(image):
    """Plot an image as a 3D surface"""
    xx, yy = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(xx, yy, image, rstride=1, cstride=1, cmap='plasma', linewidth=0)  # , antialiased=False

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_zticks([])

    # ax.set_zlim(0,100)ax.set_zticks([])

    plt.tight_layout()

    plt.show()


def plot_cube_statistics(data):
    cube = data
    logger.info('Calculating and plotting data cube statistics...')
    image_max = np.nanmax(cube, axis=2)
    image_min = np.nanmin(cube, axis=2)  # The Minimum and median are basically junk
    image_median = np.nanmedian(cube, axis=2)
    image_mean = np.nanmean(cube, axis=2)
    image_std = np.nanstd(cube, axis=2)
    image_sum = np.nansum(cube, axis=2)

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    # Plotting images
    cmap = cmap_image()
    im_max = ax[0, 0].imshow(image_max.T, interpolation='none', origin='lower', cmap=cmap)
    im_min = ax[0, 1].imshow(image_min.T, interpolation='none', origin='lower', cmap=cmap)
    im_mean = ax[1, 0].imshow(image_mean.T, interpolation='none', origin='lower', cmap=cmap)
    im_median = ax[1, 1].imshow(image_median.T, interpolation='none', origin='lower', cmap=cmap)
    im_std = ax[1, 2].imshow(image_std.T, interpolation='none', origin='lower', cmap=cmap)
    im_sum = ax[0, 2].imshow(image_sum.T, interpolation='none', origin='lower', cmap=cmap)

    # Adding colorbars
    shrink = 0.55
    cbar_max = fig.colorbar(im_max, ax=ax[0, 0], shrink=shrink)
    cbar_min = fig.colorbar(im_min, ax=ax[0, 1], shrink=shrink)
    cbar_mean = fig.colorbar(im_mean, ax=ax[1, 0], shrink=shrink)
    cbar_median = fig.colorbar(im_median, ax=ax[1, 1], shrink=shrink)
    cbar_std = fig.colorbar(im_std, ax=ax[1, 2], shrink=shrink)
    cbar_sum = fig.colorbar(im_sum, ax=ax[0, 2], shrink=shrink)

    # Setting titles
    ax[0, 0].set_title('max')
    ax[0, 1].set_title('min')
    ax[1, 0].set_title('mean')
    ax[1, 1].set_title('median')
    ax[1, 2].set_title('std')
    ax[0, 2].set_title('sum')
    plt.tight_layout()

    plt.show()


def plot_aitoff(ra_deg, dec_deg, savepath=None, color='grey', title=None):
    sky_coords = SkyCoord(ra=ra_deg, dec=dec_deg, unit='deg', frame='fk5', equinox='J2000')
    gal_coords = sky_coords.galactic

    l = -gal_coords.l.wrap_at(180 * u.deg).radian
    b = gal_coords.b.radian

    plt.figure()

    plt.subplot(111, projection='aitoff')
    plt.scatter(l, b, marker='.', s=1, color=color)
    plt.tight_layout()
    if title:
        plt.title(title)
    if savepath:
        logger.info(f'Saving Aitoff plot to {savepath}')
        plt.savefig(savepath)
    # plt.show()


def plot_aitoff_density(ra_deg, dec_deg, savepath=None):
    sky_coords = SkyCoord(ra=ra_deg, dec=dec_deg, unit='deg', frame='fk5', equinox='J2000')
    gal_coords = sky_coords.galactic

    l = -gal_coords.l.wrap_at(180 * u.deg).radian
    b = gal_coords.b.radian

    # Calculate the density of points (this is kinda slow)
    xy = np.vstack([l, b])
    z = gaussian_kde(xy)(xy)

    fig = plt.figure(figsize=(5, 2.9))

    plt.subplot(111, projection='aitoff')
    # plt.grid(linewidth=0.5)
    # plt.scatter(l, b, marker='.', s=1, color='cyan', rasterized=True)
    plt.scatter(l, b, marker='.', s=1, c=z, cmap=cmasher.cosmic, rasterized=True)
    plt.tight_layout()

    xticks = ['210°', '240°', '270°', '300°', '330°', '0°', '30°', '60°', '90°', '120°', '150°']
    plt.xticks(fig.get_axes()[0].get_xticks(), xticks)
    # plt.xlabel('Galactic longitude (l)')
    # plt.ylabel('Galactic latitude (b)')
    if savepath:
        logger.info(f'Saving to {savepath}')
        plt.savefig(savepath, dpi=300)

    # plt.show()