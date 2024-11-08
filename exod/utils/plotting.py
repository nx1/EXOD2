import base64
import io

import numpy as np
import matplotlib
from matplotlib import pyplot as plt, pyplot
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm, ListedColormap
from matplotlib.widgets import RectangleSelector
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.stats import gaussian_kde
import cmasher

from exod.utils.logger import logger


def use_scienceplots():
    """Use the scienceplots module for matplotlib plots."""
    try:
        import scienceplots
        plt.style.use('science')
    except ModuleNotFoundError:
        pass

def set_latex_font():
    """Set matplotlib global font to STIX (same as latex)."""
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'


def cmap_image():
    """Colormap used in production of images."""
    cmap = matplotlib.colormaps['hot']
    cmap.set_bad('black')
    return cmap


def plot_image(image_arr, title, log=False):
    """Plot a 2D numpy array as an image."""
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
    """
    Creates a slideshow of multiple 2D numpy arrays of the same size.
    Args:
        images (list of np.ndarray): list of numpy arrays
        titles (list): titles of images
        log (bool): use Log normalisation for image plots.
        plot (bool): switch to disable or enable plotting.
    """
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
    """
    Helper function to view which frames were masked during processing making of the data cube.

    Args:
        instrum (str): Name of the instruement.
        masks (list of np.arrays): list of masks (true/false arrays) corresponding
        labels(list of str): labels for each mask.
        plot (bool): switch for turn on/off plotting
    """
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
    """Plot various first order functions for a 3-dimensional data cube."""
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
    """
    Plot ra and dec coordinates on an aitoff plot.

    Args:
        ra_deg (list): ras in degrees.
        dec_deg (list): decs in degrees.
        savepath (str or Path): savepath for plot.
        color (str): color for the marker points.
        title (str): Title for the plot.
    """
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
    """
    Plot ra and dec coordinates on an aitoff plot and color them by spatial density.

    Args:
        ra_deg (list): ras in degrees.
        dec_deg (list): decs in degrees.
        savepath (str or Path): savepath for plot.
    """
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


def interactive_scatter_with_rectangles(x_data, y_data):
    """
    Plot x and y data, allowing the user to interactively select rectangles.
    Prints the (x, y, width, height) of the selected rectangle in the terminal.

    Args:
        x_data (array): the x coordinates of the scatter data.
        y_data (array): the y coordinates of the scatter data.
    """
    fig, ax = plt.subplots()
    ax.scatter(x_data, y_data, color='black', alpha=0.5, s=1)

    def onselect(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        width  = abs(x2 - x1)
        height = abs(y2 - y1)

        print(f"({x1:.2f}, {y1:.2f}, {width:.2f}, {height:.2f}) (x,y,w,h)")
        rect = plt.Rectangle((min(x1, x2), min(y1, y2)), width, height, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        last_rect = rect
        plt.draw()

    rect_selector = RectangleSelector(ax, onselect, useblit=True, button=[1], minspanx=0.1, minspany=0.1, spancoords='data', interactive=True)
    plt.show()


def get_image_limits(image):
    """Get the x and y limits of the non-zero pixels in an image."""
    # Find the non-zero rows and columns
    non_zero_rows = np.any(image != 0, axis=1)
    non_zero_cols = np.any(image != 0, axis=0)

    # Get the min and max of the non-zero rows and columns
    ymin, ymax = np.where(non_zero_rows)[0][[0, -1]]
    xmin, xmax = np.where(non_zero_cols)[0][[0, -1]]

    return (xmin, xmax), (ymin, ymax)

def plot_event_list_ccds(table):
    """
    Plot the images for each CCD for a given eventlist.
    
    Parameters:
        table (astropy.table.Table): Event List Table.

    Returns:
        fig (matplotlib.figure.Figure): Figure with each subplots containing the image for the specific CCD.
    """
    ccdnrs = np.unique(table['CCDNR'])
    fig, ax = plt.subplots((len(ccdnrs)+1)//2, 2 , figsize=(20,20))
    ax = ax.flatten()
    for i, ccdnr in enumerate(ccdnrs):
        sub = table[table['CCDNR'] == ccdnr]
        xbins = np.arange(sub['RAWX'].min(), sub['RAWX'].max(), 1)  
        ybins = np.arange(sub['RAWY'].min(), sub['RAWY'].max(), 1) 
        im, _, _ = np.histogram2d(x=sub['RAWX'], y=sub['RAWY'], bins=[xbins,ybins])
        ax[i].imshow(im, cmap='hot', origin='lower', interpolation='none', aspect='equal', norm=LogNorm())
        ax[i].text(5, 5, s=ccdnr, color='white', bbox=dict(facecolor='black', edgecolor='white'))
    plt.show()
    return fig


def fig2data_url(fig):
    """Convert a matplotlib Figure() to a base64 png data url (for use in a html <img> tag)"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    data_url = base64.b64encode(buf.read()).decode('ascii')
    return data_url
