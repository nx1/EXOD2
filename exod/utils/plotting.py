import matplotlib
import numpy
import numpy as np
from matplotlib import pyplot as plt, pyplot
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm, ListedColormap


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


def compare_images(images, titles, log=False, plot=True):
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