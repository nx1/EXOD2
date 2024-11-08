from exod.processing.bti import plot_bti, get_gti_threshold, get_bti
from exod.processing.coordinates import get_regions_sky_position
from exod.processing.data_cube import DataCubeXMM
from exod.utils.logger import logger
from exod.utils.plotting import cmap_image
from exod.processing.coordinates import calc_df_regions
from exod.xmm.observation import Observation

import numpy as np
import pandas as pd
from astropy.convolution import convolve
from astropy.visualization import ImageNormalize, SqrtStretch
from matplotlib import pyplot as plt, patches as patches
from scipy.stats import kstest, poisson
from skimage.measure import label, regionprops
from scipy.optimize import minimize_scalar

from exod.utils.util import save_df, save_info
from exod.xmm.event_list import EventList


class Detector:
    """
    Obsolete detector class that re-implements the method used in Inés Pastor-Marazuela (2020)
    See: https://arxiv.org/abs/2005.08673
    """
    def __init__(self, data_cube, wcs, sigma=4):
        self.data_cube = data_cube
        self.wcs = wcs
        self.sigma = sigma

        self.max_area_bbox = 6**2
        self.n_sources_max = 15  # Number of sources above which to optimise sigma
        self.n_target_regions = 10 # Number of Sources to aim for when optimising.

        self.df_regions = None
        self.df_sky = None

    def __repr__(self):
        return f'Detector({self.data_cube})'

    def run(self):
        self.image_var  = self.calc_image_var()
        self.image_var_mean = np.mean(self.image_var)
        self.image_var_std  = np.std(self.image_var)
        # self.image_var = self.conv_var_img(box_size=3)
        self.df_regions = self.extract_var_regions(sigma=self.sigma)

        if self.n_sources > self.n_sources_max:
            logger.info('More than 15 sources found! Optimising sigma.')
            self.sigma = self.optimise_sigma(self.image_var, n_target_regions=self.n_target_regions)
            self.threshold = self.calc_extraction_threshold(self.sigma)
            self.df_regions = self.extract_var_regions(sigma=self.sigma)

        self.df_sky = get_regions_sky_position(self.data_cube, self.df_regions, self.wcs)
        self.df_regions = pd.concat([self.df_regions, self.df_sky], axis=1)

        self.df_regions = self.filter_df_regions()
        self.df_lcs     = self.get_region_lightcurves()

    def calc_image_var(self):
        """
        Calculate the variability image from a data cube.
        """
        logger.info('Computing variability...')
        image_max = np.nanmax(self.data_cube.data, axis=2)
        image_std = np.nanstd(self.data_cube.data, axis=2)
        image_mean = np.nanmean(self.data_cube.data, axis=2)
        image_var = (image_max - image_mean) * image_std
        return image_var

    def conv_var_img(self, box_size=3):
        """
        Convolve the variability image with a box kernel.
        """
        logger.info('Convolving Variability')
        kernel = np.ones((box_size, box_size)) / box_size ** 2
        image_var_conv = convolve(self.image_var, kernel)

        # New version
        # convolved = gaussian_filter(image_var, 1)
        # convolved = np.where(image_var>0, convolved, 0)
        return image_var_conv

    def calc_extraction_threshold(self, sigma):
        """Calculate the extraction threshold for a given sigma value."""
        self.image_var_threshold = self.image_var_mean + sigma * self.image_var_std
        return self.image_var_threshold

    def extract_var_regions(self, sigma=5):
        """
        Use skimage to obtain the contiguous regions in the variability image.

        We currently used the weighted centroid which calculates a centroid based on
        both the size of the bounding box and the values of the pixels themselves.

        https://scikit-image.org/docs/stable/auto_examples/segmentation/index.html

        Returns
        -------
        df_regions : DataFrame of Detected Regions
        """
        logger.info('Extracting variable Regions')
        image_var_mask = self.image_var > self.calc_extraction_threshold(sigma)

        logger.info(f'threshold: {self.calc_extraction_threshold(sigma)} sigma: {sigma}')
        # self.plot_threshold_level(self.calc_extraction_threshold(sigma))

        # Obtain the region properties for the detected regions.
        df_regions = calc_df_regions(image=self.image_var, image_mask=image_var_mask)


        # Sort by Most Variable and reset in the label column
        df_regions = df_regions.sort_values(by='intensity_mean', ascending=False).reset_index(drop=True)
        df_regions['label'] = df_regions.index
        return df_regions

    def objective_function(self, sigma, image, n_target_regions):
        """Function used for optimising sigma detection threshold."""
        threshold = self.calc_extraction_threshold(sigma)
        image_mask = image > threshold
        labeled_image = label(image_mask)
        regions = regionprops(labeled_image)
        n_regions = len(regions)
        # logger.info(f'sigma={sigma:.4f} threshold={threshold:.4f} n_regions={n_regions} target={n_target_regions}')
        return np.abs(n_regions - n_target_regions)

    def optimise_sigma(self, image, n_target_regions, sigma_bounds=(0.1, 20.0)):
        """Find the optimal value of sigma to get n_target regions."""
        result = minimize_scalar(self.objective_function, args=(image, n_target_regions), bounds=sigma_bounds, method='bounded')
        logger.info(f'Optimisation results:\n{result}')
        return result.x

    def plot_threshold_level(self, threshold):
        plt.figure(figsize=(10, 3))
        plt.title('Thresholding')
        plt.plot(self.image_var.flatten(), label='Variability')
        plt.axhline(threshold, color='red', label=fr'threshold={threshold:.2f} ({self.sigma:.2f}$\sigma$)')
        plt.legend()
        plt.ylabel('V score')
        # plt.show()

    def filter_df_regions(self):
        logger.info('Removing regions with area_bbox > 12')
        df_regions = self.df_regions[self.df_regions['area_bbox'] <= self.max_area_bbox]
        return df_regions

    def get_region_lightcurves(self):
        """
        Extract the lightcurves from the variable regions found.
        Returns
        -------
        df_lc : DataFrame of Lightcurves
        """
        df_lcs = get_region_lcs(data_cube=self.data_cube, df_regions=self.df_regions)
        return df_lcs

    def plot_region_lightcurves(self, savedir=None, max_sources=15):
        if self.n_sources > max_sources:
            logger.info(f'{self.n_sources} > {max_sources} Not plotting lightcurves')
            return None

        logger.info(f'Plotting regions lightcurves')
        savepath = None
        for i, row in self.df_regions.iterrows():
            if savedir:
                savepath = savedir / f'lc_reg_{i}.png'
            plot_region_lightcurve(self.df_lcs, i, savepath=savepath)

    @property
    def n_sources(self):
        return len(self.df_regions)

    @property
    def info(self):
        info = {'data_cube'           : repr(self.data_cube),
                'sigma'               : self.sigma,
                'max_area_bbox'       : self.max_area_bbox,
                'n_sources_max'       : self.n_sources_max,
                'n_target_regions'    : self.n_target_regions,
                'image_var_mean'      : self.image_var_mean,
                'image_var_std'       : self.image_var_std,
                'image_var_threshold' : self.image_var_threshold,
                'n_sources'           : self.n_sources}

        for k, v in info.items():
            logger.info(f'{k:>20} : {v}')
        return info


def plot_region_lightcurve(df_lcs, i, savepath=None):
    """Plot the ith region lightcurve."""
    lc = df_lcs[f'src_{i}']
    fig, ax = plt.subplots(figsize=(10, 4))
    ax2 = ax.twiny()
    ax.step(df_lcs['time'], lc, where='post', color='black', lw=1.0)
    ax2.step(range(len(lc)), lc, where='post', color='none', lw=1.0)

    ax.set_title(f'Source #{i}')
    ax.set_ylabel('Counts (N)')
    ax.set_xlabel('Time (s)')
    ax.set_xlim(df_lcs['time'].min(), df_lcs['time'].max())
    ax2.set_xlabel('Window/Frame Number')
    plt.tight_layout()

    if savepath:
        logger.info(f'Saving lightcurve plot to: {savepath}')
        plt.savefig(savepath)


def get_region_lcs(data_cube, df_regions):
    logger.info("Extracting lightcurves from data cube")
    lcs = [pd.DataFrame({'time' : data_cube.bin_t[:-1]}),
           pd.DataFrame({'bti'  : data_cube.bti_bin_idx_bool[:-1]})]
    for i, row in df_regions.iterrows():
        xlo, xhi = row['bbox-0'], row['bbox-2']
        ylo, yhi = row['bbox-1'], row['bbox-3']
        data = data_cube.data[xlo:xhi, ylo:yhi]
        lc   = np.nansum(data, axis=(0, 1), dtype=np.int32)
        res  = pd.DataFrame({f'src_{i}': lc})
        lcs.append(res)
    df_lcs = pd.concat(lcs, axis=1)
    return df_lcs


def plot_image_with_regions(image, df_regions, cbar_label=None, savepath=None):
    """
    Plot the variability image with the bounding boxes of the detected regions.

    Parameters
    ----------
    image       : np.ndarray : Variability Image  (or Likelihood Image)
    df_regions  : pd.DataFrame : from extract_variability_regions
    cbar_label  : string : Label for colorbar
    savepath    : str : Path to save the figure to
    """
    logger.info('Plotting Image with source regions')

    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_title(f'Detected Regions : {len(df_regions)}')

    cmap = cmap_image()

    norm = ImageNormalize(stretch=SqrtStretch()) #LogNorm()

    m1 = ax.imshow(image.T, norm=norm, interpolation='none', origin='lower', cmap=cmap)
    cbar = plt.colorbar(mappable=m1, ax=ax, shrink=0.75)
    cbar.set_label(cbar_label)

    src_color = 'lime'
    ax.scatter(df_regions['weighted_centroid-0'], df_regions['weighted_centroid-1'], marker='+', s=10, color='white')
    ax.scatter(df_regions['centroid-0'], df_regions['centroid-1'], marker='.', s=10, color=src_color)

    for i, row in df_regions.iterrows():
        ind   = row['label']
        x_cen = row['centroid-0']
        y_cen = row['centroid-1']

        width  = row['bbox-2'] - row['bbox-0']
        height = row['bbox-3'] - row['bbox-1']

        x_pos = x_cen - width / 2
        y_pos = y_cen - height / 2

        rect = patches.Rectangle(xy=(x_pos, y_pos),
                                 width=width,
                                 height=height,
                                 linewidth=1,
                                 edgecolor=src_color,
                                 facecolor='none')

        plt.text(x_pos+width, y_pos+height, str(ind), c=src_color)
        ax.add_patch(rect)
    plt.tight_layout()
    if savepath:
        logger.info(f'Saving Image to: {savepath}')
        plt.savefig(savepath)
    plt.show()


def calc_ks_poission(lc):
    """
    Calculate the KS Probability assuming the lightcurve was
    created from a possion distribution with the mean of the lightcurve.
    """
    #logger.info("Calculating KS Probability, assuming a Poission Distribution")
    lc_mean = mean_of_poisson = np.nanmean(lc)
    N_data = len(lc)
    result = kstest(lc, [lc_mean] * N_data)
    expected_distribution = poisson(mean_of_poisson)
    ks_res = kstest(lc, expected_distribution.cdf)
    logger.debug(f'KS_prob: lc_mean = {lc_mean}, N_data = {N_data}\nks_res = {ks_res}')
    return ks_res


"""
args = {'obsid': obsid,
        'size_arcsec': 15.0,
        'time_interval': 5,
        'gti_threshold': 1.5,
        'min_energy': 0.5,
        'max_energy': 12.0,
        'sigma': 4,
        'gti_only': True,
        'remove_partial_ccd_frames': False,
        'clobber': False}
"""
def run_pipeline(obsid, time_interval=1000, size_arcsec=10, gti_only=False, min_energy=0.5,
                 max_energy=12.0, remove_partial_ccd_frames=False, sigma=5, clobber=False):

    # Create the Observation class
    observation = Observation(obsid)
    observation.filter_events(clobber=clobber)
    observation.create_images(clobber=clobber)
    observation.get_files()

    observation.get_events_overlapping_subsets()
    event_list = EventList.from_event_lists(observation.events_overlapping_subsets[0])

    gti_threshold = get_gti_threshold(event_list.N_event_lists)
    t_bin_he, lc_he = event_list.get_high_energy_lc(time_interval)
    bti = get_bti(time=t_bin_he, data=lc_he, threshold=gti_threshold)
    df_bti = pd.DataFrame(bti)

    t_bin_he, lc_he = event_list.get_high_energy_lc(time_interval=time_interval)

    event_list.filter_by_energy(min_energy=min_energy, max_energy=max_energy)

    img = observation.images[0]
    img.read(wcs_only=True)

    # Create Data Cube
    data_cube = DataCubeXMM(event_list=event_list, size_arcsec=size_arcsec, time_interval=time_interval)
    data_cube.mask_frames_with_partial_ccd_exposure(mask_frames=remove_partial_ccd_frames)
    data_cube.video(savedir=None)

    if gti_only:
        data_cube.mask_bti()

    # Detection
    detector = Detector(data_cube=data_cube, wcs=img.wcs, sigma=sigma)
    detector.run()

    # detector.plot_3d_image(detector.image_var)
    detector.plot_region_lightcurves(savedir=None) # savedir=observation.path_results
    plot_bti(time=t_bin_he[:-1], data=lc_he, threshold=gti_threshold, bti=bti, savepath=observation.path_results / 'bti_plot.png')
    plot_image_with_regions(image=detector.image_var, df_regions=detector.df_regions, cbar_label='Variability Score',
                            savepath=observation.path_results / 'image_var.png')

    # Save Results
    save_df(df=df_bti, savepath=observation.path_results / 'bti.csv')
    save_df(df=detector.df_lcs, savepath=observation.path_results / 'lcs.csv')
    save_df(df=detector.df_regions, savepath=observation.path_results / 'regions.csv')

    save_info(dictionary=observation.info, savepath=observation.path_results / 'obs_info.csv')
    save_info(dictionary=event_list.info, savepath=observation.path_results / 'evt_info.csv')
    save_info(dictionary=data_cube.info, savepath=observation.path_results / 'data_cube_info.csv')
    save_info(dictionary=detector.info, savepath=observation.path_results / 'detector_info.csv')

    plt.show()

if __name__ == "__main__":
    obs = Observation('0911791101')
    obs.get_files()
    evt = obs.events_processed[0]
    evt.read()
    img = obs.images[0]
    img.read(wcs_only=True)
    data_cube = DataCubeXMM(event_list=evt, size_arcsec=20, time_interval=50)
    detector = Detector(data_cube=data_cube, wcs=img.wcs)
    detector.run()
    detector_info = detector.info
    plt.show()
