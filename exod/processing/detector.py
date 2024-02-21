from exod.utils.logger import logger
from exod.utils.plotting import cmap_image

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.convolution import convolve
from astropy.visualization import ImageNormalize, SqrtStretch
from matplotlib import pyplot as plt, patches as patches
from scipy.interpolate import interp1d
from scipy.stats import kstest, poisson
from skimage.measure import label, regionprops, regionprops_table
from scipy.optimize import minimize_scalar


class Detector:
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

        self.df_regions = self.get_regions_sky_position()
        self.df_regions = self.filter_df_regions()
        self.df_lcs     = self.get_region_lightcurves()



    def calc_image_var(self):
        """
        Calculate the variability image from a data cube.
        """
        logger.info('Computing Variability')
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
        image_var_mask_labelled = label(image_var_mask)

        logger.info(f'threshold: {self.calc_extraction_threshold(sigma)} sigma: {sigma}')
        self.plot_threshold_level(self.calc_extraction_threshold(sigma))

        # Obtain the region properties for the detected regions.
        properties_ = ('label', 'bbox', 'centroid', 'weighted_centroid', 'intensity_mean',
                       'equivalent_diameter_area', 'area_bbox')
        region_dict = regionprops_table(label_image=image_var_mask_labelled,
                                        intensity_image=self.image_var, properties=properties_)
        df_regions = pd.DataFrame(region_dict)

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

    def filter_df_regions(self,):
        logger.info('Removing regions with area_bbox > 12')
        df_regions = self.df_regions[self.df_regions['area_bbox'] <= self.max_area_bbox]
        return df_regions

    def get_regions_sky_position(self):
        """
        Calculate the sky position of the detected regions.

        Test coords: obsid : 0803990501
        1313 X-1     : 03 18 20.00 -66 29 10.9
        1313 X-2     : 03 18 22.00 -66 36 04.3
        SN 1978K     : 03 17 38.62 -66 33 03.4
        NGC1313 XMM4 : 03 18 18.46 -66 30 00.2 (lil guy next to x-1)
        """
        # To calculate the EPIC X and Y coordinates of the variable sources, we use the final coordinates
        # in the variability map, which are not integers. To know to which X and Y correspond to this, we interpolate the
        # values of X and Y on the final coordinates. We divide by 80 because the WCS from the image is binned by x80
        # compared to X and Y values
        img_bin_size = 80
        logger.warning(f'Interpolating assuming a binning of {img_bin_size} in the image file')

        data_cube = self.data_cube
        interpX = interp1d(range(len(data_cube.bin_x)), data_cube.bin_x)
        interpY = interp1d(range(len(data_cube.bin_y)), data_cube.bin_y)

        all_res = []
        for i, row in self.df_regions.iterrows():
            X = interpX(row['weighted_centroid-0'])
            Y = interpY(row['weighted_centroid-1'])

            x_img = X / img_bin_size
            y_img = Y / img_bin_size
            skycoord = self.wcs.pixel_to_world(x_img, y_img)

            res = {'x_img'    : x_img, # x_position in the binned fits image
                   'y_img'    : y_img, # y_position in the binned fits image
                   'X'        : X,     # X in the event_list (sky coordinates)
                   'Y'        : Y,     # Y in the event-list (sky coordinates)
                   'ra'       : skycoord.ra.to_string(unit=u.hourangle, precision=2),
                   'dec'      : skycoord.dec.to_string(unit=u.deg, precision=2),
                   'ra_deg'   : skycoord.ra.value,
                   'dec_deg'  : skycoord.dec.value}
            all_res.append(res)

        self.df_sky = pd.DataFrame(all_res)
        logger.info(f'df_sky:\n{self.df_sky}')
        df_regions = pd.concat([self.df_regions, self.df_sky], axis=1)
        return df_regions

    def get_region_lightcurves(self):
        """
        Extract the lightcurves from the variable regions found.
        Returns
        -------
        df_lcs : DataFrame of Lightcurves
        """
        logger.info("Extracting lightcurves from data cube")
        lcs = [pd.DataFrame({'time': self.data_cube.bin_t[:-1]}),
               pd.DataFrame({'bti': self.data_cube.bti_bin_idx_bool[:-1]})]
        for i, row in self.df_regions.iterrows():
            xlo, xhi = row['bbox-0'], row['bbox-2']
            ylo, yhi = row['bbox-1'], row['bbox-3']
            data = self.data_cube.data[xlo:xhi, ylo:yhi]
            lc = np.nansum(data, axis=(0, 1), dtype=np.int32)
            res = pd.DataFrame({f'src_{i}': lc})
            lcs.append(res)

        df_lcs = pd.concat(lcs, axis=1)
        return df_lcs

    def plot_region_lightcurve(self, i, savepath=None):
        """Plot the ith region lightcurve."""
        lc = self.df_lcs[f'src_{i}']
        fig, ax = plt.subplots(figsize=(10, 4))
        ax2 = ax.twiny()
        ax.step(self.df_lcs['time'], lc, where='post', color='black', lw=1.0)
        ax2.step(range(len(lc)), lc, where='post', color='none', lw=1.0)

        ax.set_title(f'Source #{i}')
        ax.set_ylabel('Counts (N)')
        ax.set_xlabel('Time (s)')
        ax2.set_xlabel('Window/Frame Number')
        plt.tight_layout()

        if savepath:
            logger.info(f'Saving lightcurve plot to: {savepath}')
            plt.savefig(savepath)

    def plot_region_lightcurves(self, savedir=None, max_sources=15):
        if self.n_sources > max_sources:
            logger.info(f'{self.n_sources} > {max_sources} Not plotting lightcurves')
            return None

        logger.info(f'Plotting regions lightcurves')
        savepath = None
        for i, row in self.df_regions.iterrows():
            if savedir:
                savepath = savedir / f'lc_reg_{i}.png'
            self.plot_region_lightcurve(i, savepath=savepath)


    def plot_3d_image(self, image):
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

def plot_var_with_regions(var_img, df_regions, savepath):
    """
    Plot the variability image with the bounding boxes of the detected regions.

    Parameters
    ----------
    var_img    : np.ndarray : Variability Image  (or Likelihood Image)
    df_regions : pd.DataFrame : from extract_variability_regions
    savepath    : str : Path to save the figure to
    """
    logger.info('Plotting Variability map with source regions')

    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_title(f'Detected Regions : {len(df_regions)}')

    cmap = cmap_image()

    norm = ImageNormalize(stretch=SqrtStretch()) #LogNorm()

    m1 = ax.imshow(var_img.T, norm=norm, interpolation='none', origin='lower', cmap=cmap)
    cbar = plt.colorbar(mappable=m1, ax=ax, shrink=0.75)
    cbar.set_label('Variability')

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
    logger.info(f'Saving Variability image to: {savepath}')
    plt.tight_layout()
    plt.savefig(savepath)
    # plt.show()


def calc_KS_poission(lc):
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










if __name__ == "__main__":
    from exod.xmm.observation import Observation
    from exod.pre_processing.data_loader import DataLoader

    obs = Observation('0911791101')
    obs.get_files()
    evt = obs.events_processed[0]
    evt.read()
    img = obs.images[0]
    img.read(wcs_only=True)
    dl = DataLoader(evt)
    dl.run()
    detector = Detector(data_cube=dl.data_cube, wcs=img.wcs)
    detector.run()
    detector_info = detector.info

