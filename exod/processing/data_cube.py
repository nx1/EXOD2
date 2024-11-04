from exod.utils.logger import logger
from exod.utils.plotting import cmap_image, plot_frame_masks
from exod.pre_processing.bti import get_bti_bin_idx, get_bti_bin_idx_bool

import copy
import numpy as np
from scipy.stats import binned_statistic_dd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class DataCube:
    """
    Class to represent a 3D data cube.
    """
    def __init__(self, data):
        self.data = data
        self.shape = self.data.shape
        self.memory_mb = self.data.nbytes / (1024 ** 2)  # Convert bytes to megabytes

    def __repr__(self):
        return (f"DataCube(shape={self.shape}, "
                f"total_values={np.prod(self.shape)}, "
                f"memory={self.memory_mb:.2f} MB)")

    def copy(self):
        """Return a copy of the datacube."""
        return copy.deepcopy(self)

    def video(self, savepath=None):
        """Play each frame of the datacube sequentially in a matplotlib figure."""
        if np.isnan(self.data[:, :, 0]).all():
            logger.info('first frame all nan =no plot')
            return None

        fig, ax = plt.subplots()
        img = ax.imshow(self.data[:, :, 0].T, cmap=cmap_image(), animated=True, interpolation='none',
                        origin='lower')
        colorbar = fig.colorbar(img, ax=ax)

        def update(frame):
            ax.set_title(f'{self}\n{frame}/{num_frames}')
            img.set_array(self.data[:, :, frame].T)
            return img,

        num_frames = self.shape[2]
        ani = FuncAnimation(fig, update, frames=num_frames, interval=20)
        if savepath:
            logger.info(f'Saving {self} to {savepath}')
            ani.save(savepath)
        plt.show()


class DataCubeXMM(DataCube):
    """
    Class to represent a 3D data cube from XMM data.

    Attributes:
        event_list (EventList): EventList object.
        size_arcsec (float): size of one side of a cell in the cube in arcseconds.
        time_interval (float): time interval for each frame in seconds.
        extent (int): Initial Cube Size.
        pixel_size (float): size of one side of a cell in the cube in pixels.
        n_bins (int): Spatial Bins (x,y).
        bin_x (np.array): Bins for the x-axis.
        bin_y (np.array): Bins for the y-axis.
        bin_t (np.array): Bins for the t-axis.
        n_t_bins (int): Number of time bins.
        bti_bin_idx (np.array): Indexs of the bad time intervals (BTIs)
        bti_bin_idx_bool (np.array): Boolean Mask of the bad time intervals that can be applied to bin_t
        n_bti_bin (int): Number of bad time intervals bins.
        bti_frac (float): Fraction of bad time intervals.
        gti_bin_idx (np.array): Indexs of the good time intervals (GTIs)
        gti_bin_idx_bool (np.array): Boolean Mask of the good time intervals that can be applied to bin_t
        n_gti_bin (int): Number of good time intervals bins.
        gti_frac (float): Fraction of good time intervals.
        bccd_bin_idx (np.array): Indexs of frames marked as having irregular (bad) CCD exposures.
        bccd_frac (float): Fraction of time frames marked as having bad CCD exposures.
        data (np.array): 3-D numpy array containing the data.
        bbox_img (tuple): Bounding box of the image (single time slice).
    """
    def __init__(self, event_list, size_arcsec, time_interval):
        self.event_list = event_list
        self.size_arcsec = size_arcsec
        self.time_interval = time_interval
        self.extent = 51840  # Initial Cube Size
        self.pixel_size = size_arcsec / 0.05
        self.n_bins = int(self.extent / self.pixel_size) # Spatial Bins (x,y)
        self.bin_x = np.linspace(0, self.extent, self.n_bins + 1)
        self.bin_y = np.linspace(0, self.extent, self.n_bins + 1)
        self.bin_t = self.calc_time_bins()
        self.n_t_bins = len(self.bin_t) - 1

        self.bti_bin_idx = []      # index of bad time interval bins e.g. [2,4,6]
        self.bti_bin_idx_bool = [] # Mask of the bad time interval bins e.g. [False, False, True, False, True, ...]
        self.n_bti_bin = None
        self.bti_frac = None

        self.gti_bin_idx = []
        self.gti_bin_idx_bool = []
        self.n_gti_bin = None
        self.gti_frac = None

        # Used for keeping track of frames (time bins) with uneven ccd exposures.
        self.bccd_bin_idx = []
        self.bccd_bin_idx_bool = []
        self.n_bccd_bin = None
        self.bccd_frac = None

        self.data = self.bin_event_list()
        self.bbox_img = self.get_cube_bbox()
        self.crop_data_cube()
        self.relative_frame_exposures = np.ones(self.data.shape[2])
        super().__init__(self.data)

    def calc_time_bins(self):
        t_lo = self.event_list.time_min
        t_hi = self.event_list.time_max
        t_i = self.time_interval
        n_time_bins = int((t_hi - t_lo) / t_i)
        time_stop = t_lo + n_time_bins * t_i
        time_bins = np.arange(t_lo, time_stop + 1, t_i)
        return time_bins

    def bin_event_list(self):
        data = self.event_list.data
        sample = data['X'], data['Y'], data['TIME']
        bins = [self.bin_x, self.bin_y, self.bin_t]
        cube, bin_edges, bin_number = binned_statistic_dd(sample=sample, values=None, statistic='count', bins=bins)
        return cube

    def crop_data_cube(self):
        """Crop the surrounding areas of the data_cube that are empty."""
        bbox_img = self.bbox_img

        logger.info(f'Cropping data cube between bbox_img: {bbox_img}')
        self.data = self.data[bbox_img[0]:bbox_img[1], bbox_img[2]:bbox_img[3]]

        # Calculate the new bins
        self.bin_x = self.bin_x[bbox_img[0]:bbox_img[1]]
        self.bin_y = self.bin_y[bbox_img[2]:bbox_img[3]]

    def get_cube_bbox(self):
        """Get the Bounding Box corresponding to the cube's image plane."""
        idx_nonempty = np.where(np.sum(self.data, axis=2) > 0)
        bbox_img = (np.min(idx_nonempty[0]), np.max(idx_nonempty[0]) + 1,
                    np.min(idx_nonempty[1]), np.max(idx_nonempty[1]) + 1)
        return bbox_img

    def calc_gti_bti_bins(self, bti):
        """
        Calculate the good and bad time interval indexes & masks.

        Parameters:
            bti (dict): {['START':344.2, 'STOP':454.2], ...}
        """
        self.bti_bin_idx      = get_bti_bin_idx(bti=bti, bin_t=self.bin_t)
        self.bti_bin_idx_bool = get_bti_bin_idx_bool(bti_bin_idx=self.bti_bin_idx, bin_t=self.bin_t)
        self.gti_bin_idx_bool = ~self.bti_bin_idx_bool
        self.gti_bin_idx      = np.where(self.gti_bin_idx_bool)[0][:-1]
        self.n_gti_bin = len(self.gti_bin_idx)
        self.n_bti_bin = len(self.bti_bin_idx)
        self.bti_frac = self.n_bti_bin / self.n_t_bins
        self.gti_frac = self.n_gti_bin / self.n_t_bins
        logger.info(f'n_gti = {self.n_gti_bin:<4} / {self.n_t_bins} ({self.gti_frac:.2f})')
        logger.info(f'n_bti = {self.n_bti_bin:<4} / {self.n_t_bins} ({self.bti_frac:.2f})')

    def mask_bti(self):
        logger.info('Masking bad frames from Data Cube (setting to nan)')
        img_shape = (self.shape[0], self.shape[1], 1)
        img_nan = np.full(shape=img_shape, fill_value=np.nan, dtype=np.float64)
        self.data[:, :, self.bti_bin_idx] = img_nan

    def remove_bti_frames(self):
        """Return the cube without the masked nan frames."""
        data_non_nan = self.data[:, :, ~self.bti_bin_idx_bool[:-1]]
        return data_non_nan

    def remove_frames_with_partial_ccd_exposure(self, remove_frames=True, plot=False):
        """
        Remove the frames with irregular exposures between CCDs.
        
        https://xmm-tools.cosmos.esa.int/external/xmm_user_support/documentation/uhb/pnchipgeom.html
        https://xmm-tools.cosmos.esa.int/external/xmm_user_support/documentation/uhb/moschipgeom.html
        
        Test Cases: 0165560101, 0765080801, 0116700301, 0765080801, 0872390901, 0116700301
        0201900201, 0724840301, 0743700201

        Args:
            remove_frames (bool): If True remove the frames.
            plot (bool): If True plot diagnostics.
        """
        if not remove_frames:
            self.bccd_bin_idx_bool = np.full(self.n_t_bins, fill_value=False)
            return

        all_masks = {}
        for e in self.event_list.event_lists:
            bad_ccd_mask = self.get_bad_ccd_mask(event_list=e, plot=plot)
            all_masks[e.instrument] = bad_ccd_mask

        if all_masks:
            # Combine the frame masks from all the individual event lists.
            self.bccd_bin_idx_bool = np.any(list(all_masks.values()), axis=0)
            self.bccd_bin_idx      = np.where(self.bccd_bin_idx_bool)[0]

            # Remove from the data cube and update the relative frame exposures.
            self.data = np.where(self.bccd_bin_idx_bool, np.empty(self.data.shape) * np.nan, self.data)
            self.relative_frame_exposures = np.where(self.bccd_bin_idx_bool, 0, self.relative_frame_exposures)

            # Calculate the number and fraction of bins
            self.n_bccd_bin = len(self.bccd_bin_idx)
            self.bccd_frac = self.n_bccd_bin / self.n_t_bins
            logger.info(f'n_bccd = {self.n_bccd_bin:<4} / {self.n_t_bins} ({self.bccd_frac:.2f})')

    def get_bad_ccd_mask(self, event_list, plot=False):
        """
        Get the mask for the frames corresponding with irregular CCD exposures.

        Parameters:
            event_list (EventList): EventList object.
            plot (bool): if True, then plot.

        Returns:
            bccd_bin_idx_bool (np.array): Array of bools indicating the frames with irregular CCD exposures.
        """
        ccd_bins = event_list.get_ccd_bins()

        # Get the lightcurves for each CCD.
        lcs_ccd, bin_edges, bin_number = binned_statistic_dd(sample=(event_list.data['CCDNR'], event_list.data['TIME']),
                                                             values=None,
                                                             statistic='count',
                                                             bins=[ccd_bins, self.bin_t])

        if event_list.instrument == 'EPN':
            quadrant_split         = np.split(lcs_ccd, indices_or_sections=(3, 6, 9))
            lcs_ccd                = np.sum(quadrant_split, axis=1)
            lcs_median_quadrant    = np.median(quadrant_split, axis=1)
            lc_median_quadrant_max = np.max(lcs_median_quadrant, axis=0)
            lc_median_quadrant_min = np.min(lcs_median_quadrant, axis=0)
            ccd_bins = [1, 4, 7, 10, 13]

        lcs_ccd_max = np.max(lcs_ccd, axis=0)
        # lcs_ccd_min = np.min(lcs_ccd, axis=0)
        count_active_ccd = np.sum(lcs_ccd > 0, axis=0)  # Nbr of CCDs active in each frame

        m0 = count_active_ccd == 0  # Frame is entirely empty

        if event_list.instrument == 'EPN':
            lc_min_counts = 10
            lc_max_norm_diff = 0.5
            logger.info(f'Removing PN frames lc_min_counts > {lc_min_counts} and lc_max_norm_diff > {lc_max_norm_diff}')
            lc_median_norm_diff = (lc_median_quadrant_max - lc_median_quadrant_min) / (lc_median_quadrant_max + lc_median_quadrant_min)
            m1 = lcs_ccd_max > lc_min_counts             # Frame is more than 10 counts in the brightest CCD
            m2 = self.bti_bin_idx_bool[:-1]              # Frame is a bad time index
            m3 = count_active_ccd < len(ccd_bins) - 1    # Frame is not running all CCDs
            m4 = lc_median_norm_diff > lc_max_norm_diff  # When this is >0.5 it is equivalent to max/min > 3
            m5 = m1 & (m3 | (m2 & m4))
            bccd_bin_idx_bool = m0 | m5
            # Remove the frames that are 1 left of a True frame that are False
            bccd_bin_idx_bool = bccd_bin_idx_bool | np.roll(bccd_bin_idx_bool, shift=1)

            masks = [m0, m2, m1, m3, m4, m5, bccd_bin_idx_bool]
            labels = ['empty', 'is_bti', 'max > 10', 'active_ccds <  #_ccds', 'relative_diff > 0.5', 'combined', 'to remove']
            plot_frame_masks(instrum=event_list.instrument, masks=masks, labels=labels, plot=plot)
        else:
            bccd_bin_idx_bool = m0

        bccd_bin_idx = np.where(bccd_bin_idx_bool)[0]
        logger.info(f'Removing {np.sum(bccd_bin_idx_bool)} / {len(bccd_bin_idx_bool)} incomplete frames from {event_list.instrument}')

        # Plot the Lightcurves for PN
        if plot and event_list.instrument == 'EPN':
            fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(10, 10), sharex=True)
            for i, lc in enumerate(lcs_median_quadrant):
                ax[0].plot(lcs_ccd[i], label=f'lc ccd={i+1}')
                ax[1].plot(lcs_median_quadrant[i], label=f'median quadrant={i+1}')

            ax[-2].plot(lc_median_quadrant_min, label='ccd min')
            ax[-2].plot(lc_median_quadrant_max, label='ccd max')
            ax[-1].plot(lc_median_norm_diff, label='max-min/max+min')
            ax[-1].axhline(0.5, color='red', lw=1.0)
            for a in ax:
                a.legend(loc='right')
                # Plot the Masked frames.
                for j in range(len(bccd_bin_idx)):
                    a.axvspan(bccd_bin_idx[j], bccd_bin_idx[j]+1, color='red', alpha=0.2)
            plt.tight_layout()
            plt.show()

        return bccd_bin_idx_bool

    def multiply_time_interval(self, n_factor):
        """
        Used to increase the time_interval by a factor of n_factor, in order to quickly scan different timescales.
        #TODO the BTI need to be re-computed as well, at the DataLoader level most likely
        """

        self.time_interval = n_factor*self.time_interval
        self.bin_t = self.calc_time_bins()

        #Update the data cube
        # np.split(X, np.arange(N, len(X), N)) allows to cut X in chunks of size N (plus the remaining bit)
        datacube_twoframegroups = np.split(self.data, np.arange(n_factor, self.shape[2], n_factor), axis=2) #Splits in groups of n_factor along the time axis
        self.data = np.transpose([np.nansum(frame_grp, axis=2) for frame_grp in datacube_twoframegroups], (1,2,0)) #Nansum each group along the time axis, and makes it into a cube again

        #Update the relative exposures of each frame
        frame_exposures_twoframegroups = np.split(self.relative_frame_exposures, np.arange(n_factor, self.shape[2], n_factor)) #Splits in groups of n_factor along the time axis
        self.relative_frame_exposures = np.array([np.sum(exp_frame_grp) for exp_frame_grp in frame_exposures_twoframegroups])

        self.shape = self.data.shape


    @property
    def info(self):
        info = {'event_list'   : self.event_list.filename,
                'size_arcsec'  : self.size_arcsec,
                'time_interval': self.time_interval,
                'n_t_bins'     : self.n_t_bins,
                'n_bti_bin'    : self.n_bti_bin,
                'n_gti_bin'    : self.n_gti_bin,
                'gti_frac'     : self.gti_frac,
                'bti_frac'     : self.bti_frac,
                'n_bccd_bin'   : self.n_bccd_bin,
                'bccd_frac'    : self.bccd_frac,
                'bbox_img'     : self.bbox_img,
                'shape'        : self.shape,
                'total_values' : np.prod(self.shape),
                'memory_mb'    : self.memory_mb}
        for k, v in info.items():
            logger.info(f'{k:>13} : {v}')
        return info

def extract_lc(data_cube, xhi, xlo, yhi, ylo, dtype=np.int32):
    """
    Extract a lightcurve from a data cube by summing through a bounding box.
    """
    data = data_cube[xlo:xhi, ylo:yhi]
    lc = np.nansum(data, axis=(0, 1), dtype=dtype)
    return lc

if __name__ == "__main__":
    data_array = np.random.rand(10, 10, 10)
    data_array[:, :, 2] = np.zeros((10, 10))
    data_cube = DataCube(data_array)
    print(np.sum(data_cube.data, axis=(0, 1)))
    data_cube.video()
    print(data_cube)



