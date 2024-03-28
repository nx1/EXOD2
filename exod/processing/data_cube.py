from exod.utils.logger import logger
from exod.utils.plotting import cmap_image, plot_frame_masks
from exod.pre_processing.bti import get_bti_bin_idx, get_bti_bin_idx_bool

import numpy as np
from scipy.stats import binned_statistic_dd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm


class DataCube:
    def __init__(self, data):
        self.data = data
        self.shape = self.data.shape
        self.memory_mb = self.data.nbytes / (1024 ** 2)  # Convert bytes to megabytes

    def __repr__(self):
        return (f"DataCube(shape={self.shape}, "
                f"total_values={np.prod(self.shape)}, "
                f"memory={self.memory_mb:.2f} MB)")

    def video(self, savepath=None):
        if np.isnan(self.data[:, :, 0]).all():
            logger.info('first frame all nan =no plot')
            return None

        fig, ax = plt.subplots()
        img = ax.imshow(self.data[:, :, 0].T, cmap='hot', animated=True, interpolation='none',
                        origin='lower', norm=LogNorm())# norm=LogNorm())
        colorbar = fig.colorbar(img, ax=ax)

        def update(frame):
            ax.set_title(f'{self}\n{frame}/{num_frames}')
            img.set_array(self.data[:, :, frame].T)
            return img,

        num_frames = self.shape[2]
        ani = FuncAnimation(fig, update, frames=num_frames, interval=10)
        if savepath:
            logger.info(f'Saving {self} to {savepath}')
            ani.save(savepath)
        plt.show()

    def plot_cube_statistics(self):
        cube = self.data
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


class DataCubeXMM(DataCube):
    def __init__(self, event_list, size_arcsec, time_interval):
        self.event_list = event_list
        self.size_arcsec = size_arcsec
        self.time_interval = time_interval
        self.extent = 51840  # Initial Cube Size
        self.pixel_size = size_arcsec / 0.05
        self.n_bins = int(self.extent / self.pixel_size)
        self.bin_x = np.linspace(0, self.extent, self.n_bins + 1)
        self.bin_y = np.linspace(0, self.extent, self.n_bins + 1)
        self.bin_t = self.calc_time_bins()

        self.bti_bin_idx = []
        self.bti_bin_idx_bool = []
        self.gti_bin_idx = []
        self.gti_bin_idx_bool = []

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
        """Crop the surrounding areas of the datacube that are empty."""
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

        Parameters
        ----------
        bti : {['START':344.2, 'STOP':454.2], ...}
        """
        self.bti_bin_idx      = get_bti_bin_idx(bti=bti, bin_t=self.bin_t)
        self.bti_bin_idx_bool = get_bti_bin_idx_bool(rejected_idx=self.bti_bin_idx, bin_t=self.bin_t)
        self.gti_bin_idx_bool = ~self.bti_bin_idx_bool
        self.gti_bin_idx      = np.where(self.gti_bin_idx_bool)[0][:-1]

    def mask_bti(self):
        logger.info('Masking bad frames from Data Cube (setting to nan)')
        img_shape = (self.shape[0], self.shape[1], 1)
        img_nan = np.full(shape=img_shape, fill_value=np.nan, dtype=np.float64)
        self.data[:, :, self.bti_bin_idx] = img_nan

    def remove_bti_frames(self):
        """Return the cube without the masked nan frames."""
        data_non_nan = self.data[:, :, ~self.bti_bin_idx_bool[:-1]]
        return data_non_nan

    def remove_frames_partial_CCDexposure(self, plot=False):
        """
        Remove the frames with irregular exposures between CCDs.
        
        Test Cases: 0165560101, 0765080801, 0116700301, 0765080801, 0872390901, 0116700301
        0201900201, 0724840301, 0743700201

        """
        for evt_list in self.event_list.event_lists:
            logger.info(f'{evt_list.filename}, {evt_list.instrument} {evt_list.submode}')
            if evt_list.instrument == 'EPN':
                ccd_bins = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            elif evt_list.instrument == "EMOS1":
                ccd_bins = list(set(evt_list.data['CCDNR']))
                print('MOS 1 CCDS:', ccd_bins)
            elif evt_list.instrument == "EMOS2":
                ccd_bins = [1, 2, 3, 4, 5, 6, 7]
            ccd_bins.append(13) # Just to get a right edge for the final bin

            # Get the lightcurves for each CCD.
            sample = evt_list.data['CCDNR'], evt_list.data['TIME']
            lcs_ccd, bin_edges, bin_number = binned_statistic_dd(sample, values=None, statistic='count', bins=[ccd_bins, self.bin_t])
            if evt_list.instrument == 'EPN': #We rebin pn into quadrant-wise lightcurve, and use the median CCD of each quadrant
                quadrant_split = np.split(lcs_ccd, (3,6,9))
                lcs_ccd = np.sum(quadrant_split, axis=1)
                lc_median_quadrant = np.median(quadrant_split, axis=1)
                lc_median_quadrant_max = np.max(lc_median_quadrant, axis=0)
                lc_median_quadrant_min = np.min(lc_median_quadrant, axis=0)
                ccd_bins = [1, 4, 7, 10, 13]
            lcs_ccd_max = np.max(lcs_ccd, axis=0)
            lcs_ccd_min = np.min(lcs_ccd, axis=0)
            # lcs_ccd_min[lcs_ccd_min == 0] = 1
            count_active_ccd = np.sum(lcs_ccd > 0, axis=0) # Nbr of CCDs active in each frame

            if plot:
                # Plot all Lightcurves
                fig, ax = plt.subplots(nrows=len(lcs_ccd)+3, ncols=1, figsize=(10,10), sharex=True)
                plt.suptitle(evt_list.instrument)
                for i, lc in enumerate(lcs_ccd):
                    ax[i].plot(lc, label=f'ccd={i}')
                if evt_list.instrument == 'EPN':
                    ax[-3].plot(lc_median_quadrant_min, label='ccd min')
                    ax[-2].plot(lc_median_quadrant_max, label='ccd max')
                    ax[-1].plot(np.where(lc_median_quadrant_min > 0, lc_median_quadrant_max / lc_median_quadrant_min, np.nan), label='max/min')
                else:
                    ax[-3].plot(lcs_ccd_min, label='ccd min')
                    ax[-2].plot(lcs_ccd_max, label='ccd max')
                    ax[-1].plot(np.where(lcs_ccd_min>0, lcs_ccd_max/lcs_ccd_min, np.nan), label='max/min')
                ax[-1].axhline(3, color='red', lw=1.0)

                for a in ax:
                    a.legend(loc='right')
                plt.tight_layout()
                plt.show()
            #############################

            m0 = count_active_ccd == 0 # Frame is entirely empty

            if evt_list.instrument == 'EPN':
                # We remove bright frames in EPICpn that have either one inactive CCD or a ratio between brightest and faintest over 3
                # This corresponds to fully or partially inactive quadrants for pn
                m1 = lcs_ccd_max > 10                       # Frame is more than 10 counts in the brightest CCD
                m2 = self.bti_bin_idx_bool[:-1]           # Frame is a bad time index
                # m2 = np.full(self.shape[-1:], True)
                m3 = count_active_ccd < len(ccd_bins) - 1   # Frame is not running all CCDs
                m4 = (lc_median_quadrant_max / lc_median_quadrant_min) > 3        # Frame is max/min > 3
                # We remove frames when one quadrant is off or (BTI & ratio of medians >3)
                m5 =  m1 & (m3 | (m2 & m4)) # m1 & m2 & (m3 | m4)
                frames_to_remove = m0 | m5

                if plot:
                    # Plot masks
                    masks  = [m0, m1, m3, m4, m5, frames_to_remove]
                    labels = ['empty', 'max > 10', 'active_ccds <  #_ccds', 'max/min > 3', 'combined', 'to remove']
                    plot_frame_masks(instrum=evt_list.instrument, masks=masks, labels=labels, plot=True)

            # elif evt_list.submode.startswith('PrimePartial'):
            #     logger.info(f'PrimePartial submode!')
            #     m6 = lcs_ccd[0] == 0                    # Main central CCD is off
            #     m7 = np.mean(lcs_ccd[0]) > 5            # Main Central CCD has > 5 mean counts (NOT AN ARRAY)
            #     m8 = np.concatenate(([False], m6[:-1])) # We also remove the next frame
            #     frames_to_remove = m0 | (m6 & m7) | m8
            #
            #     logger.info(f'central ccd mean>5? {m7}')
            #     # Plot masks
            #     labels = ['empty', 'central off', 'central + 1', 'to remove']
            #     plot_frame_masks(instrum=evt_list.instrument, masks=[m0, m6, m8, frames_to_remove], labels=labels, plot=True)
            else:
                frames_to_remove = m0
            logger.info(f'Removing {np.sum(frames_to_remove)} / {len(frames_to_remove)} incomplete frames from {evt_list.instrument}')
            self.data = np.where(frames_to_remove, np.empty(self.data.shape) * np.nan, self.data)
            self.relative_frame_exposures = np.where(frames_to_remove, 0, self.relative_frame_exposures)

    def multiply_time_interval(self, n_factor):
        """Used to increase the time_interval by a factor of n_factor, in order to quickly scan different timescales.
        Important: the BTI need to be re-computed as well, at the DataLoader level most likely"""

        self.time_interval=n_factor*self.time_interval
        self.bin_t = self.calc_time_bins()

        #Update the data cube
        # np.split(X, np.arange(N, len(X), N)) allows to cut X in chunks of size N (plus the remaining bit)
        datacube_twoframegroups = np.split(self.data, np.arange(n_factor, self.shape[2], n_factor), axis=2) #Splits in groups of n_factor along the time axis
        self.data = np.transpose([np.nansum(frame_grp, axis=2) for frame_grp in datacube_twoframegroups], (1,2,0)) #Nansum each group along the time axis, and makes it into a cube again

        #Update the relative exposures of each frame
        frame_exposures_twoframegroups = np.split(self.relative_frame_exposures, np.arange(n_factor, self.shape[2], n_factor)) #Splits in groups of n_factor along the time axis
        self.relative_frame_exposures = np.array([np.sum(exp_frame_grp) for exp_frame_grp in frame_exposures_twoframegroups])




    @property
    def info(self):
        info = {'event_list': self.event_list.filename,
                'size_arcsec': self.size_arcsec,
                'time_interval': self.time_interval,
                'extent': self.extent,
                'pixel_size': self.pixel_size,
                'n_bins': self.n_bins,
                'bbox_img': self.bbox_img,
                'shape': self.shape,
                'memory_mb': self.memory_mb}
        for k, v in info.items():
            logger.info(f'{k:>13} : {v}')
        return info


if __name__ == "__main__":
    data_array = np.random.rand(10, 10, 10)
    data_array[:, :, 2] = np.zeros((10, 10))
    data_cube = DataCube(data_array)
    print(np.sum(data_cube.data, axis=(0, 1)))
    data_cube.video()
    print(data_cube)
