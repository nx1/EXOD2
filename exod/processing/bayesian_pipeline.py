from exod.utils.logger import logger
from exod.utils.path import data, read_observation_ids, savepaths_combined
from exod.utils.util import save_result, load_info, load_df, get_unique_xy
from exod.utils.plotting import cmap_image
from exod.pre_processing.data_loader import DataLoader
from exod.processing.bayesian_computations import load_precomputed_bayes_limits, PrecomputeBayesLimits, B_peak_log, B_eclipse_log
from exod.processing.data_cube import extract_lc
from exod.processing.background_inference import calc_cube_mu
from exod.processing.coordinates import get_regions_sky_position, calc_df_regions
from exod.xmm.event_list import EventList
from exod.xmm.observation import Observation

import itertools
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, BoundaryNorm
import pandas as pd
from astropy.visualization import ImageNormalize, SqrtStretch


class Pipeline:
    """
    Pipeline to process a single XMM-Newton observation and find variable sources.

    Attributes:
        runid (str): Unique identifier for the run.
        obsid (str): Observation ID.
        observation (Observation): Observation object.
        subset_number (int): The subset number.
        total_subsets (int): Total number of subsets.
        size_arcsec (float): Size of the region in arcseconds.
        time_interval (int): Time interval in seconds.
        gti_only (bool): Use only Good Time Intervals.
        remove_partial_ccd_frames (bool): Remove partial CCD frames.
        min_energy (float): Minimum energy in keV.
        max_energy (float): Maximum energy in keV.
        clobber (bool): Overwrite existing files.
        precomputed_bayes_limit (PrecomputeBayesLimits): Precomputed Bayes Limits.
        savedir (Path): Directory to save results to.
        n_regions (int): Number of regions detected.
    """
    def __init__(self, obsid, size_arcsec=20.0, time_interval=50, gti_only=False,
                 remove_partial_ccd_frames=True, min_energy=0.2,
                 max_energy=10.0, clobber=False, precomputed_bayes_limit=PrecomputeBayesLimits(threshold_sigma=3)):
        self.runid = None
        self.obsid = obsid
        self.observation = Observation(obsid)
        self.subset_number = None
        self.total_subsets = None

        self.size_arcsec = size_arcsec
        self.time_interval = time_interval
        self.gti_only = gti_only
        self.remove_partial_ccd_frames = remove_partial_ccd_frames
        self.min_energy = min_energy
        self.max_energy = max_energy
        self.clobber = clobber
        self.precomputed_bayes_limit = precomputed_bayes_limit
        self.savedir = None

        self.n_regions = None

    def generate_runid(self):
        """Generate a unique runid."""
        runid = f'{self.obsid}_{self.subset_number}_{self.time_interval}_{self.min_energy}_{self.max_energy}'
        self.runid = runid
        return runid

    def get_savedir(self, subset_number):
        """
        Create the directory to save results to:
        /results/0765080801/subset_0/tbin=50_Elo=0.2_Ehi=12.0/
        """
        savedir = (self.observation.path_results
                   / f'subset_{subset_number}'
                   / f'tbin={self.time_interval}_Elo={self.min_energy}_Ehi={self.max_energy}')
        logger.info(f'savedir set to: {savedir}')
        self.savedir = savedir
        self.subset_number = subset_number
        return self.savedir

    def pre_process(self):
        """Pre-process the data."""
        self.precomputed_bayes_limit.load()
        self.observation.filter_events(clobber=self.clobber)
        self.observation.create_images(clobber=self.clobber)
        self.observation.get_files()
        self.observation.get_events_overlapping_subsets()
        self.total_subsets = len(self.observation.events_overlapping_subsets)

    def run(self):
        self.pre_process()
        for i_subset, subset_overlapping_exposures in enumerate(self.observation.events_overlapping_subsets):
            self.run_subset(i_subset, subset_overlapping_exposures)

    def run_subset(self, i_subset, subset_overlapping_exposures):
        """
        Run the pipeline on a subset of overlapping exposures.

        Parameters:
            i_subset (int): The subset number
            subset_overlapping_exposures (list of EventList): The list of EventList objects that overlap.

        """
        savedir = self.get_savedir(subset_number=i_subset)
        savedir.mkdir(parents=True, exist_ok=True)

        runid = self.generate_runid()

        # Create Merged EventList
        event_list = EventList.from_event_lists(subset_overlapping_exposures)

        # Run Data Loader
        dl = DataLoader(event_list=event_list, time_interval=self.time_interval, size_arcsec=self.size_arcsec,
                        gti_only=self.gti_only, min_energy=self.min_energy, max_energy=self.max_energy,
                        remove_partial_ccd_frames=self.remove_partial_ccd_frames)
        dl.run()

        cube_n = dl.data_cube
        # cube_n.video()

        img = self.observation.images[0]
        img.read(wcs_only=True)

        # Calculate Expectation Cube
        cube_mu = calc_cube_mu(cube_n, wcs=img.wcs)
        cube_mask_peaks, cube_mask_eclipses = self.precomputed_bayes_limit.get_cube_masks_peak_and_eclipse(cube_n=cube_n.data, cube_mu=cube_mu)

        image_n       = np.nansum(cube_n.data, axis=2)         # Total Counts.
        image_peak    = np.nansum(cube_mask_peaks, axis=2)     # Each Pixel is the number of peaks in cube_n
        image_eclipse = np.nansum(cube_mask_eclipses, axis=2)  # Each Pixel is the number of eclipses in cube_n
        image_mask_combined = (image_peak > 0) | (image_eclipse > 0)  # Combine peaks and eclipses

        df_reg = calc_df_regions(image=image_n, image_mask=image_mask_combined)
        df_sky = get_regions_sky_position(data_cube=cube_n, df_regions=df_reg, wcs=img.wcs)
        df_regions = pd.concat([df_reg, df_sky], axis=1).reset_index(drop=True)
        self.n_regions = len(df_regions)

        dfs_lcs = get_region_lightcurves(df_regions, cube_n, cube_mu)

        # Plot Lightcurves for each pixel.
        # x_peak, y_peak, t_peak = np.where(cube_mask_peaks)
        # x_eclipse, y_eclipse, t_eclipse = np.where(cube_mask_eclipses)
        # unique_xy = [*(get_unique_xy(x_peak, y_peak)), *(get_unique_xy(x_eclipse, y_eclipse))]
        # for x, y in unique_xy:
        #     plot_lc_pixel(cube_mu, cube_n, self.time_interval, x, y)

        # Plot Lightcurves for each region
        # if dfs_lcs:
        #     for df_lc in dfs_lcs:
        #         plot_df_lc(df_lc=df_lc, savedir=savedir)

        # Plot Image
        plot_detection_image(df_regions, image_eclipse, image_n, image_peak, savepath=savedir / 'detection_img.png')

        # Save Results
        results = {}
        # Collect DataFrames
        if dfs_lcs:
            for i, df_lc in enumerate(dfs_lcs):
                results[f'lc_{i}'] = df_lc

        results['bti']      = dl.df_bti
        results['regions']  = df_regions

        # Collect info
        results['obs_info'] = self.observation.info
        results['evt_info'] = event_list.info
        results['dl_info']  = dl.info
        results['dc_info']  = cube_n.info
        results['run_info'] = self.info

        for k, v in results.items():
            save_result(key=k, value=v, runid=self.runid, savedir=savedir)
        # plt.show()
        plt.close('all')
        plt.clf()

    def load_results(self):
        """Load results for all observation subsets, returns a list of dictionaries."""
        total_subsets = len(list(self.observation.path_results.glob('subset_*')))
        logger.debug(f'Found {total_subsets} subset folders')
        if total_subsets == 0:
            logger.debug('No results :(')
            return None

        all_results = []
        for i in range(total_subsets):
            results = self.load_subset_results(i)
            if results:
                all_results.append(results)
                for k, v in results.items():
                    logger.debug(f'{k}:')
                    logger.debug(v)
                    logger.debug('========')
        return all_results

    def load_subset_results(self, i):
        """Load results for a single observation subset. returns a dictionary."""
        self.get_savedir(subset_number=i)
        results = {}
        csv_files = list(self.savedir.glob('*.csv'))
        n_csv_files = len(csv_files)
        if n_csv_files == 0:
            logger.info(f'No csv files found in {self.savedir}!')
            return None
        for f in csv_files:
            if 'info' in f.stem:
                results[f.stem] = load_info(loadpath=f)
            else:
                results[f.stem] = load_df(loadpath=f)
        return results

    @property
    def info(self):
        info = {
            'runid'                     : self.runid,
            'obsid'                     : self.obsid,
            'subset_number'             : self.subset_number,
            'total_subsets'             : self.total_subsets,
            'size_arcsec'               : self.size_arcsec,
            'time_interval'             : self.time_interval,
            'gti_only'                  : self.gti_only,
            'remove_partial_ccd_frames' : self.remove_partial_ccd_frames,
            'min_energy'                : self.min_energy,
            'max_energy'                : self.max_energy,
            'clobber'                   : self.clobber,
            'precomputed_bayes_limit'   : self.precomputed_bayes_limit,
            'savedir'                   : self.savedir,
            'n_regions'                 : self.n_regions,
        }
        for k, v in info.items():
            logger.info(f'{k:>25} : {v}')
        return info


def plot_df_lc(df_lc, savedir=None, plot=True):
    if not plot:
        return None

    label = df_lc['label'].unique()[0]
    t0 = df_lc['time'] - df_lc['time'].min()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax2 = ax.twiny()
    ax.step(t0, df_lc['n'], where='post', color='black', lw=1.0, label='Observed (n)')
    ax.step(t0, df_lc['mu'], where='post', color='red', lw=1.0, label=r'Expected ($\mu$)')
    ax2.step(range(len(df_lc['n'])), df_lc['n'], where='post', color='none', lw=1.0)

    ax.set_title(f'Detected Region #{label}')
    ax.set_ylabel('Counts (N)')
    ax.set_xlabel('Time (s)')
    ax.set_xlim(t0.min(), t0.max())
    ax2.set_xlabel('Window/Frame Number')
    ax.legend(loc='upper right')
    plt.tight_layout()

    if savedir:
        savepath = savedir / f'lc_{label}.png'
        logger.info(f'Saving lightcurve plot to: {savepath}')
        plt.savefig(savepath)


def plot_detection_image(df_regions, image_eclipse, image_n, image_peak, savepath=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    # Map the peaks=1 and eclipses=2
    image_combined = np.zeros_like(image_peak)
    image_combined[image_peak > 0] = 1
    image_combined[image_eclipse > 0] = 2
    image_combined[(image_peak > 0) & (image_eclipse > 0)] = 3
    c_0 = 'none'
    c_peak = 'cyan'
    c_eclipse = 'lime'
    c_both = 'blue'
    cmap = ListedColormap(colors=[c_0, c_peak, c_eclipse, c_both])
    norm = BoundaryNorm(boundaries=[0, 1, 2, 3, 4], ncolors=4)
    norm2 = ImageNormalize(stretch=SqrtStretch())
    im1 = ax.imshow(image_n.T, cmap=cmap_image(), norm=norm2, interpolation='none', origin='lower')
    ax.imshow(image_combined.T, cmap=cmap, norm=norm, interpolation='none', origin='lower')
    cbar = plt.colorbar(im1, ax=ax, shrink=0.75)
    cbar.set_label('Total Counts')
    ax.scatter(df_regions['weighted_centroid-0'], df_regions['weighted_centroid-1'], marker='+', s=10, color='white')
    for i, row in df_regions.iterrows():
        x_cen = row['centroid-0']
        y_cen = row['centroid-1']

        width = row['bbox-2'] - row['bbox-0']
        height = row['bbox-3'] - row['bbox-1']

        x_pos = x_cen - width / 2
        y_pos = y_cen - height / 2

        rect = patches.Rectangle(xy=(x_pos, y_pos),
                                 width=width,
                                 height=height,
                                 linewidth=1,
                                 edgecolor='white',
                                 facecolor='none')

        plt.text(x_pos + width, y_pos + height, str(i), c='white')
        ax.add_patch(rect)
    lab_kwargs = {'markeredgecolor': None, 'marker': 's', 'markersize': 10, 'ls': 'none'}
    legend_labels = [
        plt.Line2D([0], [0], label='Peak', markerfacecolor=c_peak, **lab_kwargs),
        plt.Line2D([0], [0], label='Eclipse', markerfacecolor=c_eclipse, **lab_kwargs),
        plt.Line2D([0], [0], label='Peak & Eclipse', markerfacecolor=c_both, **lab_kwargs)
    ]
    ax.legend(handles=legend_labels)

    plt.tight_layout()
    if savepath:
        logger.info(f'Saving Image to {savepath}')
        plt.savefig(savepath)
    # plt.show()


def plot_lc_pixel(cube_mu, cube_n, time_interval, x, y, plot=False):
    if not plot:
        return None
    cube_mu_xy = cube_mu[x, y]
    cube_data_xy = cube_n.data[x, y]

    # Plot lightcurves
    lw = 1.0
    frame_axis = np.arange(cube_mu.shape[2])  # Frame Number
    time_axis  = frame_axis * time_interval   # Zero'ed Time
    time_axis2 = cube_n.bin_t[:-1]            # Observation Time

    mu_3sig, n_peak_3sig, n_eclipse_3sig = load_precomputed_bayes_limits(threshold_sigma=3)
    mu_5sig, n_peak_5sig, n_eclipse_5sig = load_precomputed_bayes_limits(threshold_sigma=5)

    fig, ax = plt.subplots(2, 1, figsize=(15, 5), gridspec_kw={'height_ratios': [10, 1]}, sharex=True)
    ax[0].fill_between(time_axis, n_eclipse_5sig(cube_mu_xy), n_peak_5sig(cube_mu_xy), alpha=0.3, facecolor='blue', label=r'5 $\sigma$')
    ax[0].fill_between(time_axis, n_eclipse_3sig(cube_mu_xy), n_peak_3sig(cube_mu_xy), alpha=0.5, facecolor='blue', label=r'3 $\sigma$')
    ax[0].step(time_axis, cube_data_xy, color='black', where='mid', lw=lw, label=r'Observed ($n$)')
    ax[0].step(time_axis, cube_mu_xy, color='red', where='mid', lw=lw, label=r'Expected ($\mu$)')
    ax[1].step(time_axis, cube_mu_xy, color='red', where='mid', lw=lw, label=r'Expected ($\mu$)')
    ax2 = ax[0].twiny()
    ax2.plot(frame_axis, cube_data_xy, color='none')
    ax2.set_xlabel("Frame #")
    ax[0].legend()
    ax[1].set_xlabel("Time (s)")
    ax[0].set_ylabel("Counts")
    ax[0].set_ylim(0)
    ax[0].set_xlim(np.min(time_axis), np.max(time_axis))
    plt.subplots_adjust(hspace=0)
    plt.suptitle(f'Lightcurve for pixel x={x} y={y}')


def get_region_lightcurves(df_regions, cube_n, cube_mu):
    if len(df_regions) == 0:
        logger.info('No Regions found, No lightcurves produced.')
        return None

    dfs = []
    for i, row in df_regions.iterrows():
        xlo, xhi = row['bbox-0'], row['bbox-2']
        ylo, yhi = row['bbox-1'], row['bbox-3']
        label    = int(row['label'])

        lc_n         = extract_lc(cube_n.data, xhi, xlo, yhi, ylo, dtype=np.int32)
        lc_mu        = extract_lc(cube_mu, xhi, xlo, yhi, ylo, dtype=np.float32)
        lc_B_peak    = B_peak_log(n=lc_n, mu=lc_mu)
        lc_B_eclipse = B_eclipse_log(n=lc_n, mu=lc_mu)

        lc = {'time'          : cube_n.bin_t[:-1],
              'bti'           : cube_n.bti_bin_idx_bool[:-1],
              'bccd'          : cube_n.bccd_bin_idx_bool,
              'n'             : lc_n,
              'mu'            : lc_mu,
              'B_peak_log'    : lc_B_peak,
              'B_eclipse_log' : lc_B_eclipse,
              'label'         : label}
        df_lc = pd.DataFrame(lc)
        logger.info(f'Lightcurve for Region {label}:\n{df_lc}')
        dfs.append(df_lc)
    return dfs


def parameter_grid(obsids):
    """
    usage:
    for params in parameter_grid(['0792180301', '0792180302', ...]:
        print(params)
        pipeline = Pipeline(**params)

    Parameters:
        obsids (list): list of obsids

    Returns
        params (dict): parameters for a run.
    """
    parameter_grid = {
        'obsid'                     : obsids,
        'size_arcsec'               : [20.0],
        'time_interval'             : [5, 50, 200],
        'gti_only'                  : [False],
        'remove_partial_ccd_frames' : [True],
        'energy_ranges'             : [[0.2, 2.0], [2.0, 12.0], [0.2, 12.0]],
        'clobber'                   : [False],
        'precomputed_bayes_limit'   : [PrecomputeBayesLimits(threshold_sigma=3)],
    }
    parameter_combinations = list(itertools.product(*parameter_grid.values()))
    logger.info(f'{len(parameter_combinations)} parameter combinations\n'
                f'predicted size ~{(500*len(parameter_combinations)) / 1e6} Gb\n'
                f'predicted runtime ~{1*len(parameter_combinations)/60/24:.2f} days (@ 1min/obs)')

    for combination in parameter_combinations:
        params = dict(zip(parameter_grid.keys(), combination))
        params['min_energy'] = params['energy_ranges'][0]
        params['max_energy'] = params['energy_ranges'][1]
        del(params['energy_ranges'])
        yield params


def combine_results(obsids):
    """
    Combine all results into a single DataFrame and save to results_combined.

    Parameters:
        obsids (list): list of observation IDs
    """
    all_results = []
    for params in parameter_grid(obsids=obsids):
        p = Pipeline(**params)
        results_list = p.load_results()
        if results_list:
            for r in results_list:
                all_results.append(r)

    # Combine all DataFrames
    df_bti      = pd.concat([r.get('bti') for r in all_results], axis=0)
    df_regions  = pd.concat([r.get('regions') for r in all_results], axis=0)
    df_lc       = pd.concat([df for r in all_results for key, df in r.items() if key.startswith('lc_')])
    df_run_info = pd.DataFrame([r['run_info'] for r in all_results])
    df_obs_info = pd.DataFrame([r['obs_info'] for r in all_results])
    df_dl_info  = pd.DataFrame([r['dl_info'] for r in all_results])
    df_dc_info  = pd.DataFrame([r['dc_info'] for r in all_results])
    df_evt_info = pd.DataFrame([r['evt_info'] for r in all_results])

    # Save all dataframes
    logger.info('Saving DataFrames...')
    df_bti.to_csv(savepaths_combined['bti'], index=False)
    df_regions.to_csv(savepaths_combined['regions'], index=False)
    df_lc.to_csv(savepaths_combined['lc'], index=False)
    df_run_info.to_csv(savepaths_combined['run_info'], index=False)
    df_obs_info.to_csv(savepaths_combined['obs_info'], index=False)
    df_dl_info.to_csv(savepaths_combined['dl_info'], index=False)
    df_dc_info.to_csv(savepaths_combined['dc_info'], index=False)
    df_evt_info.to_csv(savepaths_combined['evt_info'], index=False)

    logger.info(df_bti)
    logger.info(df_regions)
    logger.info(df_lc)
    logger.info(df_run_info)
    logger.info(df_obs_info)
    logger.info(df_dl_info)
    logger.info(df_dc_info)
    logger.info(df_evt_info)


if __name__=="__main__":

    obsids = read_observation_ids(data / 'observations.txt')
    # obsids = read_observation_ids(data / 'all_obsids.txt')
    shuffle(obsids)

    for params in parameter_grid(obsids=obsids):
        p = Pipeline(**params)
        p.run()

    combine_results()
