from exod.pre_processing.event_filtering import sas_is_installed
from exod.processing.bti import get_bti, get_gti_threshold
from exod.utils.logger import logger
from exod.utils.path import data, read_observation_ids, savepaths_combined
from exod.utils.util import save_result, load_info, load_df
from exod.utils.plotting import cmap_image
from exod.processing.bayesian_computations import PrecomputeBayesLimits, B_peak_log, B_eclipse_log
from exod.processing.data_cube import extract_lc, DataCube
from exod.processing.background_inference import calc_cube_mu
from exod.processing.coordinates import get_regions_sky_position, calc_df_regions
from exod.xmm.event_list import EventList
from exod.xmm.observation import Observation
from exod.xmm.wcs import wcs_from_eventlist_file

import itertools
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import colors
import pandas as pd
from astropy.visualization import ImageNormalize, SqrtStretch


class Pipeline:
    """
    Pipeline to process a single XMM-Newton observation and find variable sources.

        Attributes:
        obsid (str): Observation ID.
        size_arcsec (float): Size of the binned pixels in arcseconds.
        time_interval (int): Time interval in seconds.
        gti_only (bool): Use only Good Time Intervals.
        remove_partial_ccd_frames (bool): Remove partial CCD frames.
        min_energy (float): Minimum energy in keV.
        max_energy (float): Maximum energy in keV.
        clobber (bool): Overwrite existing files.
        threshold_sigma (float): Sigma threshold for variability detection.
        precomputed_bayes_limit (PrecomputeBayesLimits): Precomputed Bayes Limits object.
        observation (Observation): Observation object.
        
        runid (str): Unique identifier for the run.
        event_list (EventList): EventList() object.
        data_cube (XMMDataCube): XMMDataCube() object containing the number of photons in each cell.
        subset_number (int): The subset number.
        total_subsets (int): Total number of subsets.
        gti_threshold (float): Threshold for Good Time Intervals.
        n_regions (int): Number of regions detected.
        
        savedir (Path): Directory to save results to.
    """
    def __init__(self, obsid='0724210501', size_arcsec=20.0, time_interval=50, gti_only=False, remove_partial_ccd_frames=True,
                 min_energy=0.2, max_energy=10.0, clobber=False, threshold_sigma=3, **kwargs):
        self.obsid = obsid
        self.size_arcsec = size_arcsec
        self.time_interval = time_interval
        self.gti_only = gti_only
        self.remove_partial_ccd_frames = remove_partial_ccd_frames
        self.min_energy = min_energy
        self.max_energy = max_energy
        self.clobber = clobber
        self.threshold_sigma = threshold_sigma
        self.precomputed_bayes_limit = PrecomputeBayesLimits(threshold_sigma)
        self.observation = Observation(obsid)

        self.runid = None
        self.event_list = None
        self.data_cube = None
        self.subset_number = None
        self.total_subsets = None
        self.gti_threshold = None
        self.n_regions = None

        self.savedir = None

    def generate_runid(self):
        """Generate a unique runid of the form 0765080801_0_50_0.2_12.0"""
        runid = f'{self.obsid}_{self.subset_number}_{self.time_interval}_{self.min_energy}_{self.max_energy}'
        self.runid = runid
        return runid

    def set_savedir(self):
        """Create the directory to save results to."""
        self.savedir = self.observation.path_results / self.generate_runid()
        self.savedir.mkdir(parents=True, exist_ok=True)
        logger.info(f'savedir set to: {self.savedir}')
        return self.savedir

    def pre_process(self):
        """Pre-process the data."""
        self.observation.download_events()
        self.observation.filter_events(clobber=self.clobber)
        self.observation.create_images(clobber=self.clobber)
        self.observation.get_files()
        self.observation.get_events_overlapping_subsets()
        self.subsets       = self.observation.get_events_overlapping_subsets()
        self.total_subsets = self.observation.get_number_of_overlapping_subsets()

    def run(self):
        self.pre_process()
        for i, s in enumerate(self.subsets):
            self.run_subset(subset_number=i, subset_overlapping_exposures=s)

    def run_subset(self, subset_number, subset_overlapping_exposures):
        self.subset_number = subset_number
        self.set_savedir()
        self.generate_runid()

        self.event_list = EventList.from_event_lists(subset_overlapping_exposures)
        self.calculate_bti()
        self.event_list.filter_by_energy(self.min_energy, self.max_energy)

        self.data_cube = DataCube(self.event_list, self.size_arcsec, self.time_interval)
        self.data_cube.calc_gti_bti_bins(bti=self.bti)
        self.data_cube.mask_frames_with_partial_ccd_exposure(mask_frames=self.remove_partial_ccd_frames)
        if self.gti_only:
            self.data_cube.mask_bti()
        cube_n = self.data_cube
        
        if sas_is_installed():
            img = self.observation.images[0]
            img.read(wcs_only=True)
            wcs = img.wcs
        else:
            wcs = wcs_from_eventlist_file(self.observation.events_raw[0].path)

        # Calculate Expectation Cube
        cube_mu = calc_cube_mu(cube_n, wcs=wcs)
        cube_mask_peaks, cube_mask_eclipses = self.precomputed_bayes_limit.get_cube_masks_peak_and_eclipse(cube_n=cube_n.data, cube_mu=cube_mu)

        # Extract information for each significant data cell (3D pixel)
        df_significant_cells = self.get_significant_cells(cube_mask_peaks, cube_mask_eclipses, cube_n)
        self.filter_events_from_significant_cells(df_significant_cells, self.event_list)
        unique_xy = self.get_significant_pixels_from_cells(df_significant_cells)

        # Plot Lightcurves for each cell.
        for x_cube, y_cube in unique_xy:
            plot_pixel_lc(cube_mu, cube_n, x_cube, y_cube, plot=True)

        # Get Images.
        image_n       = np.nansum(cube_n.data, axis=2)                # Total Counts.
        image_peak    = np.nansum(cube_mask_peaks, axis=2)            # Each Pixel is the number of peaks in cube_n
        image_eclipse = np.nansum(cube_mask_eclipses, axis=2)         # Each Pixel is the number of eclipses in cube_n
        image_mask_combined = (image_peak > 0) | (image_eclipse > 0)  # Combine peaks and eclipses

        df_reg = calc_df_regions(image=image_n, image_mask=image_mask_combined)
        df_reg['detid'] = self.runid + '_' + df_reg['label'].astype('str') # Create detection id from runid_label
        df_sky = get_regions_sky_position(data_cube=cube_n, df_regions=df_reg, wcs=wcs)
        df_regions = pd.concat([df_reg, df_sky], axis=1).reset_index(drop=True)
        self.n_regions = len(df_regions)

        dfs_lcs = get_region_lightcurves(df_regions, cube_n, cube_mu)

        # Plot Lightcurves for each region
        if dfs_lcs:
            for df_lc in dfs_lcs:
                plot_df_lc(df_lc=df_lc, savedir=self.savedir)

        # Plot Image
        plot_detection_image(df_regions, image_eclipse, image_n, image_peak, savepath=self.savedir / 'detection_img.png')

        # Save Results
        self.save_results(dc_info=cube_n.info, df_alerts=df_significant_cells, df_regions=df_regions, dfs_lcs=dfs_lcs,
                          df_bti=self.df_bti, evt_info=self.event_list.info)
        # plt.show()
        # plt.close('all')
        # plt.clf()

    def calculate_bti(self):
        """Calculate the bad time intervals (BTI) based upon the lightcurve from the eventlist and the threshold."""
        self.gti_threshold = get_gti_threshold(self.event_list.N_event_lists)
        t_bin_he, lc_he = self.event_list.get_high_energy_lc(self.time_interval)
        self.bti = get_bti(time=t_bin_he, data=lc_he, threshold=self.gti_threshold)
        self.df_bti = pd.DataFrame(self.bti)

    def multiply_time_interval(self, n_factor):
        logger.info(f'Rebinning the cube with longer timebins by factor {n_factor}...')
        self.data_cube.multiply_time_interval(n_factor)
        self.time_interval = self.data_cube.time_interval
        self.calculate_bti()
        self.data_cube.calc_gti_bti_bins(bti=self.bti)

    def get_significant_pixels_from_cells(self, df_significant_cells):
        """Extract the unique x,y values (2D Pixels) from the significant (3D) cells."""
        if len(df_significant_cells) == 0:
            return []
        unique_xy = list(df_significant_cells.groupby(['x_cube', 'y_cube']).groups.keys())
        return unique_xy

    def filter_events_from_significant_cells(self, df_alerts, event_list):
        """Extract the events in the events list for each significant cell."""
        for i, r in df_alerts.iterrows():
            evt_subset = event_list.get_events_within_bounds(X_lo=r['X_lo'], X_hi=r['X_hi'],
                                                             Y_lo=r['Y_lo'], Y_hi=r['Y_hi'],
                                                             TIME_lo=r['TIME_lo'], TIME_hi=r['TIME_hi'])
            logger.info(r)
            logger.info(evt_subset)
            logger.info(f'N_events = {len(evt_subset)}')

            """
            # Create an image of the significant cell.
            for instrument in np.unique(evt_subset['INSTRUMENT']):
                sub = evt_subset[evt_subset['INSTRUMENT'] == instrument]
                rawx = sub['RAWX']
                rawy = sub['RAWY']
                xbins = np.arange(rawx.min()-1, rawx.max()+1, 1)
                ybins = np.arange(rawy.min()-1, rawy.max()+1, 1)

                try:
                    hist, xedges, yedges = np.histogram2d(rawx, rawy, bins=(xbins, ybins))

                    n = int(hist.max()) + 1
                    cmap = plt.get_cmap('hot', n)
                    boundaries = np.arange(-0.5, n + 0.5, 1)
                    norm = colors.BoundaryNorm(boundaries, ncolors=n)

                    plt.figure()
                    plt.title(f'{instrument} ({r['x_cube']},{r['y_cube']},{r['t_cube']})')
                    plt.imshow(hist.T, origin="lower", aspect="equal", cmap=cmap, norm=norm,
                               extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]), interpolation='none')
                    plt.colorbar(shrink=0.5)
                    plt.xlabel("RAWX")
                    plt.ylabel("RAWY")
                    plt.subplots_adjust(left=0, right=0, top=0, bottom=0)
                except Exception as e:
                    print(f'Could not do {i} {e}')
            """

    def get_significant_cells(self, cube_mask_peaks, cube_mask_eclipses, cube_n):
        """Extract information for each pixel in the datacube that is associated with an alert."""
        logger.info('Extracting significant 3D pixels from data cube.')
        # Get the positions (x,y,t) of the bursts and eclipses
        peak_positions    = np.argwhere(cube_mask_peaks == 1)
        eclipse_positions = np.argwhere(cube_mask_eclipses == 1)

        alerts = []
        for pos in peak_positions:
            pixel_alert_info = self.extract_pixel_alert_info(cube_n, pos)
            pixel_alert_info['type'] = 'peak'
            alerts.append(pixel_alert_info)

        for pos in eclipse_positions:
            pixel_alert_info = self.extract_pixel_alert_info(cube_n, pos)
            pixel_alert_info['type'] = 'eclipse'
            alerts.append(pixel_alert_info)

        df_alerts = pd.DataFrame(alerts)
        logger.info('df_alerts:')
        logger.info(df_alerts)
        return df_alerts

    def extract_pixel_alert_info(self, cube_n, pixel_xyt_pos):
        x_cube, y_cube, t_cube = pixel_xyt_pos
        X    = cube_n.bin_x[x_cube]
        Y    = cube_n.bin_y[y_cube]
        TIME = cube_n.bin_t[t_cube]

        X_size, Y_size, TIME_size = cube_n.pixel_size, cube_n.pixel_size, self.time_interval

        t_depth   = 2
        TIME_size = t_depth * TIME_size

        alert = {'x_cube'    : x_cube,
                 'y_cube'    : y_cube,
                 't_cube'    : t_cube,
                 'X'         : X,
                 'Y'         : Y,
                 'TIME'      : TIME,
                 'X_size'    : X_size,
                 'Y_size'    : Y_size,
                 'TIME_size' : TIME_size,
                 'X_lo'      : (X - X_size),
                 'X_hi'      : (X + X_size),
                 'Y_lo'      : (Y - Y_size),
                 'Y_hi'      : (Y + Y_size),
                 'TIME_lo'   : (TIME - TIME_size),
                 'TIME_hi'   : (TIME + TIME_size)}
        return alert

    def save_results(self, dc_info, df_alerts, df_regions, dfs_lcs, df_bti, evt_info):
        results = {}
        # Collect DataFrames
        if dfs_lcs:
            for i, df_lc in enumerate(dfs_lcs):
                results[f'lc_{i}'] = df_lc

        results['bti']     = df_bti
        results['regions'] = df_regions
        results['alerts']  = df_alerts

        # Collect info
        results['obs_info'] = self.observation.info
        results['evt_info'] = evt_info
        results['dc_info']  = dc_info
        results['run_info'] = self.info
        for k, v in results.items():
            save_result(key=k, value=v, runid=self.runid, savedir=self.savedir)

    def load_subset_results(self, subset_number):
        """Load results for a single observation subset. returns a dictionary."""
        self.subset_number = subset_number
        self.set_savedir()
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

    def load_results(self):
        """Load results for all observation subsets, returns a list of dictionaries."""
        all_results = [self.load_subset_results(i) for i in range(self.total_subsets)]
        return all_results

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
            'gti_threshold'             : self.gti_threshold,
            'remove_partial_ccd_frames' : self.remove_partial_ccd_frames,
            'min_energy'                : self.min_energy,
            'max_energy'                : self.max_energy,
            'clobber'                   : self.clobber,
            'threshold_sigma'           : self.threshold_sigma,
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

    c_0       = 'none'
    c_peak    = 'cyan'
    c_eclipse = 'lime'
    c_both    = 'blue'
    cmap = ListedColormap(colors=[c_0, c_peak, c_eclipse, c_both])

    norm  = BoundaryNorm(boundaries=[0, 1, 2, 3, 4], ncolors=4)
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


def plot_pixel_lc(cube_mu, cube_n, x, y, plot=True):
    """Plot the lightcurve for a specific pixel (x,y), shows the three and five sigma error regions."""
    if not plot:
        return None
    cube_mu_xy = cube_mu[x, y]
    cube_data_xy = cube_n.data[x, y]
    time_interval = cube_n.time_interval

    # Plot lightcurves
    lw = 1.0
    frame_axis = np.arange(cube_mu.shape[2])  # Frame Number
    time_axis  = frame_axis * time_interval   # Zero'ed Time
    time_axis2 = cube_n.bin_t[:-1]            # Observation Time

    pbl_3_sig = PrecomputeBayesLimits(threshold_sigma=3)
    pbl_5_sig = PrecomputeBayesLimits(threshold_sigma=5)

    fig, ax = plt.subplots(2, 1, figsize=(15, 5), gridspec_kw={'height_ratios': [10, 1]}, sharex=True)
    ax[0].fill_between(time_axis, pbl_5_sig.n_eclipse_threshold(cube_mu_xy), pbl_5_sig.n_peak_threshold(cube_mu_xy), alpha=0.3, facecolor='steelblue', label=r'5 $\sigma$')
    ax[0].fill_between(time_axis, pbl_3_sig.n_eclipse_threshold(cube_mu_xy), pbl_3_sig.n_peak_threshold(cube_mu_xy), alpha=0.5, facecolor='steelblue', label=r'3 $\sigma$')
    ax[0].step(time_axis, cube_data_xy, color='black', where='mid', lw=lw, label=r'Observed ($n$)')
    ax[0].step(time_axis, cube_mu_xy, color='red', where='mid', lw=lw, label=r'Expected ($\mu$)')
    ax[1].step(time_axis, cube_mu_xy, color='red', where='mid', lw=lw, label=r'Expected ($\mu$)')
    ax2 = ax[0].twiny()
    ax2.plot(frame_axis, cube_data_xy, color='none')
    ax2.set_xlabel("Frame #")
    ax[0].legend()
    ax[1].set_xlabel("Time (s)")
    ax[0].set_ylabel("Counts")
    ax[1].set_ylabel("bkg")
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
        'time_interval'             : [300, 600, 1000],
        'gti_only'                  : [False],
        'remove_partial_ccd_frames' : [True],
        'energy_ranges'             : [[0.2, 2.0], [2.0, 12.0], [0.2, 12.0]],
        'clobber'                   : [False],
        'threshold_sigma'           : [3],
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


def make_df_lc_idx(df_lc):
    """
    Calculate the start and end indexs of each of the lightcurves.
    This is done so that a specific runid + label combination can be quickly accessed.
    """
    df_lc = df_lc.reset_index(drop=True)
    tot = 0
    combinations = {}
    for (label, runid), i in df_lc.groupby(['label', 'runid']).groups.items():
        combinations[(label, runid)] = (i[0], i[-1])

    df_start_stop = pd.DataFrame.from_dict(combinations, orient='index', columns=['start', 'stop'])
    df_start_stop = df_start_stop.sort_values('start')
    df_start_stop.to_csv(savepaths_combined['lc_idx'])
    logger.info(f'Saved df_lc_idx to {savepaths_combined["lc_idx"]}')


def combine_results(obsids):
    """
    Get the results for each of the observations ids and combine them into joint dataframes and save them into
    their respective paths in exod.utils.path.data_combined.

    Will raise an error if there are already files data_combined, to avoid accidentally deleting data.

    Parameters:
        obsids (list): list of observation IDs
    """
    if any([p.exists() for p in savepaths_combined.values()]):
        raise FileExistsError(f'There are already some combined files in the combined data path! '
                              f'Be careful! You dont want to accidentally overwrite these!')

    all_results = []
    for params in parameter_grid(obsids=obsids):
        p = Pipeline(**params)
        results_list = p.load_results()
        if results_list:
            for r in results_list:
                all_results.append(r)

    logger.info('Combining all DataFrames')
    df_bti      = pd.concat([r.get('bti') for r in all_results], axis=0)
    df_regions  = pd.concat([r.get('regions') for r in all_results], axis=0)
    df_alerts   = pd.concat([r.get('alerts') for r in all_results], axis=0)
    df_lc       = pd.concat([df for r in all_results for key, df in r.items() if key.startswith('lc_')])
    df_run_info = pd.DataFrame([r['run_info'] for r in all_results])
    df_obs_info = pd.DataFrame([r['obs_info'] for r in all_results])
    df_dc_info  = pd.DataFrame([r['dc_info'] for r in all_results])
    df_evt_info = pd.DataFrame([r['evt_info'] for r in all_results])

    make_df_lc_idx(df_lc)

    logger.info('Saving all DataFrames...')
    df_bti.to_csv(savepaths_combined['bti'], index=False)
    df_regions.to_csv(savepaths_combined['regions'], index=False)
    df_alerts.to_csv(savepaths_combined['alerts'], index=False)
    df_lc.to_hdf(savepaths_combined['lc'], key='df_lc', index=False, mode='w', format='table')
    df_run_info.to_csv(savepaths_combined['run_info'], index=False)
    df_obs_info.to_csv(savepaths_combined['obs_info'], index=False)
    df_dc_info.to_csv(savepaths_combined['dc_info'], index=False)
    df_evt_info.to_csv(savepaths_combined['evt_info'], index=False)

    logger.info(df_bti)
    logger.info(df_regions)
    logger.info(df_alerts)
    logger.info(df_lc)
    logger.info(df_run_info)
    logger.info(df_obs_info)
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
