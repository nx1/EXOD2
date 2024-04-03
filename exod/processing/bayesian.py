import exod.utils.path as path
from exod.utils.plotting import cmap_image
from exod.utils.logger import logger
from exod.pre_processing.data_loader import DataLoader
from exod.utils.util import save_df, save_info
from exod.xmm.event_list import EventList
from exod.xmm.observation import Observation
from exod.processing.template_based_background_inference import compute_expected_cube_using_templates
from exod.processing.coordinates import get_regions_sky_position, calc_df_regions

from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, BoundaryNorm
import pandas as pd
from scipy.stats import poisson
from scipy.interpolate import interp1d
from scipy.special import gammainc, gammaincc
from astropy.visualization import ImageNormalize, SqrtStretch
from tqdm import tqdm


def B_peak(n, mu):
    """Computes the Bayes factors of the presence of a peak, given the data_cube (mu) and observed (n) counts."""
    return gammaincc(n + 1, mu) / poisson.pmf(n, mu)

def B_eclipse(n, mu):
    """Computes the Bayes factors of the presence of an eclipse, given the data_cube (mu) and observed (n) counts"""
    return gammainc(n + 1, mu) / poisson.pmf(n, mu)

def B_peak_log(n, mu):
    """Computes the Bayes factors of the presence of a peak, given the data_cube (mu) and observed (n) counts."""
    return np.log10(gammaincc(n + 1, mu)) - np.log10(poisson.pmf(n, mu))


def B_eclipse_log(n, mu):
    """Computes the Bayes factors of the presence of an eclipse, given the data_cube (mu) and observed (n) counts"""
    return np.log10(gammainc(n + 1, mu)) - np.log10(poisson.pmf(n, mu))


def n_peak_large_mu(mu, sigma):
    """Calculate the observed (n) value required for a peak at a specific expectation (mu) and significance (sigma) in the guassian regime."""
    return np.ceil((2*mu+sigma**2+np.sqrt(8*mu*(sigma**2)+sigma**4))/2)


def n_eclipse_large_mu(mu, sigma):
    """Calculate the observed (n) value required for an eclipse at a specific expecation (mu) and significance (sigma) in the guassian regime."""
    return np.floor((2*mu+sigma**2-np.sqrt(8*mu*(sigma**2)+sigma**4))/2)


def get_bayes_thresholds(threshold_sigma):
    """
    The thresholds for B_peak and B_eclipse are calculated here for 3 and 5 sigma.

    This is sort of a hack, and is done by finding the value of B(n,mu) that is equal to
    a given significance (sigma) under the Gaussian assumption at mu=1000.

    For example, we want a significance level of sigma = 3.
    We first find what observed (n) value we need to get a 3 sigma peak Gaussian assumption.
        n_peak_large_mu(mu=1000, sigma=3) = 1139

    Next, we find the value of B_peak that an observed (n) value of 1139 would give us.
        B_peak(n=1139, mu=1000) = 872908  (5.94 in log10)

    We treat this value of B as being "Equivalent" to a 3 sigma detection and subsequently can specify:
        B > B_threshold_peak for a peak detection
        B < B_eclipse_threshold for an eclipse detection
    """
    B_peak_threshold    = B_peak_log(n=n_peak_large_mu(mu=1000, sigma=threshold_sigma), mu=1000)
    B_eclipse_threshold = B_eclipse_log(n=n_eclipse_large_mu(mu=1000, sigma=threshold_sigma), mu=1000)
    return B_peak_threshold, B_eclipse_threshold


def precompute_bayes_limits(threshold_sigma):
    """
    Compute the minimum and maximum number of observed
    counts (n) required for an eclipse or peak for a given
    confidence threshold (threshold_sigma) and expectation (mu).

    For counts > 1000, we use a Gaussian approximation.
    sigma = (N-mu) / (N+mu)^0.5
    
    Solving for N gives:
    N = 2mu + sigma^2 + (8 mu sigma^2 + sigma^4)^0.5
    N = 2mu + sigma^2 - (8 mu sigma^2 + sigma^4)^0.5

    The resulting table looks like:
        i          mu  n_peak  n_eclipse
    46000  158.556483   219.0      106.0
    46001  158.629519   219.0      107.0
    46002  158.702589   219.0      107.0
    46003  158.775693   219.0      107.0


    Each data cell in the observed and data_cube cube_n
    is then compared to the values pre-calculated here
    to determine if is it a peak or an eclipse.
    
    An example for 1 frame is given here.

     (n_cube)    (mu_cube)       is_3_sig_peak? 
      0 2 1    0.03 1.98 0.87       F F F
      0 5 0    1.01 1.10 1.00       F T F 
      3 2 1    2.30 2.14 0.98       F F F

    The evaluation of each data cell of the cube_n against
    a threshold is made faster by precomputing the counts here.
    """
    B_peak_threshold, B_eclipse_threshold = get_bayes_thresholds(threshold_sigma=threshold_sigma)

    range_mu       = np.geomspace(start=1e-7, stop=1e3, num=50000) #
    range_mu_large = np.geomspace(start=1e3, stop=1e6, num=1000)   # Above 1000 

    tab_npeak, tab_neclipse = [], []
    for mu in tqdm(range_mu):
        # Find the smallest observed (n) value for a peak
        range_n_peak = np.arange(max(10 * int(mu), 100))
        B_factors = B_peak_log(n=range_n_peak, mu=mu)
        n_peak_min = range_n_peak[B_factors > B_peak_threshold][0]
        tab_npeak.append(n_peak_min)

        # Get the largest observed (n) value for an eclipse
        range_n_eclipse = np.arange(2 * int(mu) + 1)
        B_factors = B_eclipse_log(n=range_n_eclipse, mu=mu)
        n_eclipse_max = range_n_eclipse[B_factors < B_eclipse_threshold][0]
        tab_neclipse.append(n_eclipse_max)
        
    tab_npeak    += list(n_peak_large_mu(range_mu_large, threshold_sigma))
    tab_neclipse += list(n_eclipse_large_mu(range_mu_large, threshold_sigma))
    
    range_mu = np.concatenate((range_mu, range_mu_large))

    data = np.array([range_mu, tab_npeak, tab_neclipse])
    savepath = path.utils / f'bayesfactorlimits_{threshold_sigma}.txt'
    logger.info(f'Saving to {savepath}')
    np.savetxt(savepath, data)

    # Visualization
    plt.figure(figsize=(5, 5))
    plt.plot(range_mu, range_mu, label=r'$N=\mu$', color='red')
    plt.plot(range_mu, tab_npeak, ls=':', c='k', label=fr'N with $B_{{peak}} > 10^{{{B_peak_threshold:.2f}}}$', lw=1.0)
    plt.plot(range_mu, tab_neclipse, ls='--', c='k', label=f'N with $B_{{eclipse}} > 10^{{{B_eclipse_threshold:.2f}}}$', lw=1.0)
    plt.fill_between(range_mu, range_mu-5*np.sqrt(range_mu), range_mu+5*np.sqrt(range_mu), alpha=0.2, label=fr'$5 \sigma$ Region', color='blue')
    plt.fill_between(range_mu, range_mu-3*np.sqrt(range_mu), range_mu+3*np.sqrt(range_mu), alpha=0.5, label=fr'$3 \sigma$ Region', color='blue')
    plt.yscale('log')
    plt.xscale('log')
    plt.title(r'$B_{peak} = \frac{Q(N+1, \mu)}{e^{-\mu} \mu^{N} / N!} \ \  B_{eclipse} = \frac{P(N+1, \mu)}{e^{-\mu} \mu^{N} / N!}$')
    plt.xlabel(fr'Expected Counts $\mu$')
    plt.ylabel(fr'Observed Counts $N$')
    plt.xlim(min(range_mu), max(range_mu))
    plt.ylim(min(range_mu), max(range_mu))
    plt.legend()
    plt.tight_layout()
    plt.show()


def precompute_bayes_1000():
    """Precomputes the Bayes factor at mu=1000 for a bunch of values of N. Will be interpolated to estimate the sigma"""
    range_N    = np.arange(10000)
    B_peaks    = B_peak_log(n=range_N, mu=1000)
    B_eclipses = B_eclipse_log(n=range_N, mu=1000)
    data = np.array([range_N, B_peaks, B_eclipses])
    savepath = path.utils / f'bayesfactor_mu1000.txt'
    logger.info(f'Saving to {savepath}')
    np.savetxt(savepath, data)


def load_precomputed_bayes1000():
    """Loads & interpolates the precomputed values of Bayes factors at mu=1000"""
    data              = np.loadtxt(path.utils / f'bayesfactor_mu1000.txt')
    range_N           = data[0]
    B_values_peaks    = interp1d(range_N, data[1])
    B_values_eclipses = interp1d(range_N, data[2])
    return range_N, B_values_peaks, B_values_eclipses


def load_precomputed_bayes_limits(threshold_sigma):
    """Loads the precomputed Bayes factor limit numbers, for a chosen threshold."""
    data = np.loadtxt(path.utils / f'bayesfactorlimits_{threshold_sigma}.txt')
    range_mu = data[0]
    n_peak_threshold = interp1d(range_mu, data[1])
    n_eclipse_threshold = interp1d(range_mu, data[2])
    return range_mu, n_peak_threshold, n_eclipse_threshold


def get_cube_masks_peak_and_eclipse(cube_n, cube_mu, threshold_sigma):
    """Returns two cubes with booleans where the rate correspond to a peak or an eclipse."""
    range_mu, n_peak_threshold, n_eclipse_threshold = load_precomputed_bayes_limits(threshold_sigma=threshold_sigma)
    cube_mu = np.where(cube_mu > range_mu[0], cube_mu, np.nan) # Remove small expectation values outside of interpolation range
    cube_mask_peaks   = cube_n > n_peak_threshold(cube_mu)
    cube_mask_eclipse = cube_n < n_eclipse_threshold(cube_mu)
    return cube_mask_peaks, cube_mask_eclipse


def get_unique_xy(x, y):
    """Get the unique pairs of two lists."""
    unique_xy = set()  # Set to store unique pairs
    for x, y in zip(x, y):
        unique_xy.add((x, y))
    return unique_xy

def plot_region_lightcurve(df_lcs, i, savepath=None):
    """Plot the ith region lightcurve."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax2 = ax.twiny()
    t0 = df_lcs['time'] - df_lcs['time'].min()
    ax.step(t0, df_lcs[f'n_{i}'], where='post', color='black', lw=1.0, label='Observed (n)')
    ax.step(t0, df_lcs[f'mu_{i}'], where='post', color='red', lw=1.0, label=r'Expected ($\mu$)')

    ax2.step(range(len(df_lcs[f'n_{i}'])), df_lcs[f'n_{i}'], where='post', color='none', lw=1.0)

    ax.set_title(f'Detected Region #{i}')
    ax.set_ylabel('Counts (N)')
    ax.set_xlabel('Time (s)')
    ax.set_xlim(t0.min(), t0.max())
    ax2.set_xlabel('Window/Frame Number')
    ax.legend(loc='upper right')
    plt.tight_layout()

    if savepath:
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
        ind   = row['label']
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

        plt.text(x_pos + width, y_pos + height, str(ind), c='white')
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


def plot_lc_pixel(cube_mu, cube_n, time_interval, x, y):
    cube_mu_xy = cube_mu[x, y]
    cube_data_xy = cube_n.data[x, y]
    # Plot lightcurves
    lw = 1.0
    frame_axis = np.arange(cube_mu.shape[2])  # Frame Number
    time_axis = frame_axis * time_interval  # Zero'ed Time
    time_axis2 = cube_n.bin_t[:-1]  # Observation Time
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


def get_region_lightcurves(df_regions, cube_n, cube_mu, savepath=None):
    if len(df_regions) == 0:
        logger.info('No Regions found, No lightcurves produced.')
        return None
    lcs = [pd.DataFrame({'time' : cube_n.bin_t[:-1]}),
           pd.DataFrame({'bti'  : cube_n.bti_bin_idx_bool[:-1]}),
           pd.DataFrame({'bccd' : cube_n.bccd_bin_idx_bool})]
    for i, row in df_regions.iterrows():
        xlo, xhi = row['bbox-0'], row['bbox-2']
        ylo, yhi = row['bbox-1'], row['bbox-3']

        lc_n         = extract_lc_from_cube(cube_n.data, xhi, xlo, yhi, ylo, dtype=np.int32)
        lc_mu        = extract_lc_from_cube(cube_mu, xhi, xlo, yhi, ylo, dtype=np.float32)
        lc_B_peak    = B_peak(n=lc_n, mu=lc_mu)
        lc_B_eclipse = B_eclipse(n=lc_n, mu=lc_mu)

        lcs.append(pd.DataFrame({f'n_{i}': lc_n}))
        lcs.append(pd.DataFrame({f'mu_{i}': lc_mu}))
        lcs.append(pd.DataFrame({f'B_peak_{i}': lc_B_peak}))
        lcs.append(pd.DataFrame({f'B_eclipse_{i}': lc_B_eclipse}))
    df_lcs = pd.concat(lcs, axis=1)
    if savepath:
        logger.info(f'Saving lightcurve plot to: {savepath}')
        plt.savefig(savepath)
    return df_lcs


def extract_lc_from_cube(data_cube, xhi, xlo, yhi, ylo, dtype=np.int32):
    data = data_cube[xlo:xhi, ylo:yhi]
    lc = np.nansum(data, axis=(0, 1), dtype=dtype)
    return lc


def run_pipeline(obsid,
                 size_arcsec=20,
                 time_interval=5,
                 gti_only=False,
                 remove_partial_ccd_frames=True,
                 gti_threshold=1.5,
                 min_energy=0.5,
                 max_energy=10.0,
                 threshold_sigma=3):

    observation = Observation(obsid)
    observation.get_files()
    observation.get_events_overlapping_subsets()
    for i_subset, subset_overlapping_exposures in enumerate(observation.events_overlapping_subsets):
        savedir = observation.path_results / f'subset_{i_subset}'
        savedir.mkdir(exist_ok=True)
        
        
        event_list = EventList.from_event_lists(subset_overlapping_exposures)
        dl = DataLoader(event_list=event_list, time_interval=time_interval, size_arcsec=size_arcsec,
                        gti_only=gti_only, min_energy=min_energy, max_energy=max_energy,
                        gti_threshold=gti_threshold, remove_partial_ccd_frames=remove_partial_ccd_frames)
        dl.run()

        img = observation.images[0]
        img.read(wcs_only=True)

        cube_n = dl.data_cube
        cube_mu = compute_expected_cube_using_templates(cube_n, wcs=img.wcs)
        cube_mask_peaks, cube_mask_eclipses = get_cube_masks_peak_and_eclipse(cube_n=cube_n.data, cube_mu=cube_mu, threshold_sigma=threshold_sigma)

        image_n       = np.nansum(cube_n.data, axis=2)        # Total Counts.
        image_peak    = np.nansum(cube_mask_peaks, axis=2)    # Each Pixel is the number of peaks in cube_n
        image_eclipse = np.nansum(cube_mask_eclipses, axis=2) # Each Pixel is the number of eclipses in cube_n

        image_mask_combined = (image_peak > 0) | (image_eclipse > 0)
        df_reg = calc_df_regions(image=image_n, image_mask=image_mask_combined)
        df_sky = get_regions_sky_position(data_cube=cube_n, df_regions=df_reg, wcs=img.wcs)
        df_regions = pd.concat([df_reg, df_sky], axis=1).reset_index(drop=True)
        df_lcs = get_region_lightcurves(df_regions, cube_n, cube_mu)

        # Plot Lightcurves for each region
        for i in df_regions.index:
            plot_region_lightcurve(df_lcs=df_lcs, i=i, savepath=savedir / f'lc_{i}.png')

        # Plot Lightcurves for each pixel.
        # x_peak, y_peak, t_peak = np.where(cube_mask_peaks)
        # x_eclipse, y_eclipse, t_eclipse = np.where(cube_mask_eclipses)
        # unique_xy = [*(get_unique_xy(x_peak, y_peak)), *(get_unique_xy(x_eclipse, y_eclipse))]
        #for x, y in unique_xy:
        #    plot_lc_pixel(cube_mu, cube_n, time_interval, x, y)

        # Plot Image
        plot_detection_image(df_regions, image_eclipse, image_n, image_peak, savepath=savedir / 'detection_img.png')

        # Save Results
        save_df(df=dl.df_bti, savepath=savedir / 'bti.csv')
        save_df(df=df_lcs, savepath=savedir / 'lcs.csv')
        save_df(df=df_regions, savepath=savedir / 'regions.csv')

        save_info(dictionary=observation.info, savepath=savedir / 'obs_info.csv')
        save_info(dictionary=event_list.info, savepath=savedir / 'evt_info.csv')
        save_info(dictionary=dl.info, savepath=savedir / 'dl_info.csv')
        save_info(dictionary=dl.data_cube.info, savepath=savedir / 'data_cube_info.csv')

        plt.close('all')






def main():
    from exod.pre_processing.download_observations import read_observation_ids
    from exod.utils.path import data
    # plot_B_peak()
    # plot_B_eclipse()
    precompute_bayes_limits(threshold_sigma=3)
    precompute_bayes_limits(threshold_sigma=5)
    precompute_bayes_1000()
    load_precomputed_bayes1000()

    obsids = read_observation_ids(data / 'observations.txt')
    # obsids = read_observation_ids(data / 'obs_ccd_check.txt')
    shuffle(obsids)

    # obsids=['0792180301']
    # obsids=['0112570701']
    # obsids=['0810811801']#'0764420101',
    # obsids=['0911990501']
    for obsid in obsids:
        # obsid='0765080801' #'0886121001' '0872390901',
        run_pipeline(obsid)

if __name__=="__main__":
    main()

    
