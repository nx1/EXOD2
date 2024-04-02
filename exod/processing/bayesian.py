import exod.utils.path as path
from exod.utils.plotting import cmap_image
from exod.utils.logger import logger
from exod.pre_processing.data_loader import DataLoader
from exod.xmm.event_list import EventList
from exod.xmm.observation import Observation
from exod.processing.template_based_background_inference import compute_expected_cube_using_templates
from exod.processing.coordinates import get_regions_sky_position, calc_df_regions
from exod.processing.detector import get_region_lcs, plot_region_lightcurve

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm
import pandas as pd
from scipy.stats import poisson
from scipy.interpolate import interp1d
from scipy.special import gammainc, gammaincc
from tqdm import tqdm
import cmasher as cmr


def bayes_factor_peak_nonlog(n, mu):
    """Computes the Bayes factors of the presence of a peak, given the expected (mu) and observed (n) counts."""
    return gammaincc(n + 1, mu) - poisson.pmf(n, mu)


def bayes_factor_peak(n, mu):
    """Computes the Bayes factors of the presence of a peak, given the expected (mu) and observed (n) counts."""
    return np.log10(gammaincc(n + 1, mu)) - np.log10(poisson.pmf(n, mu))


def bayes_factor_eclipse(n, mu):
    """Computes the Bayes factors of the presence of a eclipse, given the expected (mu) and observed (n) counts"""
    return np.log10(gammainc(n + 1, mu)) - np.log10(poisson.pmf(n, mu))


def N_peaks_large_mu(mu, sigma):
    """Calculate the observed (n) value required for a peak at a specific expected (mu) and significance (sigma) in the guassian regime."""
    return np.ceil((2*mu+sigma**2+np.sqrt(8*mu*(sigma**2)+sigma**4))/2)


def N_eclipses_large_mu(mu, sigma):
    """Calculate the observed (n) value required for a eclipse at a specific expected (mu) and significance (sigma) in the guassian regime."""
    return np.floor((2*mu+sigma**2-np.sqrt(8*mu*(sigma**2)+sigma**4))/2)


def get_bayes_thresholds(threshold_sigma):
    """
    The thresholds for B_peak and B_eclipse are pre-calculated here for 3 and 5 sigma.
    B > B_threshold_peak for a peak detection
    B < B_eclipse_threshold for an eclipse detection
    """
    B_peak_threshold = bayes_factor_peak(N_peaks_large_mu(1000, threshold_sigma), 1000)
    B_eclipse_threshold = bayes_factor_eclipse(N_eclipses_large_mu(1000, threshold_sigma), 1000)
    return B_peak_threshold, B_eclipse_threshold


def precompute_bayes_limits(threshold_sigma):
    """
    Compute the minimum and maximum number of observed
    counts (n) required for a eclipse or peak for a given
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


    Each data cell in the observed and expected cube
    is then compared to the values pre-calculated here
    to determine if is it a peak or an eclipse.
    
    An example for 1 frame is given here.

     (n_cube)    (mu_cube)       is_3_sig_peak? 
      0 2 1    0.03 1.98 0.87       F F F
      0 5 0    1.01 1.10 1.00       F T F 
      3 2 1    2.30 2.14 0.98       F F F

    The evaluation of each data cell of the cube against
    a threshold is made faster by precomputing the counts here.
    """
    B_peak_threshold, B_eclipse_threshold = get_bayes_thresholds(threshold_sigma=threshold_sigma)

    range_mu       = np.geomspace(start=1e-7, stop=1e3, num=50000) #
    range_mu_large = np.geomspace(start=1e3, stop=1e6, num=1000)   # Above 1000 

    tab_npeak, tab_neclipse = [], []
    for mu in tqdm(range_mu):
        # Find the smallest N for a peak
        range_n_peak = np.arange(max(10 * int(mu), 100))
        B_factors = bayes_factor_peak(n=range_n_peak, mu=mu)
        n_peak_max = range_n_peak[B_factors > B_peak_threshold][0]
        tab_npeak.append(n_peak_max)

        # Get the largest N for an eclipse
        range_n_eclipse = np.arange(2 * int(mu) + 1)
        B_factors = bayes_factor_eclipse(n=range_n_eclipse, mu=mu)
        n_eclipse_min = range_n_eclipse[B_factors < B_eclipse_threshold][0]
        tab_neclipse.append(n_eclipse_min)
        
    tab_npeak    += list(N_peaks_large_mu(range_mu_large, threshold_sigma))
    tab_neclipse += list(N_eclipses_large_mu(range_mu_large, threshold_sigma))
    
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
    range_N        = np.arange(10000)
    tab_B_peaks    = bayes_factor_peak(n=range_N, mu=1000)
    tab_B_eclipses = bayes_factor_eclipse(n=range_N, mu=1000)
    data = np.array([range_N, tab_B_peaks, tab_B_eclipses])
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
    minimum_for_peak = interp1d(range_mu, data[1])
    maximum_for_eclipse = interp1d(range_mu, data[2])
    return range_mu, minimum_for_peak, maximum_for_eclipse


def get_cube_masks_peak_and_eclipse(cube, expected, threshold_sigma):
    """Returns two cubes with booleans where the rate correspond to a peak or an eclipse."""
    range_mu, minimum_for_peak, maximum_for_eclipse = load_precomputed_bayes_limits(threshold_sigma=threshold_sigma)
    cube_mask_peaks   = cube > minimum_for_peak(np.where(expected > range_mu[0], expected, np.nan))
    cube_mask_eclipse = cube < maximum_for_eclipse(np.where(expected > range_mu[0], expected, np.nan))
    return cube_mask_peaks, cube_mask_eclipse


def plot_B_peak():
    """
    Plot the peak Bayes factor for different observed (n) counts as a function of expectation (mu).
    Also plot the 3 and 5 sigma threshold values.
    """
    n_lines_to_plot = 20 # 1 line is drawn for each value of n from 0 to n-1
    colors = plt.cm.winter(np.linspace(0, 1, n_lines_to_plot)) 
    
    B_peak_3sig, B_eclipse_3sig = get_bayes_thresholds(3)
    B_peak_5sig, B_eclipse_5sig = get_bayes_thresholds(5)
    
    mu_lo, mu_hi = 1e-3, 50
    mus = np.geomspace(mu_lo, mu_hi, 1000)
    
    plt.figure(figsize=(5,5))
    for n in range(n_lines_to_plot):
        label=None
        if (n == 0) or (n==n_lines_to_plot-1): # Label first and last line
            label = f'n={n}'
        plt.plot(mus, bayes_factor_peak(n=n, mu=mus), color=colors[n], label=label)
    
    plt.axhline(B_peak_3sig, color='red', label=rf'3 $\sigma$ (B={B_peak_3sig:.2f})')
    plt.axhline(B_peak_5sig, color='black', label=rf'5 $\sigma$ (B={B_peak_5sig:.2f})')
    plt.title(f'Peak Bayes factor for n=0-{n_lines_to_plot}')
    plt.xlabel(r'Expected Value $\mu$')
    plt.ylabel(r'$log_{10}$($B_{peak}$)')
    plt.xscale('log')
    plt.tight_layout()
    plt.xlim(mu_lo, mu_hi)
    plt.legend()
    plt.show()

def plot_B_eclipse():
    """
    Plot the eclipse Bayes factor for different observed (n) counts as a function of expecation (mu).
    Also plot the 3 and 5 sigma threshold values.
    """
    n_lines_to_plot = 20 # 1 line is drawn for each value of n from 0 to n-1
    colors = plt.cm.winter(np.linspace(0, 1, n_lines_to_plot)) 
    
    B_peak_3sig, B_eclipse_3sig = get_bayes_thresholds(3)
    B_peak_5sig, B_eclipse_5sig = get_bayes_thresholds(5)
    
    mu_lo, mu_hi = 1e-3, 50
    mus = np.geomspace(mu_lo, mu_hi, 1000)
    
    plt.figure(figsize=(5,5))
    for n in range(n_lines_to_plot):
        label=None
        if (n == 0) or (n==n_lines_to_plot-1): # Label first and last line
            label = f'n={n}'
        plt.plot(mus, bayes_factor_eclipse(n=n, mu=mus), color=colors[n], label=label)
    
    plt.axhline(B_eclipse_3sig, color='red', label=rf'3 $\sigma$ (B={B_peak_3sig:.2f})')
    plt.axhline(B_eclipse_5sig, color='black', label=rf'5 $\sigma$ (B={B_peak_5sig:.2f})')
    plt.title(f'Eclipse Bayes factor for n=0-{n_lines_to_plot}')
    plt.xlabel(r'Expected Value $\mu$')
    plt.ylabel(r'$log_{10}$($B_{eclipse}$)')
    # plt.xscale('log')
    plt.tight_layout()
    plt.xlim(mu_lo, mu_hi)
    plt.legend()
    plt.show()


def get_unique_xy(x, y):
    """Get the unique pairs of two lists."""
    unique_xy = set()  # Set to store unique pairs
    for x, y in zip(x, y):
        unique_xy.add((x, y))
    return unique_xy


def run_pipeline(obsid,
                 size_arcsec=20,
                 time_interval=50,
                 gti_only=False,
                 gti_threshold=1.5,
                 min_energy=0.5,
                 max_energy=10.0,
                 threshold_sigma=3):
    # obsid='0765080801' #'0886121001' '0872390901'

    # Load data
    observation = Observation(obsid)
    observation.get_files()
    # try:
    observation.get_events_overlapping_subsets()
    for ind_exp, subset_overlapping_exposures in enumerate(observation.events_overlapping_subsets):
        event_list = EventList.from_event_lists(subset_overlapping_exposures)
        event_list.info
        dl = DataLoader(event_list=event_list, time_interval=time_interval, size_arcsec=size_arcsec,
                        gti_only=gti_only, min_energy=min_energy, max_energy=max_energy,
                        gti_threshold=gti_threshold, remove_partial_ccd_frames=True)
        dl.run()

        img = observation.images[0]
        img.read(wcs_only=True)

        cube_n = dl.data_cube
        cube_mu = compute_expected_cube_using_templates(cube_n, wcs=img.wcs)
        cube_mask_peaks, cube_mask_eclipses = get_cube_masks_peak_and_eclipse(cube_n.data, cube_mu, threshold_sigma=threshold_sigma)

        image_peak = np.nansum(cube_mask_peaks, axis=2)       # Each Pixel is the number of peaks in cube
        image_eclipse = np.nansum(cube_mask_eclipses, axis=2) # Each Pixel is the number of eclipses in cube
        image_n = np.nansum(cube_n.data, axis=2)

        df_reg1 = calc_df_regions(image=image_n, image_mask=image_peak>0)
        df_reg1['Alert'] = 'Peak'
        n_unique_peak_reg = len(df_reg1)

        df_reg2 = calc_df_regions(image=image_n, image_mask=image_eclipse>0)
        df_reg2['Alert'] = 'Eclipse'
        n_unique_eclipse_reg = len(df_reg2)

        df_regions = pd.concat([df_reg1, df_reg2]).reset_index()
        print(df_regions)


        df_sky = get_regions_sky_position(data_cube=cube_n, df_regions=df_regions, wcs=img.wcs)

        df_regions = pd.concat([df_regions, df_sky], axis=1)
        print(df_regions)

        df_lcs = get_region_lcs(data_cube=cube_n, df_regions=df_regions)
        print(df_lcs)

        for i in df_regions['index']:
            plot_region_lightcurve(df_lcs=df_lcs, i=i, savepath=None)


        x_peak, y_peak, t_peak = np.where(cube_mask_peaks)
        x_eclipse, y_eclipse, t_eclipse = np.where(cube_mask_eclipses)

        unique_xy_peak    = get_unique_xy(x_peak, y_peak)
        unique_xy_eclipse = get_unique_xy(x_eclipse, y_eclipse)
        unique_xy = [*unique_xy_peak, *unique_xy_eclipse]

        for x, y in unique_xy:
            cube_mu_xy   = cube_mu[x, y]
            cube_data_xy = cube_n.data[x, y]

            # Plot lightcurves
            colors = cmr.take_cmap_colors('cmr.ocean', N=2, cmap_range=(0, 0.5))
            lw = 1.0

            frame_axis = np.arange(cube_mu.shape[2]) # Frame Number
            time_axis  = frame_axis * time_interval  # Zero'ed Time
            time_axis2 = cube_n.bin_t[:-1]           # Observation Time

            mu_3sig, n_peak_3sig, n_eclipse_3sig = load_precomputed_bayes_limits(threshold_sigma=3)
            mu_5sig, n_peak_5sig, n_eclipse_5sig = load_precomputed_bayes_limits(threshold_sigma=5)

            fig, ax = plt.subplots(figsize=(15,5))
            ax.fill_between(time_axis, n_eclipse_5sig(cube_mu_xy), n_peak_5sig(cube_mu_xy), alpha=0.3, facecolor='blue', label=r'5 $\sigma$')
            ax.fill_between(time_axis, n_eclipse_3sig(cube_mu_xy), n_peak_3sig(cube_mu_xy), alpha=0.5, facecolor='blue', label=r'3 $\sigma$')

            ax.step(time_axis, cube_data_xy, color='black', where='mid', lw=lw, label=r'Observed ($n$)')
            ax.step(time_axis, cube_mu_xy, color='red', where='mid', lw=lw, label=r'Expected ($\mu$)')

            ax2 = ax.twiny()
            ax2.plot(frame_axis, cube_data_xy, color='none')
            ax2.set_xlabel("Frame #")

            ax.legend()
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Counts")
            ax.set_ylim(0)
            ax.set_xlim(np.min(time_axis), np.max(time_axis))
            plt.suptitle(f'Lightcurve for x={x} y={y}')


        fig, ax = plt.subplots(figsize=(9,9))
        # Map the peaks=1 and eclipses=2
        image_combined = np.zeros_like(image_peak)
        image_combined[image_peak > 0] = 1
        image_combined[image_eclipse > 0] = 2

        c_0       = 'none'
        c_peak    = 'cyan'
        c_eclipse = 'lime'
        cmap = ListedColormap(colors=[c_0, c_peak, c_eclipse])

        norm = BoundaryNorm(boundaries=[0,1,2,3], ncolors=3)
        im1 = ax.imshow(image_n, cmap=cmap_image(), norm=LogNorm())
        ax.imshow(image_combined, cmap=cmap, norm=norm)

        plt.colorbar(im1, ax=ax, shrink=0.75)

        legend_labels = [
            plt.Line2D([0], [0], label='Peak', markerfacecolor=c_peak, markeredgecolor=c_peak, marker='s', markersize=10, ls='none'),
            plt.Line2D([0], [0], label='Eclipse', markerfacecolor=c_eclipse, markeredgecolor=c_eclipse, marker='s', markersize=10, ls='none')
        ]
        ax.legend(handles=legend_labels)
        ax.set_title(f'N_peaks={len(x_peak)} unique_xy={len(unique_xy_peak)} unique_reg={n_unique_peak_reg}\n'
                     f'N_eclipses={len(x_eclipse)} unique_xy={len(unique_xy_eclipse)} unique_reg={n_unique_eclipse_reg}')
        plt.tight_layout()
        plt.show()


def main():
    from exod.pre_processing.download_observations import read_observation_ids
    from exod.utils.path import data
    # plot_B_peak()
    # plot_B_eclipse()
    # precompute_bayes_limits(threshold_sigma=3)
    # precompute_bayes_limits(threshold_sigma=5)

    obsids = read_observation_ids(data / 'observations.txt')
    # obsids = read_observation_ids(data / 'obs_ccd_check.txt')
    # shuffle(obsids)
    # obsids=['0792180301']
    # obsids=['0112570701']
    # obsids=['0810811801']#'0764420101',
    # obsids=['0911990501']
    for obsid in tqdm(obsids):
        run_pipeline(obsid)

if __name__=="__main__":
    main()

    
