from exod.pre_processing.data_loader import DataLoader
from exod.processing.background_inference import calc_cube_mu
from exod.utils.path import data_processed
from exod.processing.bayesian_computations import B_peak_log, sigma_equivalent

import numpy as np
import os
import matplotlib.pyplot as plt
import cmasher as cmr
from scipy.stats import poisson
from scipy.special import gammainc, gammaincc, gammaincinv

from exod.xmm.event_list import EventList
from exod.xmm.observation import Observation


def plot_lightcurve_alerts_with_background(cube, cube_background, cube_background_withsource, tab_boundingboxes):
    """
    This function creates the multi-panel lightcurve of the transient object. It will retrieve the background,
    and use it to compare the (source+background) to the background, and compute the likelihood in each frame
    :param cube: full data cube
    :param cube_background: data cube of de-sourced background estimate
    :param cube_background_withsource: data cube of de-sourced background estimate + constant contribution from the
    sources (i.e. we assume they are constant, take their stacked flux and distribute it over all frames)
    :param tab_boundingboxes: bounding boxes of variable objects, as obtained from variability.py
    :return: nothing, but saves the lightcurve of each (source+background) and its background, lightcurve of
    background-subtracted source, and lightcurve of likelihood of creating the signal with the background
    """
    color = cmr.take_cmap_colors('cmr.ocean', N=1, cmap_range=(0.3, 0.3))[0]
    for ind, source in enumerate(tab_boundingboxes):
        legend_plots_ax1 = []
        legend_labels_ax1 = []
        legend_plots_ax2 = []
        legend_labels_ax2 = []
        legend_plots_ax3 = []
        legend_labels_ax3 = []

        # First step: we generate the lightcurves and a number of Poisson realisations of it
        lc = np.sum(cube[source[0]:source[2], source[1]:source[3]], axis=(0, 1))
        lc_generated = np.random.poisson(lc, (5000, len(lc)))
        lc_percentiles = np.nanpercentile(lc_generated, (16, 84), axis=0)
        lc_back = np.sum(cube_background[source[0]:source[2], source[1]:source[3]], axis=(0, 1))
        lc_back_generated = np.random.poisson(lc_back, (5000, len(lc)))
        lc_back_percentiles = np.nanpercentile(lc_back_generated, (16, 84), axis=0)
        lc_back_withsource = np.sum(cube_background_withsource[source[0]:source[2], source[1]:source[3]], axis=(0, 1))
        lc_back_generated_withsource = np.random.poisson(lc_back_withsource, (5000, len(lc)))

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

        # First panel: Source + background lightcurve
        p1 = ax1.step(range(len(lc)), lc, c=color, where="mid")
        p2 = ax1.fill(np.NaN, np.NaN, facecolor=color, alpha=0.4)
        ax1.fill_between(range(len(lc)), lc_percentiles[0], lc_percentiles[1], alpha=0.4, facecolor=color, step="mid")
        legend_plots_ax1.append((p1[0], p2[0]))
        legend_labels_ax1.append("Source+background")
        p1 = ax1.step(range(len(lc_back)), lc_back, c="gray", where="mid")
        p2 = ax1.fill(np.NaN, np.NaN, facecolor="gray", alpha=0.4)
        ax1.fill_between(range(len(lc_back)), lc_back_percentiles[0], lc_back_percentiles[1],
                         alpha=0.4, facecolor="gray", step="mid")
        legend_plots_ax1.append((p1[0], p2[0]))
        legend_labels_ax1.append("Background")
        ax1.legend(legend_plots_ax1, legend_labels_ax1)

        # Second panel: Background subtracted lightcurve
        p1 = ax2.step(range(len(lc)), lc - lc_back, c=color, where='mid')
        p2 = ax2.fill(np.NaN, np.NaN, facecolor=color, alpha=0.4)
        lc_diff_percentiles = np.nanpercentile(lc_generated - lc_back_generated, (16, 84), axis=0)
        ax2.fill_between(range(len(lc)), lc_diff_percentiles[0], lc_diff_percentiles[1],
                         alpha=0.4, facecolor=color, step='mid')
        legend_plots_ax2.append((p1[0], p2[0]))
        legend_labels_ax2.append("Source-Background")
        ax2.legend(legend_plots_ax2, legend_labels_ax2)

        # Third panel: Poisson likelihood
        p1 = ax3.step(range(len(lc)), -poisson.logpmf(lc, lc_back_withsource), c=color, where='mid')
        p2 = ax3.fill(np.NaN, np.NaN, facecolor=color, alpha=0.4)
        likelihood_percentiles = np.nanpercentile(-poisson.logpmf(lc_generated, lc_back_generated_withsource), (16, 84),
                                                  axis=0)
        ax3.fill_between(range(len(lc)), likelihood_percentiles[0], likelihood_percentiles[1],
                         alpha=0.4, facecolor=color, step='mid')
        legend_plots_ax3.append((p1[0], p2[0]))
        legend_labels_ax3.append("Poisson likelihood")
        ax3.legend(legend_plots_ax2, legend_labels_ax2)

        plt.savefig(os.path.join(data_processed, '0831790701', f'lightcurve_transient_{ind}.png'))


def clean_up_peaks(data_cube, peaks):
    """Removes peaks coming from flaring CCD edges"""
    new_peaks = np.copy(peaks)
    peak_x, peak_y, peak_t = np.where(peaks == True)
    cube = data_cube.data
    for (x, y, t) in zip(peak_x, peak_y, peak_t):
        non_source_peakframe = np.nansum(cube[:, :, t]) - np.nansum(cube[x - 1:x + 2, y - 1:y + 2, t])
        non_source_around_peakframe = (np.nansum(cube[:, :, t - 5:t + 6]) - np.nansum(
            cube[x - 1:x + 2, y - 1:y + 2, t - 5:t + 6]) - non_source_peakframe) / 10
        source_peakframe = np.nansum(cube[x - 1:x + 2, y - 1:y + 2, t])
        source_around_peakframe = (np.nansum(cube[x - 1:x + 2, y - 1:y + 2, t - 5:t + 6]) - source_peakframe) / 10
        # We reject peaks for which the rest of the frame has a peak as well, and the source peak
        # is at most 25% larger in relative amplitude than the background peak.
        background_peak_amplitude = ((non_source_peakframe - non_source_around_peakframe) / non_source_around_peakframe)
        source_peak_amplitude = ((source_peakframe - source_around_peakframe) / source_around_peakframe)
        if (background_peak_amplitude > 0) and ((source_peak_amplitude - background_peak_amplitude) < 0.25):
            new_peaks[x, y, t] = False
            # data_cube.data[:,:,t]=np.full(data_cube.shape[:2],np.nan)
    return new_peaks


def clean_up_eclipses(data_cube, eclipses):
    """Removes eclipses in bright sources coming from merging of partial exposures. This is done by checking if the flux
    change is the same in the source and the rest of the frame"""
    new_eclipses = np.copy(eclipses)
    eclipse_x, eclipse_y, eclipse_t = np.where(eclipses == True)
    cube = data_cube.data
    for (x, y, t) in zip(eclipse_x, eclipse_y, eclipse_t):
        non_source_eclipseframe = np.nansum(cube[:, :, t]) - np.nansum(cube[x - 1:x + 2, y - 1:y + 2, t])
        non_source_around_eclipseframe = (np.nansum(cube[:, :, t - 5:t + 6]) - np.nansum(
            cube[x - 1:x + 2, y - 1:y + 2, t - 5:t + 6]) - non_source_eclipseframe) / 10
        source_eclipseframe = np.nansum(cube[x - 1:x + 2, y - 1:y + 2, t])
        source_around_eclipseframe = (np.nansum(cube[x - 1:x + 2, y - 1:y + 2, t - 5:t + 6]) - source_eclipseframe) / 10
        # We reject eclipses for which the rest of the frame has had an eclipse as well, and this background "eclipse"
        # is at most 25% larger in relative amplitude than the source eclipse.
        if ((non_source_eclipseframe < non_source_around_eclipseframe) and
                ((((non_source_around_eclipseframe - non_source_eclipseframe) / non_source_around_eclipseframe) -
                  ((source_around_eclipseframe - source_eclipseframe) / source_around_eclipseframe)) < 0.25)):
            new_eclipses[x, y, t] = False
            # data_cube.data[:,:,t]=np.full(data_cube.shape[:2],np.nan)
    return new_eclipses


def count_peaks(peaks_or_eclipses):
    """Counts the individual number of times the lightcurve went above the threshold for variability"""
    nbr_of_variability_events = np.nansum(np.abs(np.diff(peaks_or_eclipses, axis=2)), axis=2) / 2
    return nbr_of_variability_events


def peak_count_estimate(fraction, N, mu):
    """Estimate the upper limit on the count of the peak, given an data_cube and observed counts,
     and a confidence fraction"""
    return gammaincinv(N + 1, fraction * gammaincc(N + 1, mu) + gammainc(N + 1, mu)) - mu


def eclipse_count_estimate(fraction, N, mu):
    """Estimate the upper limit on the count of the eclipse, given an data_cube and observed counts,
     and a confidence fraction"""
    return mu - gammaincinv(N + 1, gammainc(N + 1, mu) - fraction * gammainc(N + 1, mu))


def convert_count_to_flux(count, position, data_cube):
    # TODO: find the vignetting functions depending on energy / submode / filters. Maybe build the exposure map for
    # each observation  (https://xmm-tools.cosmos.esa.int/external/sas/current/doc/eexpmap.pdf)
    # Find the EEF (Encircled Energy Frac.), maybe with ARFGEN if we manage to convert data_cube regions to XY regions
    # Think about which Energy Conversion Factor (ECF) to use. Maybe depends on spectral search (hard or soft).
    """Used to convert count rates to fluxes, using vignetting / EEF / ECF"""

    """
    ds9 --> square region
    from this create an ARF for the region. http://xmm-tools.cosmos.esa.int/external/sas/current/doc/arfgen/
    This will allow you to convert a count to a count rate.
    from a count rate, you can get the flux by using an ECF.
    
    A spectral should be assumed that will likely depend on the band you're looking at:
    0.2 - 2.0 --> bbody
    2.0 - 12.0 --> something else
    this will change the flux by a factor of 2-3x
    
    this will give us an L for a given distance. :)
    """

    return count / data_cube.time_interval


def calc_sigma_cube(cube_n, cube_mu):
    """
    Calculate the sigma equivalent over a data cube.
    cube_n  : observed data cube
    cube_mu : Expected data cube
    """
    result_cube = np.zeros_like(cube_n)
    for i in range(cube_n.shape[0]):
        print(f'Calculating sigma cube... {i}/{cube_n.shape[0]}')
        for j in range(cube_n.shape[1]):
            for k in range(cube_n.shape[2]):
                n = cube_n[i, j, k]
                mu = cube_mu[i, j, k]
                result_cube[i, j, k] = sigma_equivalent(n, mu)
    return result_cube


if __name__ == "__main__":
    observation = Observation('0654800101')
    observation.get_files()
    observation.get_events_overlapping_subsets()

    for i_subset, subset_overlapping_exposures in enumerate(observation.events_overlapping_subsets):
        event_list = EventList.from_event_lists(subset_overlapping_exposures)
        dl = DataLoader(event_list=event_list, time_interval=500, size_arcsec=20, gti_only=False, min_energy=0.2,
                        max_energy=10.0, remove_partial_ccd_frames=True)
        dl.run()

        img = observation.images[0]
        img.read(wcs_only=True)
        
        cube_n  = dl.data_cube
        cube_mu = calc_cube_mu(cube_n, wcs=img.wcs)

        cube_sigma = calc_sigma_cube(cube_n=cube_n.data, cube_mu=cube_mu)
        s_vec = np.vectorize(sigma_equivalent)
        cube_sigma1 = s_vec(n=cube_n.data, mu=cube_mu)

        stacked_array = np.dstack((cube_n.data, cube_mu)).reshape(-1,2)

        def sigma(n_mu):
            n  = n_mu[0]
            mu = n_mu[1]
            return sigma_equivalent(n=n, mu=mu)

        res = np.apply_along_axis(sigma, stacked_array.reshape(-1,2))
        res2 = res.reshape(shape=cube_n.shape)

