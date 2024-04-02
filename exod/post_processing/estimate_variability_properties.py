from exod.utils.path import data_processed
from exod.pre_processing.download_observations import read_observation_ids
from exod.utils.path import data
from exod.pre_processing.data_loader import DataLoader
from exod.xmm.event_list import EventList
from exod.xmm.observation import Observation
from exod.processing.template_based_background_inference import compute_expected_cube_using_templates
from exod.processing.bayesian import get_cube_masks_peak_and_eclipse, load_precomputed_bayes_limits

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cmasher as cmr
from scipy.stats import poisson
from scipy.special import gammainc, gammaincc, gammaincinv
from tqdm import tqdm
from scipy.optimize import root_scalar


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
    color=cmr.take_cmap_colors('cmr.ocean',N=1,cmap_range=(0.3,0.3))[0]
    for ind,source in enumerate(tab_boundingboxes):
        legend_plots_ax1 = []
        legend_labels_ax1 = []
        legend_plots_ax2 = []
        legend_labels_ax2 = []
        legend_plots_ax3 = []
        legend_labels_ax3 = []

        #First step: we generate the lightcurves and a number of Poisson realisations of it
        lc = np.sum(cube[source[0]:source[2], source[1]:source[3]], axis=(0,1))
        lc_generated = np.random.poisson(lc,(5000,len(lc)))
        lc_percentiles = np.nanpercentile(lc_generated, (16,84),axis=0)
        lc_back = np.sum(cube_background[source[0]:source[2], source[1]:source[3]], axis=(0,1))
        lc_back_generated = np.random.poisson(lc_back,(5000,len(lc)))
        lc_back_percentiles = np.nanpercentile(lc_back_generated, (16,84),axis=0)
        lc_back_withsource = np.sum(cube_background_withsource[source[0]:source[2], source[1]:source[3]], axis=(0,1))
        lc_back_generated_withsource = np.random.poisson(lc_back_withsource,(5000,len(lc)))

        fig, (ax1,ax2, ax3) = plt.subplots(3,1)

        #First panel: Source + background lightcurve
        p1=ax1.step(range(len(lc)),lc, c=color, where="mid")
        p2=ax1.fill(np.NaN, np.NaN, facecolor=color, alpha=0.4)
        ax1.fill_between(range(len(lc)),lc_percentiles[0],lc_percentiles[1],alpha=0.4, facecolor=color, step="mid")
        legend_plots_ax1.append((p1[0],p2[0]))
        legend_labels_ax1.append("Source+background")
        p1=ax1.step(range(len(lc_back)),lc_back, c="gray", where="mid")
        p2=ax1.fill(np.NaN, np.NaN, facecolor="gray", alpha=0.4)
        ax1.fill_between(range(len(lc_back)),lc_back_percentiles[0],lc_back_percentiles[1],
                         alpha=0.4, facecolor="gray", step="mid")
        legend_plots_ax1.append((p1[0],p2[0]))
        legend_labels_ax1.append("Background")
        ax1.legend(legend_plots_ax1,legend_labels_ax1)

        #Second panel: Background subtracted lightcurve
        p1=ax2.step(range(len(lc)),lc-lc_back, c=color, where='mid')
        p2=ax2.fill(np.NaN, np.NaN, facecolor=color, alpha=0.4)
        lc_diff_percentiles = np.nanpercentile(lc_generated-lc_back_generated, (16,84),axis=0)
        ax2.fill_between(range(len(lc)),lc_diff_percentiles[0],lc_diff_percentiles[1],
                         alpha=0.4, facecolor=color,step='mid')
        legend_plots_ax2.append((p1[0],p2[0]))
        legend_labels_ax2.append("Source-Background")
        ax2.legend(legend_plots_ax2,legend_labels_ax2)

        #Third panel: Poisson likelihood
        p1=ax3.step(range(len(lc)),-poisson.logpmf(lc, lc_back_withsource), c=color, where='mid')
        p2=ax3.fill(np.NaN, np.NaN, facecolor=color, alpha=0.4)
        likelihood_percentiles = np.nanpercentile(-poisson.logpmf(lc_generated,lc_back_generated_withsource), (16,84), axis=0)
        ax3.fill_between(range(len(lc)),likelihood_percentiles[0],likelihood_percentiles[1],
                         alpha=0.4, facecolor=color,step='mid')
        legend_plots_ax3.append((p1[0],p2[0]))
        legend_labels_ax3.append("Poisson likelihood")
        ax3.legend(legend_plots_ax2,legend_labels_ax2)

        plt.savefig(os.path.join(data_processed,'0831790701',f'lightcurve_transient_{ind}.png'))


def clean_up_peaks(data_cube, peaks):
    """Removes peaks coming from flaring CCD edges"""
    new_peaks=np.copy(peaks)
    peak_x, peak_y, peak_t = np.where(peaks==True)
    cube=data_cube.data
    for (x,y,t) in zip(peak_x, peak_y, peak_t):
        non_source_peakframe = np.nansum(cube[:, :, t]) - np.nansum(cube[x - 1:x + 2, y - 1:y + 2, t])
        non_source_around_peakframe = (np.nansum(cube[:, :, t - 5:t + 6]) - np.nansum(cube[x - 1:x + 2, y - 1:y + 2, t - 5:t + 6]) - non_source_peakframe) / 10
        source_peakframe = np.nansum(cube[x - 1:x + 2, y - 1:y + 2, t])
        source_around_peakframe = (np.nansum(cube[x - 1:x + 2, y - 1:y + 2, t - 5:t + 6]) - source_peakframe) / 10
        #We reject peaks for which the rest of the frame has a peak as well, and the source peak
        #is at most 25% larger in relative amplitude than the background peak.
        background_peak_amplitude = ((non_source_peakframe-non_source_around_peakframe)/non_source_around_peakframe)
        source_peak_amplitude = ((source_peakframe-source_around_peakframe)/source_around_peakframe)
        if (background_peak_amplitude>0) and ((source_peak_amplitude-background_peak_amplitude)<0.25):
            new_peaks[x,y,t]=False
            # data_cube.data[:,:,t]=np.full(data_cube.shape[:2],np.nan)
    return new_peaks

def clean_up_eclipses(data_cube, eclipses):
    """Removes eclipses in bright sources coming from merging of partial exposures. This is done by checking if the flux
    change is the same in the source and the rest of the frame"""
    new_eclipses=np.copy(eclipses)
    eclipse_x, eclipse_y, eclipse_t = np.where(eclipses==True)
    cube=data_cube.data
    for (x,y,t) in zip(eclipse_x, eclipse_y, eclipse_t):
        non_source_eclipseframe = np.nansum(cube[:,:,t]) - np.nansum(cube[x-1:x+2,y-1:y+2,t])
        non_source_around_eclipseframe = (np.nansum(cube[:,:,t-5:t+6]) - np.nansum(cube[x-1:x+2,y-1:y+2,t-5:t+6]) - non_source_eclipseframe)/10
        source_eclipseframe = np.nansum(cube[x - 1:x + 2, y - 1:y + 2, t])
        source_around_eclipseframe = (np.nansum(cube[x - 1:x + 2, y - 1:y + 2, t-5:t+6]) - source_eclipseframe)/10
        #We reject eclipses for which the rest of the frame has had an eclipse as well, and this background "eclipse"
        #is at most 25% larger in relative amplitude than the source eclipse.
        if ((non_source_eclipseframe<non_source_around_eclipseframe) and
                ((((non_source_around_eclipseframe-non_source_eclipseframe)/non_source_around_eclipseframe)-
                 ((source_around_eclipseframe-source_eclipseframe)/source_around_eclipseframe))<0.25)):
            new_eclipses[x,y,t]=False
            # data_cube.data[:,:,t]=np.full(data_cube.shape[:2],np.nan)
    return new_eclipses

def find_sigma(n, mu):
    """Uses the interpolated values of B(n,mu) to convert (n,mu) to a sigma level.
    The idea is that, whether it's a peak or eclipse, you want to find the sigma level of B(n,mu). To do this,
     we go to large counts along the iso-B lines: in this case, N*(mu=1000,sigma) is the observed counts that
     correspond to a sigma-level departure from mu=1000. This is the solution of the 2nd degree polynomial (see doc).
    We want to find sigma such that B(N*(mu=1000,sigma),mu=1000) is the same as the observed B. This is done by inverting
    the function (by finding the root of B(n,mu)-B(N*(mu=1000,sigma),mu=1000)"""
    if n>mu: #Means it's a peak
        b = bayes_factor_peak(n,mu)
        function_to_invert = lambda sigma : b - bayes_factor_peak(N_peaks_large_mu(1000, sigma), 1000)
        #We need to provide a range for the inversion method. To exclude edge cases, we check if it's above 10 sigma
        #or below 1 sigma, which we both exclude. We can then look in the region in between
        if function_to_invert(10)>0:
            return 10
        elif function_to_invert(1)<0:
            return 0
        else:
            return root_scalar(function_to_invert, bracket=(1,10)).root

    else: #Means it's an eclipse
        b = bayes_factor_eclipse(n,mu)
        function_to_invert = lambda sigma : b - bayes_factor_eclipse(N_eclipses_large_mu(1000, sigma), 1000)
        if function_to_invert(10) > 0:
            return 10
        elif function_to_invert(1)<0:
            return 0
        else:
            return root_scalar(function_to_invert, bracket=(1, 10)).root

def count_peaks(peaks_or_eclipses):
    """Counts the individual number of times the lightcurve went above the threshold for variability"""
    nbr_of_variability_events = np.nansum(np.abs(np.diff(peaks_or_eclipses, axis=2)),axis=2)/2
    return nbr_of_variability_events

def peak_count_estimate(fraction, N, mu):
    """Estimate the upper limit on the count of the peak, given an expected and observed counts,
     and a confidence fraction"""
    return gammaincinv(N+1, fraction*gammaincc(N+1, mu) + gammainc(N+1, mu)) - mu

def eclipse_count_estimate(fraction, N, mu):
    """Estimate the upper limit on the count of the eclipse, given an expected and observed counts,
     and a confidence fraction"""
    return mu - gammaincinv(N+1, gammainc(N+1, mu) - fraction*gammainc(N+1, mu))

def convert_count_to_flux(count, position, data_cube):
    #TODO: find the vignetting functions depending on energy / submode / filters. Maybe build the exposure map for
    # each obsid  (https://xmm-tools.cosmos.esa.int/external/sas/current/doc/eexpmap.pdf)
    # Find the EEF (Encircled Energy Frac.), maybe with ARFGEN if we manage to convert datacube regions to XY regions
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

    return count/data_cube.time_interval


if __name__=="__main__":
    #Testing clean up effect
    obsids = read_observation_ids(data / 'observations.txt')
    # obsids=['0202670701']
    # obsids=['0675010401']
    tab_old_peaks=[]
    tab_old_eclipses=[]
    tab_new_peaks=[]
    tab_new_eclipses=[]
    for obsid in tqdm(obsids):
        size_arcsec = 20
        time_interval = 10
        gti_only = False
        gti_threshold = 1.5
        min_energy = 0.2
        max_energy = 2.

        threshold_sigma=5

        # Load data
        observation = Observation(obsid)
        observation.get_files()
        observation.get_events_overlapping_subsets()
        for ind_exp,subset_overlapping_exposures in enumerate(observation.events_overlapping_subsets):
            event_list = EventList.from_event_lists(subset_overlapping_exposures)
            if event_list.exposure>2*time_interval:
                dl = DataLoader(event_list=event_list, time_interval=time_interval, size_arcsec=size_arcsec,
                                gti_only=gti_only, min_energy=min_energy, max_energy=max_energy,
                                gti_threshold=gti_threshold)
                dl.run()
                img = observation.images[0]
                img.read(wcs_only=True)

                estimated_cube = compute_expected_cube_using_templates(data_cube=dl.data_cube, wcs=img.wcs)
                peaks, eclipses = get_cube_masks_peak_and_eclipse(dl.data_cube.data, estimated_cube, threshold_sigma=threshold_sigma)
                tab_old_peaks.append(np.sum(np.nansum(peaks,axis=2)>0))
                tab_old_eclipses.append(np.sum(np.nansum(eclipses,axis=2)>0))
                new_peaks = clean_up_peaks(dl.data_cube,peaks)
                new_eclipses = clean_up_eclipses(dl.data_cube,eclipses)
                tab_new_peaks.append(np.sum(np.nansum(new_peaks,axis=2)>0))
                tab_new_eclipses.append(np.sum(np.nansum(new_eclipses,axis=2)>0))
                # if np.sum(np.nansum(new_eclipses,axis=2)>0)>0:
                if tab_new_peaks[-1]>0:#tab_new_peaks[-1]<tab_old_peaks[-1]:
                    range_mu_3sig, minimum_for_peak_3sig, maximum_for_eclipse_3sig = load_precomputed_bayes_limits(
                        threshold_sigma=3)
                    range_mu_5sig, minimum_for_peak_5sig, maximum_for_eclipse_5sig = load_precomputed_bayes_limits(
                        threshold_sigma=5)
                    fig, axes = plt.subplots(2, 2)
                    colors = cmr.take_cmap_colors('cmr.ocean', N=2, cmap_range=(0, 0.5))
                    plt.suptitle(f'ObsID {obsid} -- Exposure {ind_exp} --  Binning {time_interval}s')
                    axes[0][0].imshow(np.nansum(dl.data_cube.data, axis=2), norm=LogNorm(), interpolation='none')
                    axes[1][0].imshow(np.where(np.nansum(dl.data_cube.data, axis=2) > 0, np.nansum(new_peaks, axis=2),
                                               np.empty(dl.data_cube.shape[:2]) * np.nan),
                                      vmax=1, vmin=0, interpolation='none')
                    m = axes[1][1].imshow(
                        np.where(np.nansum(dl.data_cube.data, axis=2) > 0, np.nansum(new_eclipses, axis=2),
                                 np.empty(dl.data_cube.shape[:2]) * np.nan), vmax=1, vmin=0, interpolation='none')
                    # cbar=plt.colorbar(ax=axes[1][1],mappable=m)
                    # cbar.set_label("Nbr of peaks")
 
                    legend_plots = []
                    legend_labels = []
                    x, y = np.where(np.nansum(new_peaks, axis=2) == np.max(np.nansum(new_peaks, axis=2)))
                    #x, y = np.where((np.nansum(new_peaks,axis=2)==0)&(np.nansum(peaks,axis=2)>0))
                    # print(x,y)
                    x, y = x[0], y[0]
 
                    time_axis = np.arange(estimated_cube.shape[2]) * time_interval
                    p1 = axes[0][1].step(time_axis, estimated_cube[x, y], c=colors[1], where='mid', lw=3)
                    p3 = axes[0][1].step(time_axis, dl.data_cube.data[x, y], c=colors[0], where='mid')
                    axes[0][1].set_yscale('log')
                    axes[0][1].fill_between(time_axis,
                                            maximum_for_eclipse_3sig(
                                                np.where(estimated_cube[x, y] > range_mu_3sig[0], estimated_cube[x, y],
                                                         np.nan)),
                                            minimum_for_peak_3sig(
                                                np.where(estimated_cube[x, y] > range_mu_3sig[0], estimated_cube[x, y],
                                                         np.nan)),
                                            alpha=0.3, facecolor=colors[1])
                    axes[0][1].fill_between(time_axis,
                                            maximum_for_eclipse_5sig(
                                                np.where(estimated_cube[x, y] > range_mu_5sig[0], estimated_cube[x, y],
                                                         np.nan)),
                                            minimum_for_peak_5sig(
                                                np.where(estimated_cube[x, y] > range_mu_5sig[0], estimated_cube[x, y],
                                                         np.nan)),
                                            alpha=0.3, facecolor=colors[1])
                    p2 = axes[0][1].fill(np.NaN, np.NaN, c=colors[1], alpha=0.3)
                    axes[0][1].set_xlabel("Time (s)")
                    axes[0][1].scatter(time_axis, peaks[x, y], c='r', marker='^', zorder=1)
                    axes[0][1].scatter(time_axis, new_peaks[x, y]+.5, c='g', marker='^', zorder=1)
                    axes[0][1].scatter(time_axis, eclipses[x, y], c='r', marker='v', zorder=1)
                    axes[0][1].scatter(time_axis, new_eclipses[x, y]+.5, c='g', marker='v', zorder=1)
 
                    second_axis_func = (lambda x: x / time_interval, lambda x: time_interval * x)
                    secax = axes[0][1].secondary_xaxis('top', functions=second_axis_func)
                    secax.set_xlabel("Time (frame #)")
                    legend_plots.append((p1[0], p2[0]))
                    legend_labels.append("Expected")
                    legend_plots.append((p3[0],))
                    legend_labels.append(f"Observed {x}-{y}")
 
                    axes[0][1].legend(legend_plots, legend_labels)
                    axes[0][0].axis('off')
                    axes[1][0].axis('off')
                    axes[1][1].axis('off')
                    plt.show()
    print("Peaks before vs. after:",np.sum(tab_old_peaks), np.sum(tab_new_peaks))
    print("Eclipses before vs. after:",np.sum(tab_old_eclipses), np.sum(tab_new_eclipses))
