import exod.utils.path as path
from exod.utils.logger import logger
from exod.pre_processing.data_loader import DataLoader
from exod.processing.data_cube import DataCube
from exod.xmm.event_list import EventList
from exod.xmm.observation import Observation
from exod.processing.experimental.template_based_background_inference import compute_expected_cube_using_templates
from exod.utils.synthetic_data import create_fake_burst,create_fake_onebin_burst,create_fake_Nbins_burst

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import poisson, binned_statistic_dd
from scipy.interpolate import interp1d
from scipy.special import gammainc, gammaincc, gammaincinv
from tqdm import tqdm
import cmasher as cmr

def bayes_factor_peak(n, mu):
    """Computes the Bayes factors of the presence of a peak, given the expected and observed counts mu and n"""
    return np.log10(gammaincc(n + 1, mu)) - np.log10(poisson.pmf(n, mu))


def bayes_factor_eclipse(n, mu):
    """Computes the Bayes factors of the presence of a peak, given the expected and observed counts mu and n"""
    return np.log10(gammainc(n + 1, mu)) - np.log10(poisson.pmf(n, mu))

def N_peaks_large_mu(mu, sigma):
    return np.ceil((2*mu+sigma**2+np.sqrt(8*mu*(sigma**2)+sigma**4))/2)

def N_eclipses_large_mu(mu, sigma):
    return np.floor((2*mu+sigma**2-np.sqrt(8*mu*(sigma**2)+sigma**4))/2)

def precompute_bayes_limits(threshold_sigma):
    """Computes the maximum and minimum accepted observed counts, for a given range of mu, so that it is an acceptable
    Poissonian realisation at a given confidence threshold. It is faster to precompute them rather than computing it
    on the fly during the observation treatment. For large numbers, we use a Gaussian approximation"""
    range_mu = np.geomspace(1e-7, 1e3, 50000)

    threshold_peak=bayes_factor_peak(N_peaks_large_mu(1000,threshold_sigma),1000)
    threshold_eclipse=bayes_factor_eclipse(N_eclipses_large_mu(1000,threshold_sigma),1000)

    tab_npeak, tab_neclipse = [],[]
    for mu in tqdm(range_mu):
        range_n_peak =  np.arange(max(10*mu, 100))
        result=bayes_factor_peak(range_n_peak,mu)
        tab_npeak.append(range_n_peak[result>threshold_peak][0])

        range_n_eclipse = np.arange(2*int(mu)+1)
        result=bayes_factor_eclipse(range_n_eclipse,mu)
        tab_neclipse.append(range_n_eclipse[result<threshold_eclipse][0])

    range_mu_large=np.geomspace(1e3,1e6,1000)

    tab_npeak+=list(N_peaks_large_mu(range_mu_large, threshold_sigma))
    tab_neclipse+=list(N_eclipses_large_mu(range_mu_large, threshold_sigma))
    range_mu=np.concatenate((range_mu,range_mu_large))


    plt.figure(figsize=(5, 5))
    plt.plot(range_mu, range_mu, label=r'$N=\mu$')
    plt.plot(range_mu, tab_npeak, ls=':', c='k', label=fr'$B_{{peak}} > 10^{{{threshold_peak}}}$', lw=1.0)
    plt.plot(range_mu, tab_neclipse, ls='--', c='k', label=f'$B_{{eclipse}} > 10^{{{threshold_eclipse}}}$', lw=1.0)
    plt.fill_between(range_mu, range_mu-3*np.sqrt(range_mu), range_mu+3*np.sqrt(range_mu), alpha=0.2, label=fr'$3 \sigma$ Region')
    plt.yscale('log')
    plt.xscale('log')
    #plt.loglog()
    plt.title(r'$B_{peak} = \frac{Q(N+1, \mu)}{e^{-\mu} \mu^{N} / N!} \ \  B_{eclipse} = \frac{P(N+1, \mu)}{e^{-\mu} \mu^{N} / N!}$')
    plt.xlabel(fr'Expected Counts $\mu$')
    plt.ylabel(fr'Observed Counts $N$')
    plt.xlim(min(range_mu), max(range_mu))
    plt.ylim(min(range_mu), max(range_mu))
    plt.legend()
    plt.tight_layout()

    data = np.array([range_mu, tab_npeak, tab_neclipse])
    np.savetxt(path.utils / f'bayesfactorlimits_{threshold_sigma}.txt', data)

def load_precomputed_bayes_limits(threshold_sigma):
    """Loads the precomputed Bayes factor limit numbers, for a chosen threshold. """
    data = np.loadtxt(path.utils / f'bayesfactorlimits_{threshold_sigma}.txt')
    range_mu = data[0]
    minimum_for_peak = interp1d(range_mu, data[1])
    maximum_for_eclipse = interp1d(range_mu, data[2])
    return range_mu, minimum_for_peak, maximum_for_eclipse

def precompute_bayes_1000():
    """Precomputes the Bayes factor at mu=1000 for a bunch of values of N. Will be interpolated to estimate the sigma"""
    range_N = np.arange(10000)
    tab_B_peaks = bayes_factor_peak(range_N, 1000)
    tab_B_eclipses = bayes_factor_eclipse(range_N, 1000)
    data = np.array([range_N, tab_B_peaks, tab_B_eclipses])
    np.savetxt(path.utils / f'bayesfactor_mu1000.txt', data)

def load_precomputed_bayes1000():
    """Loads & interpolates the precomputed values of Bayes factors at mu=1000"""
    data = np.loadtxt(path.utils / f'bayesfactor_mu1000.txt')
    range_N = data[0]
    B_values_peaks = interp1d(range_N, data[1])
    B_values_eclipses = interp1d(range_N, data[2])
    return range_N, B_values_peaks, B_values_eclipses

def variability_maps(cube, expected, threshold_sigma):
    """Returns two cubes with booleans where the rate correspond to a peak or an eclipse"""
    range_mu, minimum_for_peak, maximum_for_eclipse = load_precomputed_bayes_limits(threshold_sigma=threshold_sigma)
    peaks = cube>minimum_for_peak(np.where(expected>range_mu[0], expected, np.nan))
    eclipse =  cube<maximum_for_eclipse(np.where(expected>range_mu[0], expected, np.nan))
    return peaks, eclipse

if __name__=="__main__":
    from exod.pre_processing.download_observations import read_observation_ids
    from exod.utils.path import data
    from random import shuffle

    obsids = read_observation_ids(data / 'observations.txt')
    # obsids = read_observation_ids(data / 'obs_ccd_check.txt')
    # shuffle(obsids)
    # obsids=['0165560101','0765080801','0116700301', '0765080801', '0872390901', '0116700301'
    #     '0201900201', '0724840301', '0743700201']
    # obsids=['0112570701']
    # obsids=['0810811801']#'0764420101',
    obsids=['0911990501']
    for obsid in tqdm(obsids[:1]):
        # obsid='0765080801'#'0886121001'#'0872390901' #
        size_arcsec = 20
        time_interval = 25
        gti_only = False
        gti_threshold = 1.5
        min_energy = 0.2
        max_energy = 2.0

        threshold_sigma = 5

        # Load data
        observation = Observation(obsid)
        observation.get_files()
        try:
            observation.get_events_overlapping_subsets()
            for ind_exp, subset_overlapping_exposures in enumerate(observation.events_overlapping_subsets):
                event_list = EventList.from_event_lists(subset_overlapping_exposures)
                event_list.info
                while time_interval < 5000:
                    dl = DataLoader(event_list=event_list, size_arcsec=size_arcsec, time_interval=time_interval, gti_only=gti_only,
                                    gti_threshold=gti_threshold, min_energy=min_energy, max_energy=max_energy)
                    dl.run()
                    estimated_cube = compute_expected_cube_using_templates(dl.data_cube)

                    print("Computing variability...")
                    peaks, eclipses=variability_maps(dl.data_cube.data, estimated_cube, threshold_sigma=threshold_sigma)
                    range_mu_3sig, minimum_for_peak_3sig, maximum_for_eclipse_3sig = load_precomputed_bayes_limits(threshold_sigma=3)
                    range_mu_5sig, minimum_for_peak_5sig, maximum_for_eclipse_5sig = load_precomputed_bayes_limits(threshold_sigma=5)

                    fig, axes = plt.subplots(2,2, figsize=(10,10))
                    colors=cmr.take_cmap_colors('cmr.ocean',N=2,cmap_range=(0,0.5))
                    plt.suptitle(f'ObsID {obsid} -- Exposure {ind_exp} --  Binning {time_interval}s')
                    axes[0][0].imshow(np.nansum(dl.data_cube.data, axis=2), norm=LogNorm(), interpolation='none')
                    axes[1][0].imshow(np.where(np.nansum(dl.data_cube.data, axis=2)>0,np.nansum(peaks, axis=2),np.empty(dl.data_cube.shape[:2])*np.nan),
                                      vmax=1,vmin=0, interpolation='none')
                    m=axes[1][1].imshow(np.where(np.nansum(dl.data_cube.data, axis=2)>0,np.nansum(eclipses, axis=2),np.empty(dl.data_cube.shape[:2])*np.nan), interpolation='none')
                    # cbar=plt.colorbar(ax=axes[1][1],mappable=m)
                    # cbar.set_label("Nbr of peaks")

                    legend_plots=[]
                    legend_labels=[]
                    if np.max(np.nansum(peaks, axis=2))>0:
                        x,y = np.where(np.nansum(peaks, axis=2)==np.max(np.nansum(peaks, axis=2)))
                    elif np.max(np.nansum(eclipses, axis=2))>0:
                        x, y = np.where(np.nansum(eclipses, axis=2) == np.max(np.nansum(eclipses, axis=2)))
                    else:
                        x, y = np.where(np.nansum(dl.data_cube.data, axis=2) == np.max(np.nansum(dl.data_cube.data, axis=2)))
                    x,y=x[0],y[0]

                    time_axis = np.arange(estimated_cube.shape[2])*time_interval
                    p1=axes[0][1].step(time_axis, estimated_cube[x,y],c=colors[1], where='mid', lw=3)
                    p3=axes[0][1].step(time_axis, dl.data_cube.data[x,y],c=colors[0], where='mid')
                    axes[0][1].set_yscale('log')
                    # axes[0][1].bar(time_axis,minimum_for_peak_3sig(np.where(estimated_cube[x,y]>range_mu_3sig[0], estimated_cube[x,y], np.nan)),alpha=0.3,color=colors[1])
                    # axes[0][1].bar(time_axis,maximum_for_eclipse_3sig(np.where(estimated_cube[x,y]>range_mu_3sig[0], estimated_cube[x,y], np.nan)), color='w')
                    # axes[0][1].bar(time_axis,maximum_for_eclipse_5sig(np.where(estimated_cube[x,y]>range_mu_5sig[0], estimated_cube[x,y], np.nan)),
                    #                         minimum_for_peak_5sig(np.where(estimated_cube[x,y]>range_mu_5sig[0], estimated_cube[x,y], np.nan)),
                    #                         alpha=0.3,facecolor=colors[1])
                    axes[0][1].fill_between(time_axis,
                                            maximum_for_eclipse_3sig(np.where(estimated_cube[x,y]>range_mu_3sig[0], estimated_cube[x,y], np.nan)),
                                            minimum_for_peak_3sig(np.where(estimated_cube[x,y]>range_mu_3sig[0], estimated_cube[x,y], np.nan)),
                                            alpha=0.3,facecolor=colors[1])
                    axes[0][1].fill_between(time_axis,
                                            maximum_for_eclipse_5sig(np.where(estimated_cube[x,y]>range_mu_5sig[0], estimated_cube[x,y], np.nan)),
                                            minimum_for_peak_5sig(np.where(estimated_cube[x,y]>range_mu_5sig[0], estimated_cube[x,y], np.nan)),
                                            alpha=0.3,facecolor=colors[1])
                    p2=axes[0][1].fill(np.NaN,np.NaN,c=colors[1],alpha=0.3)
                    axes[0][1].set_xlabel("Time (s)")
                    axes[0][1].scatter(time_axis, peaks[x,y],c='r',marker='^',zorder=1)
                    axes[0][1].scatter(time_axis, eclipses[x,y],c='r',marker='v',zorder=1)

                    second_axis_func = (lambda x: x/time_interval, lambda x: time_interval*x)
                    secax = axes[0][1].secondary_xaxis('top', functions=second_axis_func)
                    secax.set_xlabel("Time (frame #)")
                    legend_plots.append((p1[0],p2[0]))
                    legend_labels.append("Expected")
                    legend_plots.append((p3[0],))
                    legend_labels.append(f"Observed {x}-{y}")

                    axes[0][1].legend(legend_plots,legend_labels)
                    axes[0][0].axis('off')
                    axes[1][0].axis('off')
                    axes[1][1].axis('off')

                    # plt.show()
                    # plt.figure()
                    # plt.imshow(dl.data_cube.data[:, :, 1], norm=LogNorm(), interpolation='none')
                    dl.multiply_time_interval(5)
                    time_interval=dl.data_cube.time_interval

        except Exception as e:
            logger.warning(f'{type(e).__name__} occurred: {e}')
        #
        # fig,axes=plt.subplots(1,2)
        # axes[0].imshow(np.where(np.nansum(cube_with_peak, axis=2)>0,np.nansum(peaks, axis=2),np.empty(cube.shape[:2])*np.nan),
        #                   vmax=1,vmin=0, interpolation='none')
        # cleaned_peaks=clean_up_peaks(cube_with_peak,peaks)
        # axes[1].imshow(
        #     np.where(np.nansum(cube_with_peak, axis=2) > 0, np.nansum(cleaned_peaks, axis=2), np.empty(cube.shape[:2]) * np.nan),
        #     vmax=1, vmin=0, interpolation='none')

