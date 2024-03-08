import exod.utils.path as path
from exod.pre_processing.data_loader import DataLoader
from exod.processing.data_cube import DataCube
from exod.xmm.event_list import EventList
from exod.xmm.observation import Observation
from exod.processing.experimental.template_based_background_inference import compute_expected_cube_using_templates
from exod.utils.synthetic_data import create_fake_burst,create_fake_onebin_burst,create_fake_Nbins_burst

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import poisson
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


def precompute_bayes_limits(threshold_sigma):
    """Computes the maximum and minimum accepted observed counts, for a given range of mu, so that it is an acceptable
    Poissonian realisation at a given confidence threshold. It is faster to precompute them rather than computing it
    on the fly during the observation treatment"""
    range_mu = np.geomspace(1e-7, 5e3, 50000)
    if threshold_sigma==3:
        threshold_peak=5.94
        threshold_eclipse=5.70
    elif threshold_sigma==5:
        threshold_peak=13.27
        threshold_eclipse=12.38
    else:
        print("You need to precompute the corresponding values !")
    tab_npeak, tab_neclipse = [],[]
    for mu in tqdm(range_mu):
        range_n_peak =  np.arange(max(10*mu, 100))
        result=bayes_factor_peak(range_n_peak,mu)
        tab_npeak.append(range_n_peak[result>threshold_peak][0])

        range_n_eclipse = np.arange(2*int(mu)+1)
        result=bayes_factor_eclipse(range_n_eclipse,mu)
        tab_neclipse.append(range_n_eclipse[result<threshold_eclipse][0])

    plt.figure(figsize=(5, 5))
    plt.plot(range_mu, range_mu, label=r'$\mu=B$')
    plt.plot(range_mu, tab_npeak, ls=':', c='k', label=fr'$B_{{peak}} > 10^{threshold_peak}$', lw=1.0)
    plt.plot(range_mu, tab_neclipse, ls='--', c='k', label=f'$B_{{eclipse}} > 10^{threshold_eclipse}$', lw=1.0)
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


def peak_rate_estimate(fraction, N, mu):
    """Estimate the upper limit on the rate of the peak, given an expected and observed counts,
     and a confidence fraction"""
    return gammaincinv(N+1, fraction*gammaincc(N+1, mu) + gammainc(N+1, mu)) - mu


def eclipse_rate_estimate(fraction, N, mu):
    """Estimate the upper limit on the rate of the eclipse, given an expected and observed counts,
     and a confidence fraction"""
    return mu - gammaincinv(N+1, gammainc(N+1, mu) - fraction*gammainc(N+1, mu))


def variability_maps(cube, expected, threshold_sigma):
    """Returns two cubes with booleans where the rate correspond to a peak or an eclipse"""
    range_mu, minimum_for_peak, maximum_for_eclipse = load_precomputed_bayes_limits(threshold_sigma=threshold_sigma)
    peaks = cube>minimum_for_peak(np.where(expected>range_mu[0], expected, np.nan))
    eclipse =  cube<maximum_for_eclipse(np.where(expected>range_mu[0], expected, np.nan))
    return peaks, eclipse

def clean_up_peaks(cube, peaks):
    """Removes peaks coming from flaring CCD edges"""
    peak_x, peak_y, peak_t = np.where(peaks==True)
    for (x,y,t) in zip(peak_x, peak_y, peak_t):
        local_flux = np.nansum(cube[x-5:x+5,y-5:y+5,t])
        #We reject peaks for which the local emission is bright, more than twice the peak and 10% of the total frame
        if (local_flux> 2*cube[x,y,t])&(local_flux>0.1*np.nansum(cube[:,:,t])):
            peaks[x,y,t]=False
    return peaks

def repeated_peaks(cube,expected,threshold_sigma):
    peaks, eclipses = variability_maps(cube, expected, threshold_sigma)
    nbr_of_peaks = np.nansum(np.abs(np.diff(peaks, axis=2)),axis=2)/2
    return nbr_of_peaks

if __name__=="__main__":
    obsid='0765080801'#'0872390901' #'0886121001'#
    size_arcsec = 20
    time_interval = 5
    gti_only = False
    gti_threshold = 1.5
    min_energy = 0.2
    max_energy = 2.0

    threshold_sigma=3

    # Load data
    observation = Observation(obsid)
    observation.get_files()
    observation.get_events_overlapping_subsets()
    event_list = EventList.from_event_lists(observation.events_overlapping_subsets[0])
    # event_list = observation.events_processed_pn[0]
    # event_list.read()
    dl = DataLoader(event_list=event_list, size_arcsec=size_arcsec, time_interval=time_interval, gti_only=gti_only,
                    gti_threshold=gti_threshold, min_energy=min_energy, max_energy=max_energy)
    dl.run()
    cube = dl.data_cube.data
    rejected = dl.data_cube.bti_bin_idx
    print("Cube loaded ! Adding synthetic data...")
    cube_with_peak = cube + create_fake_Nbins_burst(dl.data_cube, 20, 30,
                                                     time_peak_fractions=(0.3,0.4,0.6,0.7), amplitude=30)
    estimated_cube = compute_expected_cube_using_templates(cube_with_peak, rejected)

    print("Computing variability...")
    peaks, eclipses=variability_maps(cube_with_peak, estimated_cube, threshold_sigma=threshold_sigma)
    nbr_of_peaks = repeated_peaks(cube_with_peak,estimated_cube,threshold_sigma=threshold_sigma)
    range_mu_3sig, minimum_for_peak_3sig, maximum_for_eclipse_3sig = load_precomputed_bayes_limits(threshold_sigma=3)
    range_mu_5sig, minimum_for_peak_5sig, maximum_for_eclipse_5sig = load_precomputed_bayes_limits(threshold_sigma=5)

    fig, axes= plt.subplots(2,2)
    colors=cmr.take_cmap_colors('cmr.ocean',N=2,cmap_range=(0,0.5))
    plt.suptitle(f'Binning {time_interval}s')
    axes[0][0].imshow(np.nansum(cube_with_peak, axis=2), norm=LogNorm(), interpolation='none')
    axes[1][0].imshow(np.where(np.nansum(cube_with_peak, axis=2)>0,np.nansum(peaks, axis=2),np.empty(cube.shape[:2])*np.nan),
                      vmax=1,vmin=0, interpolation='none')
    m=axes[1][1].imshow(np.where(np.nansum(cube_with_peak, axis=2)>0,np.nansum(eclipses, axis=2),np.empty(cube.shape[:2])*np.nan), interpolation='none')
    # m=axes[1][1].imshow(np.where(np.nansum(cube_with_peak, axis=2)>0,nbr_of_peaks,np.empty(cube.shape[:2])*np.nan), interpolation='none')
    # cbar=plt.colorbar(ax=axes[1][1],mappable=m)
    # cbar.set_label("Nbr of peaks")

    legend_plots=[]
    legend_labels=[]
    x,y = 82,62
    p3=axes[0][1].plot(range(estimated_cube.shape[2]), cube_with_peak[x,y],c=colors[0])
    p1=axes[0][1].plot(range(estimated_cube.shape[2]), estimated_cube[x,y],c=colors[1])
    axes[0][1].set_yscale('log')
    axes[0][1].fill_between(range(cube.shape[2]),
                            maximum_for_eclipse_3sig(np.where(estimated_cube[x,y]>range_mu_3sig[0], estimated_cube[x,y], np.nan)),
                            minimum_for_peak_3sig(np.where(estimated_cube[x,y]>range_mu_3sig[0], estimated_cube[x,y], np.nan)),
                            alpha=0.3,facecolor=colors[1])
    axes[0][1].fill_between(range(cube.shape[2]),
                            maximum_for_eclipse_5sig(np.where(estimated_cube[x,y]>range_mu_5sig[0], estimated_cube[x,y], np.nan)),
                            minimum_for_peak_5sig(np.where(estimated_cube[x,y]>range_mu_5sig[0], estimated_cube[x,y], np.nan)),
                            alpha=0.3,facecolor=colors[1])
    p2=axes[0][1].fill(np.NaN,np.NaN,c=colors[1],alpha=0.3)
    legend_plots.append((p1[0],p2[0]))
    legend_labels.append("Expected")
    legend_plots.append((p3[0],))
    legend_labels.append("Observed")
    axes[0][1].scatter(range(len(estimated_cube[x,y])), peaks[x,y],c='r',marker='^',zorder=1)
    axes[0][1].scatter(range(len(estimated_cube[x,y])), eclipses[x,y],c='r',marker='v',zorder=1)

    axes[0][1].legend(legend_plots,legend_labels)
    print(np.nansum(cube_with_peak[x, y]), np.sum(peaks[x,y]),
              np.nansum(cube_with_peak[x,y])-np.nansum(cube[x,y]),
              np.nansum(cube_with_peak)-np.nansum(cube))

    axes[0][0].axis('off')
    axes[1][0].axis('off')
    axes[1][1].axis('off')
    plt.show()

    fig,axes=plt.subplots(1,2)
    axes[0].imshow(np.where(np.nansum(cube_with_peak, axis=2)>0,np.nansum(peaks, axis=2),np.empty(cube.shape[:2])*np.nan),
                      vmax=1,vmin=0, interpolation='none')
    cleaned_peaks=clean_up_peaks(cube_with_peak,peaks)
    axes[1].imshow(
        np.where(np.nansum(cube_with_peak, axis=2) > 0, np.nansum(cleaned_peaks, axis=2), np.empty(cube.shape[:2]) * np.nan),
        vmax=1, vmin=0, interpolation='none')
    # for time_interval in (5,10,100,1000):
    #
    #     dl = DataLoader(event_list=event_list, size_arcsec=size_arcsec, time_interval=time_interval, gti_only=gti_only,
    #                     gti_threshold=gti_threshold, min_energy=min_energy, max_energy=max_energy)
    #     dl.run()
    #
    #     cube = dl.data_cube.data
    #     rejected = dl.data_cube.bti_bin_idx
    #
    #     cube_with_peak = cube+create_fake_onebin_burst(dl.data_cube,25, 25,
    #                                                    time_peak_fraction=0.3, amplitude=10)
    #
    #     estimated_cube = compute_expected_cube_using_templates(cube_with_peak, rejected)
    #     peaks, eclipses=variability_maps(cube_with_peak, estimated_cube, threshold_sigma=5)
    #     fig, axes= plt.subplots(2,2)
    #     plt.suptitle(f'Binning {time_interval}s')
    #     axes[0][0].imshow(np.nansum(cube_with_peak, axis=2), norm=LogNorm(), interpolation='none')
    #     axes[1][0].imshow(np.where(np.nansum(cube_with_peak, axis=2)>0,np.nansum(peaks, axis=2),np.empty(cube.shape[:2])*np.nan),
    #                       vmax=1,vmin=0, interpolation='none')
    #     axes[1][1].imshow(np.where(np.nansum(cube_with_peak, axis=2)>0,np.nansum(eclipses, axis=2),np.empty(cube.shape[:2])*np.nan),
    #                       vmax=1,vmin=0, interpolation='none')
    #     axes[0][1].plot(cube_with_peak[25,25])
    #     axes[0][1].plot(estimated_cube[25,25],ls=':')
    #     axes[0][1].set_yscale('log')
    #     print(np.nansum(cube_with_peak[25,25]),
    #           np.nansum(cube_with_peak[25,25])-np.nansum(cube[25,25]),
    #           np.nansum(cube_with_peak)-np.sum(cube))
