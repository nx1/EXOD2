import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import cmasher as cmr
from scipy.interpolate import interp1d
from exod.pre_processing.data_loader import DataLoader
from exod.xmm.observation import Observation
from exod.processing.experimental.template_based_background_inference import compute_expected_cube_using_templates
from exod.utils.synthetic_data import create_fake_burst,create_fake_onebin_burst,create_fake_Nbins_burst
from matplotlib.colors import LogNorm
import cmasher as cmr
from tqdm import tqdm
from math import factorial
from scipy.special import gammainc, gammaincc, gammaincinv


def bayes_factor_peak(mu,n):
    """Computes the Bayes factors of the presence of a peak, given the expected and observed counts mu and n"""
    return np.log10(gammaincc(n + 1, mu)) - np.log10(poisson.pmf(n,mu))

def bayes_factor_eclipse(mu,n):
    """Computes the Bayes factors of the presence of a peak, given the expected and observed counts mu and n"""
    return np.log10(gammainc(n + 1, mu)) - np.log10(poisson.pmf(n,mu))

def precompute_bayes_limits(threshold):
    """Computes the maximum and minimum accepted observed counts, for a given range of mu, so that it is an acceptable
    Poissonian realisation at a given confidence threshold. It is faster to precompute them rather than computing it
    on the fly during the observation treatment"""
    range_mu = np.geomspace(1e-5, 5e3, 5000)
    tab_npeak, tab_neclipse = [],[]
    for mu in tqdm(range_mu):
        range_n_peak =  np.arange(max(10*mu, 100))
        result=bayes_factor_peak(mu,range_n_peak)
        tab_npeak.append(range_n_peak[result>threshold][0])

        range_n_eclipse = np.arange(2*int(mu)+1)
        result=bayes_factor_eclipse(mu,range_n_eclipse)
        tab_neclipse.append(range_n_eclipse[result<threshold][0])

    plt.figure()
    plt.plot(range_mu, range_mu)
    plt.plot(range_mu, tab_npeak, ls=':', c='k')
    plt.plot(range_mu, tab_neclipse, ls='--', c='k')
    plt.fill_between(range_mu, range_mu-3*np.sqrt(range_mu), range_mu+3*np.sqrt(range_mu),alpha=0.2)
    plt.loglog()

    data=np.array([range_mu, tab_npeak, tab_neclipse])
    np.savetxt(f'/home/erwan/Documents/GitHub/EXOD2/exod/utils/bayesfactorlimits_{threshold}.txt', data)

def load_precomputed_bayes_limits(threshold):
    """Loads the precomputed Bayes factor limit numbers, for a chosen threshold. """
    data = np.loadtxt(f'/home/erwan/Documents/GitHub/EXOD2/exod/utils/bayesfactorlimits_{threshold}.txt')
    range_mu = data[0]
    minimum_for_peak = interp1d(range_mu, data[1])
    maximum_for_eclipse = interp1d(range_mu, data[2])
    return minimum_for_peak, maximum_for_eclipse

def peak_rate_estimate(fraction, mu, N):
    """Estimate the upper limit on the rate of the peak, given an expected and observed counts,
     and a confidence fraction"""
    return gammaincinv(N+1,fraction*gammaincc(N+1,mu)+gammainc(N+1,mu))-mu

def eclipse_rate_estimate(fraction, mu, N):
    """Estimate the upper limit on the rate of the eclipse, given an expected and observed counts,
     and a confidence fraction"""
    return mu-gammaincinv(N+1,gammainc(N+1,mu)-fraction*gammainc(N+1,mu))

def variability_maps(cube, expected, threshold):
    """Returns two cubes with booleans where the rate correspond to a peak or an eclipse"""
    minimum_for_peak, maximum_for_eclipse = load_precomputed_bayes_limits(threshold=threshold)
    peaks = cube>minimum_for_peak(np.where(expected>0, expected, np.nan))
    eclipse =  cube<maximum_for_eclipse(np.where(expected>0, expected, np.nan))
    return peaks, eclipse

def repeated_peaks(cube,expected,threshold):
    peaks, eclipses = variability_maps(cube, expected, threshold)
    nbr_of_peaks = np.nansum(np.abs(np.diff(peaks, axis=2)),axis=2)/2
    return nbr_of_peaks


if __name__=="__main__":
    obsid='0886121001'#'0831790701' #
    size_arcsec = 20
    gti_only = False
    gti_threshold = 0.5
    min_energy = 0.2
    max_energy = 12.0

    observation = Observation(obsid)
    observation.get_files()

    event_list = observation.events_processed_pn[0]
    event_list.read()

    img = observation.images[0]
    img.read(wcs_only=True)

    time_interval=500
    dl = DataLoader(event_list=event_list, size_arcsec=size_arcsec, time_interval=time_interval, gti_only=gti_only,
                    gti_threshold=gti_threshold, min_energy=min_energy, max_energy=max_energy)
    dl.run()
    cube = dl.data_cube.data
    rejected = dl.data_cube.bti_bin_idx
    cube_with_peak = cube + create_fake_Nbins_burst(dl.data_cube, 25, 25,
                                                     time_peak_fractions=(0.3,0.4,0.6,0.8), amplitude=40)
    estimated_cube = compute_expected_cube_using_templates(cube_with_peak, rejected)
    peaks, eclipses=variability_maps(cube_with_peak, estimated_cube, threshold=5)
    nbr_of_peaks = repeated_peaks(cube_with_peak,estimated_cube,threshold=5)
    minimum_for_peak, maximum_for_eclipse = load_precomputed_bayes_limits(threshold=5)

    fig, axes= plt.subplots(2,2)
    colors=cmr.take_cmap_colors('cmr.ocean',N=2,cmap_range=(0,0.3))
    plt.suptitle(f'Binning {time_interval}s')
    axes[0][0].imshow(np.nansum(cube_with_peak, axis=2), norm=LogNorm(), interpolation='none')
    axes[1][0].imshow(np.where(np.nansum(cube_with_peak, axis=2)>0,np.nansum(peaks, axis=2),np.empty(cube.shape[:2])*np.nan),
                      vmax=1,vmin=0, interpolation='none')
    m=axes[1][1].imshow(np.where(np.nansum(cube_with_peak, axis=2)>0,nbr_of_peaks,np.empty(cube.shape[:2])*np.nan), interpolation='none')
    cbar=plt.colorbar(ax=axes[1][1],mappable=m)
    cbar.set_label("Nbr of peaks")
    axes[0][1].plot(estimated_cube[25,25],c=colors[1], label="Expected")
    axes[0][1].fill_between(range(cube.shape[2]),maximum_for_eclipse(estimated_cube[25,25]),minimum_for_peak(estimated_cube[25,25]),alpha=0.1,facecolor=colors[1])
    axes[0][1].plot(cube_with_peak[25,25],c=colors[0], label="Observed")
    # axes[0][1].set_yscale('log')
    axes[0][1].legend()

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
    #     peaks, eclipses=variability_maps(cube_with_peak, estimated_cube, threshold=5)
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
    #     print(np.sum(cube_with_peak[25,25]),
    #           np.sum(cube_with_peak[25,25])-np.sum(cube[25,25]),
    #           np.sum(cube_with_peak)-np.sum(cube))
