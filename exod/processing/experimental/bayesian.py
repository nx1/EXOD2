import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import cmasher as cmr
from scipy.interpolate import interp1d
from exod.pre_processing.data_loader import DataLoader
from exod.xmm.observation import Observation
from exod.processing.experimental.template_based_background_inference import compute_expected_cube_using_templates
from exod.utils.synthetic_data import create_fake_burst
from matplotlib.colors import LogNorm

def bayes_factor_peak(mu,n):
    range_test = max(10*max(n,mu), 100)
    return np.log10(np.nansum(poisson.pmf(n, mu+np.arange(range_test)))/(range_test)) - np.log10(poisson.pmf(n,mu))

def bayes_factor_eclipse(mu,n):
    eclipse_range = np.linspace(0,mu, max(100,int(10*mu)))
    return np.log10(np.nansum(poisson.pmf(n, mu-eclipse_range))/(len(eclipse_range))) - np.log10(poisson.pmf(n,mu))

def precompute_bayes_limits(threshold):
    range_mu = np.geomspace(1e-5, 1e3, 500)
    tab_npeak, tab_neclipse = [],[]
    for mu in tqdm(range_mu):
        n_peak = 0
        while (np.isnan(bayes_factor_peak(mu, n_peak))) or (bayes_factor_peak(mu, n_peak)<threshold):
            n_peak+=1
        tab_npeak.append(n_peak)
        n_eclipse = int(mu)+1
        while bayes_factor_eclipse(mu, n_eclipse)<threshold:
            n_eclipse-=1
        tab_neclipse.append(n_eclipse)
    data=np.array([tab_npeak, tab_neclipse])
    np.savetxt(f'/home/erwan/Documents/GitHub/EXOD2/exod/utils/bayesfactorlimits_{threshold}.txt', data)

def load_precomputed_bayes_limits(threshold):
    range_mu = np.geomspace(1e-5, 1e3, 500)
    data = np.loadtxt(f'/home/erwan/Documents/GitHub/EXOD2/exod/utils/bayesfactorlimits_{threshold}.txt')
    minimum_for_peak = interp1d(range_mu, data[0])
    maximum_for_eclipse = interp1d(range_mu, data[1])
    return minimum_for_peak, maximum_for_eclipse



def accepted_n_values():
    range_n = np.arange(10)
    range_mu = np.geomspace(1e-3, 1e3, 500)
    tab_npeak, tab_neclipse = [],[]
    for mu in tqdm(range_mu):
        n_peak = 0
        while (np.isnan(bayes_factor_peak(mu, n_peak))) or (bayes_factor_peak(mu, n_peak)<5):
            n_peak+=1
        tab_npeak.append(n_peak)
        n_eclipse = int(mu)+1
        while bayes_factor_eclipse(mu, n_eclipse)<5:
            n_eclipse-=1
        tab_neclipse.append(n_eclipse)
    plt.figure()
    plt.plot(range_mu, range_mu)
    plt.plot(range_mu, range_mu-5*np.sqrt(range_mu), ls='--',c='k')
    plt.plot(range_mu, range_mu+5*np.sqrt(range_mu), ls='--',c='k')

    plt.fill_between(range_mu,tab_neclipse, tab_npeak,alpha=0.5)
    plt.loglog()
    plt.xlabel(r"$\mu$")
    plt.ylabel("Range of accepted # photons")
    plt.show()
# accepted_n_values()

def plot_some_n_bayes():
    range_n = np.arange(10)
    range_mu = np.geomspace(1e-3, 1e3, 500)
    plt.figure()
    colors = cmr.take_cmap_colors('cmr.ocean',N=len(range_n),cmap_range=(0,0.7))
    for c,n in zip(colors,range_n):
        bayes_peak = [bayes_factor_peak(mu, n) for mu in range_mu]
        bayes_eclipse = [bayes_factor_eclipse(mu, n) for mu in range_mu]
        plt.plot(range_mu, bayes_peak, label=n, c=c)
        plt.plot(range_mu, bayes_eclipse, c=c)
    plt.axhline(y=5, ls='--', lw=3, c="k")
    plt.legend()
    plt.xlabel("Mu")
    plt.ylabel("P(Peak|Data)/P(No Peak|Data)")
    plt.loglog()



def test_bayes_on_false_cube(size):
    minimum_for_peak, maximum_for_eclipse = load_precomputed_bayes_limits(3)
    cube = np.random.poisson(1e-1, (size,size,size))
    estimated = np.ones((size,size,size))*1e-1
    peaks = cube>minimum_for_peak(estimated)
    eclipse =  cube<maximum_for_eclipse(estimated)
    return np.sum(peaks), np.sum(eclipse)

def test_on_data(cube, expected, threshold):
    minimum_for_peak, maximum_for_eclipse = load_precomputed_bayes_limits(threshold=threshold)
    peaks = cube>minimum_for_peak(np.where(expected>0, expected, np.nan))
    eclipse =  cube<maximum_for_eclipse(np.where(expected>0, expected, np.nan))
    return peaks, eclipse

if __name__=="__main__":
    obsid='0886121001'#'0831790701' #
    for time_interval in (5,10,100,1000):
        size_arcsec = 20
        gti_only = False
        gti_threshold = 0.5
        min_energy = 0.5
        max_energy = 12.0

        observation = Observation(obsid)
        observation.get_files()

        event_list = observation.events_processed_pn[0]
        event_list.read()

        img = observation.images[0]
        img.read(wcs_only=True)

        dl = DataLoader(event_list=event_list, size_arcsec=size_arcsec, time_interval=time_interval, gti_only=gti_only,
                        obsid='0886121001',gti_threshold=gti_threshold, min_energy=min_energy, max_energy=max_energy)
        dl.run()

        cube = dl.data_cube.data
        rejected = dl.data_cube.bti_bin_idx

        cube_with_peak = cube+create_fake_burst(dl.data_cube,25, 25, time_peak_fraction=0.15,
                                           width_time=time_interval/2, amplitude=2)

        estimated_cube = compute_expected_cube_using_templates(cube_with_peak, rejected)
        peaks, eclipses=test_on_data(cube_with_peak, estimated_cube, threshold=3)
        fig, axes= plt.subplots(2,2)
        axes[0][0].imshow(np.nansum(cube_with_peak, axis=2), norm=LogNorm())
        axes[1][0].imshow(np.sum(peaks, axis=2), vmax=1,vmin=0)
        axes[1][1].imshow(np.sum(eclipses, axis=2), vmax=1,vmin=0)
        axes[0][1].plot(cube_with_peak[25,25])
        axes[0][1].plot(estimated_cube[25,25],ls=':')
        axes[0][1].set_yscale('log')
        print(np.sum(cube_with_peak[25-1:25+2,25-1:25+2]), np.sum(estimated_cube[25-1:25+2,25-1:25+2]))