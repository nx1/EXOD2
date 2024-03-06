import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os
from tqdm import tqdm
from exod.pre_processing.data_loader import DataLoader
from exod.xmm.observation import Observation
from exod.processing.experimental.template_based_background_inference import compute_expected_cube_using_templates,compute_likelihood_variability
from exod.processing.experimental.bayesian import load_precomputed_bayes_limits, peak_rate_estimate
from exod.utils.synthetic_data import create_fake_burst
from scipy.stats import poisson
import cmasher as cmr


def check_estimate_success():
    fig, axes = plt.subplots(2,1)
    colors=cmr.take_cmap_colors('cmr.ocean',N=3,cmap_range=(0,.7))
    for peak, color in zip((0.5,2.5,10),colors):
        tab_up,tab_mid,tab_low=[],[],[]
        tab_N=[]
        tab_mu = np.geomspace(1e-3,1e2,100)
        for mu in tab_mu:
            N = np.random.poisson(mu+peak)
            tab_N.append(N)
            for fraction, tab in zip((0.01,0.5,0.99),(tab_low,tab_mid,tab_up)):
                tab.append(peak_rate_estimate(fraction, mu, N))
        axes[0].scatter(tab_mu,tab_N, c=color, label=f'Peak amplitude {peak}')
        axes[0].plot(tab_mu,tab_mid, c=color)
        axes[0].fill_between(tab_mu,tab_low,tab_up, alpha=0.4, facecolor=color)
        axes[0].set_xscale("log")
        axes[0].axhline(y=peak, c=color,ls='--')
        tab_mid = np.array(tab_mid)
        tab_low = np.array(tab_low)
        tab_up = np.array(tab_up)
        axes[1].errorbar(tab_mu,np.where(tab_mid>peak,
                                         (tab_mid-peak)/(tab_mid-tab_low),
                                         (peak - tab_mid) / (tab_up - tab_mid)
                                         ), yerr=1, fmt='o', c=color)
        axes[1].set_xscale("log")
    plt.show()

    plt.figure()
    for mu, color in zip((0.01,1,10),colors):
        tab_mid, tab_err, tab_errneg, tab_errpos=[],[],[],[]
        tab_peak = np.geomspace(1e-3,1e2,100)
        for peak in tab_peak:
        #     tabN = np.random.poisson(mu+peak, 50)
        #     tab_rates = [peak_rate_estimate(0.5,mu, N) for N in tabN]
        #     tab_mid.append(np.median(tab_rates))
        #     tab_err.append(np.std(tab_rates))
        # plt.errorbar(tab_peak,tab_mid,yerr=tab_err, c=color, fmt='o', label=f"$\mu={mu}$")
            N = np.random.poisson(mu+peak)
            tab_rates = peak_rate_estimate(np.array((0.16,0.5,0.84)),mu, N)
            tab_mid.append(tab_rates[1])
            tab_errneg.append(tab_rates[1]-tab_rates[0])
            tab_errpos.append(tab_rates[2] - tab_rates[1])
        plt.errorbar(tab_peak,tab_mid,yerr=[tab_errneg,tab_errpos], c=color, fmt='o', label=f"$\mu={mu}$")
    plt.loglog()
    plt.legend()
    plt.xlabel("Peak amplitude")
    plt.ylabel("Estimated peak amplitude")
    plt.plot(tab_peak, tab_peak,c='k')
    plt.show()

def check_eclipse_estimate_success():
    fig, axes = plt.subplots(2,1)
    colors=cmr.take_cmap_colors('cmr.ocean',N=4,cmap_range=(0,.7))
    for eclipse, color in zip((0.5,2.5,10),colors):
        tab_up,tab_mid,tab_low=[],[],[]
        tab_N=[]
        tab_mu = np.geomspace(1e1,1e5,100)
        for mu in tab_mu:
            N = np.random.poisson(max(mu-eclipse,0))
            tab_N.append(N)
            for fraction, tab in zip((0.01,0.5,0.99),(tab_low,tab_mid,tab_up)):
                tab.append(peak_rate_estimate(fraction, mu, N))
        axes[0].scatter(tab_mu,tab_N, c=color)
        axes[0].plot(tab_mu,tab_mid, c=color)
        axes[0].fill_between(tab_mu,tab_low,tab_up, alpha=0.4, facecolor=color)
        axes[0].set_xscale("log")
        axes[0].axhline(y=eclipse, c=color,ls='--')
        tab_mid = np.array(tab_mid)
        tab_low = np.array(tab_low)
        tab_up = np.array(tab_up)
        axes[1].errorbar(tab_mu,np.where(tab_mid>eclipse,
                                         (tab_mid-eclipse)/(tab_mid-tab_low),
                                         (eclipse - tab_mid) / (tab_up - tab_mid)
                                         ), yerr=1, fmt='o', c=color)
        axes[1].set_xscale("log")
    plt.show()

    plt.figure()
    for mu, color in zip((100,200,500,1000),colors):
        tab_mid, tab_err, tab_errneg, tab_errpos=[],[],[],[]
        tab_eclipse = np.geomspace(1e0,2e2,100)
        for eclipse in tab_eclipse:
            # tabN = np.random.poisson(max(mu-eclipse,0), 50)
            # tab_rates = [eclipse_rate_estimate(0.5,mu, N) for N in tabN]
            # tab_mid.append(np.median(tab_rates))
            # tab_err.append(np.std(tab_rates))
            N = np.random.poisson(max(mu - eclipse,0))
            rates = eclipse_rate_estimate(np.array((0.16,0.5,0.84)),mu, N)
            tab_mid.append(rates[1])
            tab_errneg.append(rates[1]-rates[0])
            tab_errpos.append(rates[2]-rates[1])
        plt.errorbar(tab_eclipse,tab_mid,yerr=[tab_errneg,tab_errpos], c=color, fmt='o', label=mu)
    plt.loglog()
    plt.xlabel("Eclipse amplitude")
    plt.legend()
    plt.plot(tab_eclipse, tab_eclipse,c='k')
    plt.show()

def plot_some_n_bayes():
    range_n = np.arange(10)
    range_mu = np.geomspace(1e-3, 1e3, 500)
    plt.figure()
    colors = cmr.take_cmap_colors('cmr.ocean',N=len(range_n),cmap_range=(0,0.7))
    for c,n in zip(colors,range_n):
        bayes_peak = [bayes_factor_new(mu, n) for mu in range_mu]#[bayes_factor_peak(mu, n) for mu in range_mu]
        bayes_eclipse = [bayes_factor_eclipse_new(mu, n) for mu in range_mu]
        plt.plot(range_mu, bayes_peak, label=n, c=c)
        plt.plot(range_mu, bayes_eclipse, c=c)
    plt.axhline(y=5, ls='--', lw=3, c="k")
    plt.legend()
    plt.xlabel("Mu")
    plt.ylabel("P(Peak|Data)/P(No Peak|Data)")
    plt.loglog()



def test_bayes_on_false_cube(size):
    # minimum_for_peak, maximum_for_eclipse = load_precomputed_bayes_limits(3)
    cube = np.random.poisson(1e-1, (size,size,size))
    estimated = np.ones((size,size,size))*1e-1
    peaks = bayes_factor_peak(estimated, cube)>5 #cube>minimum_for_peak(estimated)
    eclipse = bayes_factor_eclipse(estimated, cube)>5#cube<maximum_for_eclipse(estimated)
    print(np.sum(peaks), np.sum(eclipse))
    return np.sum(peaks), np.sum(eclipse)

def test_on_data(cube, expected, threshold):
    minimum_for_peak, maximum_for_eclipse = load_precomputed_bayes_limits(threshold=threshold)
    peaks = cube>minimum_for_peak(np.where(expected>0, expected, np.nan))
    eclipse =  cube<maximum_for_eclipse(np.where(expected>0, expected, np.nan))
    return peaks, eclipse

def accepted_n_values():
    """Testing function, showing the accepted counts for a range of mu. Similar to the pre-compute function"""
    range_n = np.arange(10)
    range_mu = np.geomspace(1e-3, 1e3, 5000)
    tab_npeak, tab_neclipse = [],[]
    for mu in tqdm(range_mu):
        range_n_peak =  np.arange(max(10*mu, 100))
        result=bayes_factor_peak(mu,range_n_peak)
        tab_npeak.append(range_n_peak[result>3][0])

        range_n_eclipse = np.arange(2*int(mu)+1)
        result=bayes_factor_eclipse(mu,range_n_eclipse)
        tab_neclipse.append(range_n_eclipse[result<3][0])
    plt.figure()
    plt.plot(range_mu, range_mu)
    plt.plot(range_mu, range_mu-3*np.sqrt(range_mu), ls='--',c='k')
    plt.plot(range_mu, range_mu+3*np.sqrt(range_mu), ls='--',c='k')

    plt.fill_between(range_mu,tab_neclipse, tab_npeak,alpha=0.5)
    plt.loglog()
    plt.xlabel(r"$\mu$")
    plt.ylabel("Range of accepted # photons")
    plt.show()
# accepted_n_values()

def bayes_rate_estimate(obsid='0886121001'):
    gti_threshold = 0.5
    min_energy = 0.2
    max_energy = 12.0
    size_arcsec=20
    timebin=10

    observation = Observation(obsid)
    observation.get_files()

    event_list = observation.events_processed_pn[0]
    event_list.read()

    img = observation.images[0]
    img.read(wcs_only=True)

    n_amplitude = 15
    n_draws = 50
    colors=cmr.take_cmap_colors('cmr.ocean',N=2,cmap_range=(0,0.7))

    minimum_for_peak, maximum_for_eclipse = load_precomputed_bayes_limits(threshold=3)

    dl = DataLoader(event_list=event_list, size_arcsec=size_arcsec, time_interval=timebin, gti_only=False,
                    gti_threshold=gti_threshold, min_energy=min_energy, max_energy=max_energy)
    dl.run()
    cube = dl.data_cube.data
    rejected = dl.data_cube.bti_bin_idx

    tab_result_gti = []
    tab_err_gti=[]
    tab_result_bti = []
    tab_err_bti=[]
    tab_amplitude = np.geomspace(5 / timebin, 100 / timebin, n_amplitude)
    for amplitude in tab_amplitude:
        tab_current_gti = []
        tab_current_bti = []
        for trial in tqdm(range(n_draws)):
            x_pos, y_pos = np.random.randint(10, cube.shape[0] - 10), np.random.randint(10, cube.shape[1] - 10)
            while np.sum(cube[x_pos, y_pos]) < 1:
                x_pos, y_pos = np.random.randint(10, cube.shape[0] - 10), np.random.randint(10, cube.shape[1] - 10)
            time_fraction = np.random.random()
            cube_with_peak = cube + create_fake_burst(dl.data_cube, x_pos, y_pos, time_peak_fraction=time_fraction,
                                                      width_time=timebin / 2, amplitude=amplitude)
            estimated_cube = compute_expected_cube_using_templates(cube_with_peak, rejected)
            peaks = cube_with_peak > minimum_for_peak(np.where(estimated_cube > 0, estimated_cube, np.nan))
            if int(time_fraction * cube.shape[2]) in rejected:
                if np.max(peaks[x_pos, y_pos]) > 0:
                    tab_current_bti.append(peak_rate_estimate(0.5,estimated_cube[x_pos, y_pos],cube_with_peak[x_pos, y_pos]))
            else:
                if np.max(peaks[x_pos, y_pos]) > 0:
                    tab_current_gti.append(peak_rate_estimate(0.5,estimated_cube[x_pos, y_pos],cube_with_peak[x_pos, y_pos]))
        tab_result_gti.append(np.mean(tab_current_gti))
        tab_err_gti.append(np.std(tab_current_gti))
        tab_result_bti.append(np.mean(tab_current_bti))
        tab_err_bti.append(np.std(tab_current_bti))

    plt.figure()
    plt.errorbar(tab_amplitude, tab_result_gti, yerr=tab_err_gti, c=colors[0], label='GTI',fmt='o')
    plt.errorbar(tab_amplitude, tab_result_bti, yerr=tab_err_bti, c=colors[1], label='BTI',fmt='o')
    plt.legend()
    plt.xlabel('True peak amplitude')
    plt.ylabel('Estimated amplitude')
    plt.xscale('log')
    plt.yscale("log")
    plt.show()

def bayes_successrate_spacebinning(obsid='0886121001'):
    gti_threshold = 0.5
    min_energy = 0.2
    max_energy = 12.0

    observation = Observation(obsid)
    observation.get_files()

    event_list = observation.events_processed_pn[0]
    event_list.read()

    img = observation.images[0]
    img.read(wcs_only=True)

    n_spacebins=4
    n_amplitude = 15
    n_draws = 50
    colors=cmr.take_cmap_colors('cmr.ocean',N=n_spacebins,cmap_range=(0,0.7))
    spacebins = np.geomspace(5,30,n_spacebins)
    all_spacebin_results=[]

    minimum_for_peak, maximum_for_eclipse = load_precomputed_bayes_limits(threshold=3)

    tab_all_amplitudes=[]
    timebin=100
    for size_arcsec in spacebins:
        dl = DataLoader(event_list=event_list, size_arcsec=size_arcsec, time_interval=timebin, gti_only=False,
                        gti_threshold=gti_threshold, min_energy=min_energy, max_energy=max_energy)
        dl.run()
        cube = dl.data_cube.data
        rejected = dl.data_cube.bti_bin_idx

        tab_result_gti=[]
        tab_result_bti=[]
        tab_amplitude = np.geomspace(1/timebin, 100/timebin, n_amplitude)
        tab_all_amplitudes.append(tab_amplitude)
        for amplitude in tab_amplitude:
            nbr_caught_gti=0
            nbr_caught_bti=0
            n_draws_bti=0
            n_draws_gti=0
            for trial in tqdm(range(n_draws)):
                x_pos, y_pos = np.random.randint(10,cube.shape[0]-10),np.random.randint(10,cube.shape[1]-10)
                while np.sum(cube[x_pos,y_pos])<1:
                    x_pos, y_pos = np.random.randint(10, cube.shape[0] - 10), np.random.randint(10, cube.shape[1] - 10)
                time_fraction = np.random.random()
                cube_with_peak = cube+create_fake_burst(dl.data_cube,x_pos, y_pos, time_peak_fraction=time_fraction,
                                                   width_time=timebin/2, amplitude=amplitude)
                estimated_cube = compute_expected_cube_using_templates(cube_with_peak, rejected)
                peaks = cube_with_peak>minimum_for_peak(np.where(estimated_cube>0, estimated_cube, np.nan))
                if int(time_fraction*cube.shape[2]) in rejected:
                    n_draws_bti+=1
                    if np.max(peaks[x_pos, y_pos]) > 0:
                        nbr_caught_bti+=1
                else:
                    n_draws_gti+=1
                    if np.max(peaks[x_pos, y_pos]) > 0:
                        nbr_caught_gti+=1
            tab_result_gti.append(nbr_caught_gti/n_draws_gti)
            if n_draws_bti>0:
                tab_result_bti.append(nbr_caught_bti/n_draws_bti)
            else:
                tab_result_bti.append(np.nan)
        all_spacebin_results.append([tab_result_gti,tab_result_bti])

    plt.figure()
    for (tab_result_gti,tab_result_bti), spacebin, tab_amplitude, color in zip(all_spacebin_results, spacebins, tab_all_amplitudes,colors):
        plt.plot(tab_amplitude, tab_result_gti, c=color, label=f'{int(spacebin)}"')
        plt.plot(tab_amplitude, tab_result_bti, c=color, ls="--")
        # plt.fill_between(tab_amplitude, np.array(tab_result)-np.sqrt(np.array(tab_result))/np.sqrt(n_draws),
        #                  np.array(tab_result) + np.sqrt(np.array(tab_result)) / np.sqrt(n_draws),
        #                  facecolor = color, alpha=0.5)
    plt.legend()
    plt.xlabel('Peak amplitude')
    plt.ylabel('Fraction of detected')
    plt.xscale('log')
    plt.show()

    plt.figure()
    for (tab_result_gti,tab_result_bti), spacebin, tab_amplitude, color in zip(all_spacebin_results, spacebins,tab_all_amplitudes, colors):
        plt.plot(tab_amplitude*timebin, tab_result_gti, c=color, label=f'{int(spacebin)}"')
        plt.plot(tab_amplitude*timebin, tab_result_bti, c=color, ls="--")
        # plt.fill_between(tab_amplitude*timebin, np.array(tab_result)-np.sqrt(np.array(tab_result))/np.sqrt(n_draws),
        #                  np.array(tab_result) + np.sqrt(np.array(tab_result)) / np.sqrt(n_draws),
        #                  facecolor = color, alpha=0.3)
    plt.legend()
    plt.xlabel('Peak count')
    plt.ylabel('Fraction of detected')
    plt.xscale('log')
    plt.show()

def bayes_successrate_timebinning(obsid='0886121001'):
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

    n_timebins=4
    n_amplitude = 15
    n_draws = 50
    colors=cmr.take_cmap_colors('cmr.ocean',N=n_timebins,cmap_range=(0,0.7))
    timebins = np.geomspace(10,1000,n_timebins)
    all_timebin_results=[]

    minimum_for_peak, maximum_for_eclipse = load_precomputed_bayes_limits(threshold=5)

    tab_all_amplitudes=[]
    for timebin in timebins:
        dl = DataLoader(event_list=event_list, size_arcsec=size_arcsec, time_interval=timebin, gti_only=gti_only,
                        gti_threshold=gti_threshold, min_energy=min_energy, max_energy=max_energy)
        dl.run()
        cube = dl.data_cube.data
        rejected = dl.data_cube.bti_bin_idx

        tab_result_gti=[]
        tab_result_bti=[]
        tab_amplitude = np.geomspace(1/timebin, 100/timebin, n_amplitude)
        tab_all_amplitudes.append(tab_amplitude)
        for amplitude in tab_amplitude:
            nbr_caught_gti=0
            nbr_caught_bti=0
            n_draws_bti=0
            n_draws_gti=0
            for trial in tqdm(range(n_draws)):
                x_pos, y_pos = np.random.randint(5,cube.shape[0]-5),np.random.randint(5,cube.shape[1]-5)
                time_fraction = np.random.random()
                cube_with_peak = cube+create_fake_burst(dl.data_cube,x_pos, y_pos, time_peak_fraction=time_fraction,
                                                   width_time=timebin/2, amplitude=amplitude)
                estimated_cube = compute_expected_cube_using_templates(cube_with_peak, rejected)
                peaks = cube_with_peak>minimum_for_peak(np.where(estimated_cube>0, estimated_cube, np.nan))
                if int(time_fraction*cube.shape[2]) in rejected:
                    n_draws_bti+=1
                    if np.max(peaks[x_pos, y_pos]) > 0:
                        nbr_caught_bti+=1
                else:
                    n_draws_gti+=1
                    if np.max(peaks[x_pos, y_pos]) > 0:
                        nbr_caught_gti+=1
            tab_result_gti.append(nbr_caught_gti/n_draws_gti)
            if n_draws_bti>0:
                tab_result_bti.append(nbr_caught_bti/n_draws_bti)
            else:
                tab_result_bti.append(np.nan)
        all_timebin_results.append([tab_result_gti,tab_result_bti])

    plt.figure()
    for (tab_result_gti,tab_result_bti), timebin,tab_amplitude, color in zip(all_timebin_results, timebins, tab_all_amplitudes,colors):
        plt.plot(tab_amplitude, tab_result_gti, c=color, label=f'{int(timebin)}s')
        plt.plot(tab_amplitude, tab_result_bti, c=color, ls="--")
        # plt.fill_between(tab_amplitude, np.array(tab_result)-np.sqrt(np.array(tab_result))/np.sqrt(n_draws),
        #                  np.array(tab_result) + np.sqrt(np.array(tab_result)) / np.sqrt(n_draws),
        #                  facecolor = color, alpha=0.5)
    plt.legend()
    plt.xlabel('Peak amplitude')
    plt.ylabel('Fraction of detected')
    plt.xscale('log')
    plt.show()

    plt.figure()
    for (tab_result_gti,tab_result_bti), timebin, tab_amplitude, color in zip(all_timebin_results, timebins,tab_all_amplitudes, colors):
        plt.plot(tab_amplitude*timebin, tab_result_gti, c=color, label=f'{int(timebin)}s')
        plt.plot(tab_amplitude*timebin, tab_result_bti, c=color, ls="--")
        # plt.fill_between(tab_amplitude*timebin, np.array(tab_result)-np.sqrt(np.array(tab_result))/np.sqrt(n_draws),
        #                  np.array(tab_result) + np.sqrt(np.array(tab_result)) / np.sqrt(n_draws),
        #                  facecolor = color, alpha=0.3)
    plt.legend()
    plt.xlabel('Peak count')
    plt.ylabel('Fraction of detected')
    plt.xscale('log')
    plt.show()


