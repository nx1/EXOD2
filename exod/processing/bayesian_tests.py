"""
This module contains functions to test the Bayesian computations in the exod package.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from exod.processing.data_cube import DataCube
from exod.processing.pipeline import Pipeline
from exod.utils.path import data_plots
from exod.xmm.observation import Observation
from exod.xmm.event_list import EventList
from exod.processing.background_inference import calc_cube_mu
from exod.processing.bayesian_computations import load_precomputed_bayes_limits, get_cube_masks_peak_and_eclipse, \
    B_peak_log, B_eclipse_log, get_bayes_thresholds, sigma_equivalent_B_peak, sigma_equivalent_B_eclipse, \
    PrecomputeBayesLimits
from exod.post_processing.estimate_variability_properties import peak_count_estimate, eclipse_count_estimate
from exod.utils.synthetic_data import create_fake_burst, create_fake_onebin_burst, create_multiple_fake_eclipses
import cmasher as cmr


def check_estimate_success():
    """
    If I remember correctly, it is to assess if the count-rate estimation
    worked, independently of the value of the quiescent state mu. So, for a
    range of mu, I add a peak of three different amplitudes, I generate poisson
    realisations of mu+peak (the crosses), and see the value we can estimate
    for the peak (the envelopes). You have the residuals at the bottom It might
    be an earlier version, and not work anymore. It was just to check the
    success of the rate estimate function
    """
    fig, ax = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios' : [3, 1]})
    colors = cmr.take_cmap_colors(cmap='cmr.ocean', N=3, cmap_range=(0, 0.7))
    all_res = []

    for peak, color in zip((0.5, 2.5, 10), colors):
        tab_up, tab_mid, tab_low = [], [], []
        tab_N = []

        tab_mu = np.geomspace(start=1e-3, stop=1e2, num=100)
        for mu in tab_mu:
            N = np.random.poisson(mu+peak)
            tab_N.append(N)
            for fraction, tab in zip((0.01, 0.5, 0.99), (tab_low, tab_mid, tab_up)):
                peak_est = peak_count_estimate(fraction, N, mu)
                tab.append(peak_est)

                res = {}
                res['peak'] = peak
                res['mu'] = mu
                res['N'] = N
                res['fraction'] = fraction
                res['peak_count_estimate'] = peak_est
                all_res.append(res)

        tab_mid = np.array(tab_mid)
        tab_low = np.array(tab_low)
        tab_up  = np.array(tab_up)
        residual = np.where(tab_mid > peak,
                            (tab_mid - peak) / (tab_mid - tab_low),
                            (peak - tab_mid) / (tab_up - tab_mid))

        ax[0].fill_between(tab_mu, tab_low, tab_up, alpha=0.4, facecolor=color)
        ax[0].scatter(tab_mu, tab_N, color=color, s=10, marker='x', label=f'Peak amplitude={peak}')
        ax[0].plot(tab_mu, tab_mid, color=color)
        ax[0].axhline(y=peak, color=color, ls='--')

        ax[1].errorbar(tab_mu, residual, yerr=1, fmt='.', color=color, capsize=1.0, lw=1.0)
        ax[1].axhline(0, ls='--', color='black', lw=1.0)

        ax[0].set_xlim(0)
        ax[0].set_ylim(0)
        ax[0].set_xscale('log')
        ax[0].set_ylabel('Observed (n)')
        ax[1].set_xscale("log")
        ax[1].set_xlabel(r'Expectation ($\mu$)')
        ax[1].set_ylabel('Residual')

    plt.subplots_adjust(hspace=0)
    ax[0].legend(loc='upper left')
    plt.savefig(data_plots / 'check_estimate_success1.png')
    plt.savefig(data_plots / 'check_estimate_success1.pdf')
    df_res = pd.DataFrame(all_res)
    print(df_res)



    plt.figure(figsize=(6,6))
    for mu, color in zip((0.01, 1, 10), colors):
        tab_mid, tab_err, tab_errneg, tab_errpos=[],[],[],[]
        tab_peak = np.geomspace(1e-3,1e2,100)
        for peak in tab_peak:
        #     tabN = np.random.poisson(mu+peak, 50)
        #     tab_rates = [peak_rate_estimate(0.5,mu, N) for N in tabN]
        #     tab_mid.append(np.median(tab_rates))
        #     tab_err.append(np.std(tab_rates))
        # plt.errorbar(tab_peak,tab_mid,yerr=tab_err, color=color, fmt='o', label=f"$\mu={mu}$")
            N = np.random.poisson(mu+peak)
            tab_rates = peak_count_estimate(np.array((0.16, 0.5, 0.84)), N, mu)
            tab_mid.append(tab_rates[1])
            tab_errneg.append(tab_rates[1]-tab_rates[0])
            tab_errpos.append(tab_rates[2] - tab_rates[1])
        plt.errorbar(tab_peak, tab_mid, yerr=[tab_errneg,tab_errpos],
                     color=color, label=fr"$\mu={mu}$", lw=1.0, capsize=1.0, fmt='.', markersize=10)
    plt.loglog()
    plt.xlabel("Peak amplitude (n)")
    plt.ylabel("Estimated peak amplitude (n)")
    plt.plot(tab_peak, tab_peak, c='k')
    plt.xlim(min(tab_peak), max(tab_peak))
    plt.ylim(min(tab_peak), max(tab_peak))
    plt.legend()
    plt.savefig(data_plots / 'check_estimate_success2.png')
    plt.savefig(data_plots / 'check_estimate_success2.pdf')
    plt.show()


def check_eclipse_estimate_success():
    fig, ax = plt.subplots(2,1, sharex=True)
    colors = cmr.take_cmap_colors('cmr.ocean', N=4, cmap_range=(0,.7))
    all_res = []
    for eclipse, color in zip((0.5,2.5,10),colors):
        tab_up, tab_mid, tab_low = [], [], []
        tab_N = []
        tab_mu = np.geomspace(1e1,1e5,100)
        for mu in tab_mu:
            N = np.random.poisson(max(mu-eclipse, 0))
            tab_N.append(N)
            for fraction, tab in zip((0.01,0.5,0.99), (tab_low,tab_mid,tab_up)):
                peak_count_est = peak_count_estimate(fraction, N, mu)
                tab.append(peak_count_est)
                res = {}
                res['eclipse'] = eclipse
                res['mu'] = mu
                res['N'] = N
                res['fraction'] = fraction
                res['peak_count_est'] = peak_count_est
                print(res)
                all_res.append(res)


        tab_mid = np.array(tab_mid)
        tab_low = np.array(tab_low)
        tab_up = np.array(tab_up)
        residual = np.where(tab_mid > eclipse,
                            (tab_mid - eclipse) / (tab_mid - tab_low),
                            (eclipse - tab_mid) / (tab_up - tab_mid))

        ax[0].fill_between(tab_mu, tab_low, tab_up, alpha=0.4, facecolor=color)
        ax[0].scatter(tab_mu, tab_N, color=color, marker='x', s=10)
        ax[0].plot(tab_mu, tab_mid, color=color, label=f'eclipse={eclipse}')
        ax[0].axhline(y=eclipse, color=color, ls='--')
        ax[1].errorbar(tab_mu, residual, yerr=1, fmt='.', color=color, markersize=5, lw=1.0)

        ax[0].set_xscale("log")
        ax[1].set_xscale("log")
        ax[0].set_ylabel('Observed (N)')
        ax[1].set_xlabel(r'Expected ($\mu$)')
        ax[1].set_ylabel('Residual')
    ax[0].legend(loc='upper left')
    plt.subplots_adjust(hspace=0)
    plt.savefig(data_plots / 'check_eclipse_estimate_success1.png')
    plt.savefig(data_plots / 'check_eclipse_estimate_success1.pdf')
    plt.show()
    df = pd.DataFrame(all_res)
    print(df)


    plt.figure(figsize=(6,6))
    for mu, color in zip((100,200,500,1000), colors):
        tab_mid, tab_err, tab_errneg, tab_errpos=[], [], [], []
        tab_eclipse = np.geomspace(start=1e0, stop=2e2, num=100)
        for eclipse in tab_eclipse:
            # tabN = np.random.poisson(max(mu-eclipse,0), 50)
            # tab_rates = [eclipse_rate_estimate(0.5,mu, N) for N in tabN]
            # tab_mid.append(np.median(tab_rates))
            # tab_err.append(np.std(tab_rates))
            N = np.random.poisson(max(mu - eclipse,0))
            rates = eclipse_count_estimate(np.array((0.16, 0.5, 0.84)), mu, N)
            tab_mid.append(rates[1])
            tab_errneg.append(rates[1]-rates[0])
            tab_errpos.append(rates[2]-rates[1])
        plt.errorbar(tab_eclipse, tab_mid, yerr=[tab_errneg, tab_errpos], color=color, fmt='.',
                     markersize=5, lw=1.0, label=fr'$\mu$={mu}')

    plt.plot(tab_eclipse, tab_eclipse, c='k')
    plt.loglog()
    plt.xlabel('Eclipse amplitude')
    plt.ylabel('Eclipse Count Estimate')
    plt.legend()
    plt.savefig(data_plots / 'check_eclipse_estimate_success2.png')
    plt.savefig(data_plots / 'check_eclipse_estimate_success2.pdf')
    plt.show()


def plot_mu_vs_bayes_factor_for_diff_n():
    range_n = [0,1,2,3,5,10,20,50,100]
    range_mu = np.geomspace(start=1e-3, stop=1e3, num=500)

    plt.figure(figsize=(5, 5))
    colors = cmr.take_cmap_colors(cmap='winter', N=len(range_n), cmap_range=(0, 1.0))
    for c, n in zip(colors, range_n):
        B_peak = [B_peak_log(n=n, mu=mu) for mu in range_mu]
        B_eclipse = [B_eclipse_log(n=n, mu=mu) for mu in range_mu]
        plt.plot(range_mu, B_peak, label=rf'N={n}', c=c)
        plt.plot(range_mu, B_eclipse, c=c)
    plt.axhline(y=5, ls='--', lw=1.0, c="k", label="5")
    plt.ylim(0.001, 1000)
    plt.legend()
    plt.xlabel(r"Expectation Value $\mu$")
    plt.ylabel(r"Bayes Factor (B)")
    plt.loglog()
    plt.xlim(1e-3, 1e3)
    plt.ylim(1e-3, 1e3)
    plt.savefig(data_plots / 'plot_some_n_bayes.png')
    plt.savefig(data_plots / 'plot_some_n_bayes.pdf')
    plt.show()


def bayes_test_on_false_cube(size):
    # minimum_for_peak, maximum_for_eclipse = load_precomputed_bayes_limits(3)
    cube = np.random.poisson(1e-1, (size,size,size))
    estimated = np.ones((size,size,size))*1e-1
    peaks = B_peak_log(estimated, cube) > 5 #cube_n>minimum_for_peak(estimated)
    eclipse = B_eclipse_log(estimated, cube) > 5#cube_n<maximum_for_eclipse(estimated)
    print(np.sum(peaks), np.sum(eclipse))
    return np.sum(peaks), np.sum(eclipse)


def bayes_test_on_data(cube, expected, threshold):
    minimum_for_peak, maximum_for_eclipse = load_precomputed_bayes_limits(threshold_sigma=threshold)
    peaks = cube>minimum_for_peak(np.where(expected>0, expected, np.nan))
    eclipse =  cube<maximum_for_eclipse(np.where(expected>0, expected, np.nan))
    return peaks, eclipse

def bayes_rate_estimate(obsid='0886121001'):
    gti_threshold = 0.5
    min_energy    = 0.2
    max_energy    = 12.0
    size_arcsec   = 20
    timebin       = 10

    observation = Observation(obsid)
    observation.get_files()

    event_list = observation.events_processed_pn[0]
    event_list.read()

    img = observation.images[0]
    img.read(wcs_only=True)

    range_mu, minimum_for_peak, maximum_for_eclipse = load_precomputed_bayes_limits(threshold_sigma=3)

    # We create two copies of DataCube()
    # The first (dl.datacube) will be used to create the original + synthetic data
    # we then set the data of the copied cube to this. This is extremely messy but it works for now.
    data_cube = DataCube(event_list, size_arcsec=20, time_interval=timebin)
    cube = data_cube.data
    rejected = data_cube.bti_bin_idx
    print('Creating copy of datacube (takes a second)...')
    data_cube2 = data_cube.copy()

    tab_result_gti = []
    tab_err_gti = []
    tab_result_bti = []
    tab_err_bti = []

    n_amplitude = 5
    n_draws = 5
    tab_amplitude = np.geomspace(5 / timebin, 100 / timebin, n_amplitude)

    for amplitude in tab_amplitude:
        tab_current_gti = []
        tab_current_bti = []
        for trial in range(n_draws):
            x_pos, y_pos = np.random.randint(10, cube.shape[0] - 10), np.random.randint(10, cube.shape[1] - 10)
            while np.sum(cube[x_pos, y_pos]) < 1:
                x_pos, y_pos = np.random.randint(10, cube.shape[0] - 10), np.random.randint(10, cube.shape[1] - 10)

            time_fraction = np.random.random()
            data_cube2.data = data_cube.data + create_fake_burst(data_cube=data_cube, x_pos=x_pos, y_pos=y_pos, time_peak_fraction=time_fraction, width_time=timebin / 2, amplitude=amplitude)
            estimated_cube  = calc_cube_mu(data_cube=data_cube2, wcs=img.wcs)
            cube_with_peak  = data_cube2.data

            peaks = cube_with_peak > minimum_for_peak(np.where(estimated_cube > 0, estimated_cube, np.nan))
            if int(time_fraction * cube.shape[2]) in rejected:
                if np.max(peaks[x_pos, y_pos]) > 0:
                    tab_current_bti.append(peak_count_estimate(0.5, cube_with_peak[x_pos, y_pos], estimated_cube[x_pos, y_pos]))
            else:
                if np.max(peaks[x_pos, y_pos]) > 0:
                    tab_current_gti.append(peak_count_estimate(0.5, cube_with_peak[x_pos, y_pos], estimated_cube[x_pos, y_pos]))

            print(f'amplitude={amplitude} trial={trial}/{n_draws}')
        tab_result_gti.append(np.mean(tab_current_gti))
        tab_err_gti.append(np.std(tab_current_gti))
        tab_result_bti.append(np.mean(tab_current_bti))
        tab_err_bti.append(np.std(tab_current_bti))

    plt.figure()
    colors = cmr.take_cmap_colors('cmr.ocean', N=2, cmap_range=(0,0.7))
    plt.errorbar(tab_amplitude, tab_result_gti, yerr=tab_err_gti, color=colors[0], label='GTI',fmt='o')
    plt.errorbar(tab_amplitude, tab_result_bti, yerr=tab_err_bti, color=colors[1], label='BTI',fmt='o')
    plt.legend()
    plt.xlabel('True peak amplitude')
    plt.ylabel('Estimated amplitude')
    # plt.xscale('log')
    # plt.yscale("log")
    plt.savefig(data_plots / 'bayes_rate_estimate.png')
    plt.savefig(data_plots / 'bayes_rate_estimate.pdf')
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

    n_spacebins = 4
    n_amplitude = 15
    n_draws = 5
    timebin = 100
    spacebins = np.geomspace(5,30, n_spacebins)

    all_spacebin_results = []

    range_mu, minimum_for_peak, maximum_for_eclipse = load_precomputed_bayes_limits(threshold_sigma=3)

    tab_all_amplitudes = []
    for size_arcsec in spacebins:
        data_cube = DataCube(event_list, size_arcsec=size_arcsec, time_interval=timebin)
        cube = data_cube.data
        rejected = data_cube.bti_bin_idx
        print('Creating copy of datacube (takes a second)...')
        data_cube2 = data_cube.copy()

        tab_result_gti = []
        tab_result_bti = []
        tab_amplitude = np.geomspace(1/timebin, 100/timebin, n_amplitude)
        tab_all_amplitudes.append(tab_amplitude)
        for amplitude in tab_amplitude:
            nbr_caught_gti = 0
            nbr_caught_bti = 0
            n_draws_bti = 0
            n_draws_gti = 0
            for trial in range(n_draws):
                print(f'size_arcsec    = {size_arcsec}\n'
                      f'amplitude      = {amplitude}\n'
                      f'trial          = {trial}/{n_draws}\n'
                      f'nbr_caught_gti = {nbr_caught_bti}\n'
                      f'nbr_caught_bti = {nbr_caught_bti}\n'
                      f'n_draws_bti    = {n_draws_bti}\n'
                      f'n_draws_gti    = {n_draws_gti}')
                x_pos = np.random.randint(low=10, high=cube.shape[0]-10)
                y_pos = np.random.randint(low=10, high=cube.shape[1]-10)
                while np.sum(cube[x_pos,y_pos]) < 1:
                    x_pos, y_pos = np.random.randint(10, cube.shape[0] - 10), np.random.randint(10, cube.shape[1] - 10)
                time_fraction = np.random.random()

                data_cube2.data = data_cube.data + create_fake_burst(data_cube=data_cube, x_pos=x_pos, y_pos=y_pos, time_peak_fraction=time_fraction, width_time=timebin / 2, amplitude=amplitude)
                cube_mu = calc_cube_mu(data_cube=data_cube2, wcs=img.wcs)
                cube_mu = np.where(cube_mu > range_mu[0], cube_mu, np.nan)  # Remove small expectation values outside of interpolation range
                cube_with_peak = data_cube2.data

                peaks = cube_with_peak>minimum_for_peak(np.where(cube_mu>0, cube_mu, np.nan))
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
    colors = cmr.take_cmap_colors('cmr.ocean', N=n_spacebins, cmap_range=(0,0.7))
    for (tab_result_gti,tab_result_bti), spacebin, tab_amplitude, color in zip(all_spacebin_results, spacebins, tab_all_amplitudes,colors):
        plt.plot(tab_amplitude, tab_result_gti, color=color, label=f'{int(spacebin)}"')
        plt.plot(tab_amplitude, tab_result_bti, color=color, ls="--")
        # plt.fill_between(tab_amplitude, np.array(tab_result)-np.sqrt(np.array(tab_result))/np.sqrt(n_draws),
        #                  np.array(tab_result) + np.sqrt(np.array(tab_result)) / np.sqrt(n_draws),
        #                  facecolor = color, alpha=0.5)
    plt.legend()
    plt.xlabel('Peak amplitude')
    plt.ylabel('Fraction of detected')
    plt.xscale('log')
    plt.savefig(data_plots / 'bayes_rate_estimate.png')
    plt.savefig(data_plots / 'bayes_rate_estimate.pdf')
    plt.show()

    plt.figure()
    for (tab_result_gti,tab_result_bti), spacebin, tab_amplitude, color in zip(all_spacebin_results, spacebins,tab_all_amplitudes, colors):
        plt.plot(tab_amplitude*timebin, tab_result_gti, color=color, label=f'{int(spacebin)}"')
        plt.plot(tab_amplitude*timebin, tab_result_bti, color=color, ls="--")
        # plt.fill_between(tab_amplitude*timebin, np.array(tab_result)-np.sqrt(np.array(tab_result))/np.sqrt(n_draws),
        #                  np.array(tab_result) + np.sqrt(np.array(tab_result)) / np.sqrt(n_draws),
        #                  facecolor = color, alpha=0.3)
    plt.legend()
    plt.xlabel('Peak count')
    plt.ylabel('Fraction of detected')
    plt.xscale('log')
    plt.show()


def bayes_successrate_timebinning2(tab_obsids=['0886121001']):
    size_arcsec = 20
    min_energy = 0.2
    max_energy = 12.0

    n_amplitude = 15  # Number of amplitude bins
    n_draws = 10
    timebins = [500, 1000, 5000]
    n_timebins = len(timebins)

    # Load precomputed Bayesian limits
    range_mu, minimum_for_peak, maximum_for_eclipse = load_precomputed_bayes_limits(threshold_sigma=3)

    all_res = []

    for obsidx, obsid in enumerate(tab_obsids):

        observation = Observation(obsid)
        observation.get_files()
        observation.get_events_overlapping_subsets()

        event_list = EventList.from_event_lists(observation.events_overlapping_subsets[0])
        event_list.read()

        img = observation.images[0]
        img.read(wcs_only=True)

        for timebin in timebins:
            p = Pipeline(obsid, time_interval=timebin)
            p.run()

            data_cube = p.data_cube
            rejected = data_cube.bti_bin_idx

            data_cube2 = data_cube.copy()
            tab_amplitude = np.geomspace(0.5 / timebin, 100 / timebin, n_amplitude)

            for amplitude in tab_amplitude:
                res = {
                    "obsid": obsid,
                    "timebin": timebin,
                    "amplitude": amplitude,
                    "draws_bti": 0,
                    "draws_gti": 0,
                    "caught_bti": 0,
                    "caught_gti": 0,
                }

                for trial in range(n_draws):
                    x_pos, y_pos = (np.random.randint(5, data_cube.shape[0] - 5),
                                    np.random.randint(5, data_cube.shape[1] - 5))
                    time_fraction = np.random.random()

                    # Inject synthetic burst
                    data_cube2.data = data_cube.data + create_fake_onebin_burst(
                        data_cube=data_cube,
                        x_pos=x_pos,
                        y_pos=y_pos,
                        time_peak_fraction=time_fraction,
                        amplitude=amplitude * timebin,
                    )

                    cube_mu = calc_cube_mu(data_cube=data_cube2, wcs=img.wcs)
                    cube_with_peak = data_cube2.data
                    cube_mu = np.where(cube_mu > 0, cube_mu, np.nan)
                    peaks = cube_with_peak > minimum_for_peak(cube_mu)

                    # Check detection based on GTI/BTI
                    if int(time_fraction * data_cube.shape[2]) in rejected:
                        res["draws_bti"] += 1
                        if np.max(peaks[x_pos, y_pos]) > 0:
                            res["caught_bti"] += 1
                    else:
                        res["draws_gti"] += 1
                        if np.max(peaks[x_pos, y_pos]) > 0:
                            res["caught_gti"] += 1

                # Save results for current (timebin, amplitude) configuration
                all_res.append(res)
    return all_res



def bayes_successrate_timebinning(tab_obsids=['0886121001']):
    size_arcsec = 20
    min_energy = 0.2
    max_energy = 12.0

    n_amplitude = 15 # Number of amplitude bins
    n_draws     = 10
    timebins    = [500, 1000, 5000] #np.geomspace(10,1000, n_timebins)
    n_timebins  = len(timebins)

    all_timebin_results = []
    tab_all_amplitudes = []
    range_mu, minimum_for_peak, maximum_for_eclipse = load_precomputed_bayes_limits(threshold_sigma=3)
    range_mu_3sig, minimum_for_peak_3sig, maximum_for_eclipse_3sig = load_precomputed_bayes_limits(threshold_sigma=3)
    tab_drawn_gti = np.zeros((len(tab_obsids), n_timebins, n_amplitude))
    tab_drawn_bti = np.zeros((len(tab_obsids), n_timebins, n_amplitude))
    tab_seen_gti  = np.zeros((len(tab_obsids), n_timebins, n_amplitude))
    tab_seen_bti  = np.zeros((len(tab_obsids), n_timebins, n_amplitude))

    pbar = tqdm(total=len(tab_obsids))
    for obsidx, obsid in enumerate(tab_obsids):
            observation = Observation(obsid)
            observation.get_files()
            observation.get_events_overlapping_subsets()

            event_list = EventList.from_event_lists(observation.events_overlapping_subsets[0])
            event_list.read()

            img = observation.images[0]
            img.read(wcs_only=True)

            for ind_timebin, timebin in enumerate(timebins):
                data_cube = DataCube(event_list, size_arcsec=size_arcsec, time_interval=timebin)
                rejected = data_cube.bti_bin_idx
                print('Creating copy of datacube (takes a second)...')
                data_cube2 = data_cube.copy()
                tab_amplitude = np.geomspace(0.5 / timebin, 100 / timebin, n_amplitude)
                tab_all_amplitudes.append(tab_amplitude)
                for ind_ampl, amplitude in enumerate(tab_amplitude):
                    nbr_caught_gti = 0
                    nbr_caught_bti = 0
                    n_draws_bti = 0
                    n_draws_gti = 0
                    for trial in range(n_draws):
                        print(f'amplitude={amplitude} '
                              f'trial={trial} / {n_draws} '
                              f'nbr_caught_gti = {nbr_caught_gti} '
                              f'nbr_caught_bti = {nbr_caught_bti} '
                              f'n_draws_bti = {n_draws_bti} '
                              f'n_draws_gti = {n_draws_gti}')

                        x_pos, y_pos = np.random.randint(5, data_cube.shape[0]-5), np.random.randint(5,data_cube.shape[1]-5)
                        time_fraction = np.random.random()
                        # cube_with_peak = cube + create_fake_burst(dl.data_cube, x_pos, y_pos, time_peak_fraction=time_fraction, width_time=timebin/2, amplitude=amplitude)
                        data_cube2.data = data_cube.data + create_fake_onebin_burst(data_cube=data_cube,
                                                                                    x_pos=x_pos,
                                                                                    y_pos=y_pos,
                                                                                    time_peak_fraction=time_fraction,
                                                                                    amplitude=amplitude*timebin)
                        cube_mu = calc_cube_mu(data_cube=data_cube2, wcs=img.wcs)
                        cube_with_peak = data_cube2.data
                        cube_mu = np.where(cube_mu > 0, cube_mu, np.nan)
                        peaks = cube_with_peak > minimum_for_peak(cube_mu)

                        # peaks_3sig = cube_with_peak>minimum_for_peak_3sig(cube_mu)

                        if int(time_fraction*data_cube.shape[2]) in rejected:
                            n_draws_bti+=1
                            if np.max(peaks[x_pos, y_pos]) > 0:
                                nbr_caught_bti+=1
                        else:
                            n_draws_gti+=1
                            if np.max(peaks[x_pos, y_pos]) > 0:
                                nbr_caught_gti+=1
                    tab_drawn_bti[obsidx, ind_timebin, ind_ampl]=n_draws_bti
                    tab_drawn_gti[obsidx, ind_timebin, ind_ampl]=n_draws_gti
                    tab_seen_bti[obsidx, ind_timebin, ind_ampl]=nbr_caught_bti
                    tab_seen_gti[obsidx, ind_timebin, ind_ampl]=nbr_caught_gti

    # print(tab_seen_gti[0,0],tab_seen_gti[1,0],tab_seen_gti[2,0])

    colors = cmr.take_cmap_colors('cmr.ocean', N=n_timebins, cmap_range=(0,0.7))
    legend_plots = []
    legend_labels = []
    fig, (ax1,ax2) = plt.subplots(2,1, figsize=(5,10))
    for index, timebin, tab_amplitude, color in zip(range(len(timebins)), timebins, tab_all_amplitudes, colors):
        gti_ratios =  np.array(tab_seen_gti[:,index,:])/np.array(tab_drawn_gti[:,index,:])
        bti_ratios =  np.array(tab_seen_bti[:,index,:])/np.array(tab_drawn_bti[:,index,:])
        # total_ratios = np.sum(np.array(tab_seen_bti[:,index,:])+np.array(tab_seen_gti[:,index,:]), axis=0)/\
        #                 np.sum(np.array(tab_drawn_bti[:,index,:])+np.array(tab_drawn_gti[:,index,:]), axis=0)
        # err_total_ratios = np.sqrt(np.sum(np.array(tab_seen_bti[:,index,:])+np.array(tab_seen_gti[:,index,:]), axis=0))/\
        #                 np.sum(np.array(tab_drawn_bti[:,index,:])+np.array(tab_drawn_gti[:,index,:]), axis=0)
        total_ratios = np.nanmean((np.array(tab_seen_bti[:, index, :]) + np.array(tab_seen_gti[:, index, :]))/ \
                            (np.array(tab_drawn_bti[:,index,:])+np.array(tab_drawn_gti[:,index,:])), axis=0)
        err_total_ratios = np.nanstd((np.array(tab_seen_bti[:, index, :]) + np.array(tab_seen_gti[:, index, :]))/ \
                            (np.array(tab_drawn_bti[:,index,:])+np.array(tab_drawn_gti[:,index,:])), axis=0)
        print(total_ratios)
        print(err_total_ratios)
        gti_percentiles = np.nanpercentile(gti_ratios, (0.16,0.5,0.84), axis=0)
        bti_percentiles = np.nanpercentile(bti_ratios, (0.16,0.5,0.84), axis=0)
        total_percentiles = np.nanpercentile(total_ratios, (0.16,0.5,0.84), axis=0)

        # 1.5 comes from an unknown correction factor, after checking the count difference between true cube and
        # the one with a synthetic peak. Most likely from geometric PSF configuration ?
        p1 = ax1.plot(tab_amplitude * 1.5,total_ratios, color=color, label=f'{int(timebin)}s')
        ax1.fill_between(tab_amplitude * 1.5, total_ratios-err_total_ratios, total_ratios+err_total_ratios, color=color, alpha=0.2)
        p2 = plt.fill(np.nan, np.nan, color=color, alpha=0.2)
        legend_plots.append((p2[0], p1[0]))
        legend_labels.append(f'{int(timebin)}s')

        # ax1.plot(tab_amplitude * 1.5,np.sum(np.array(tab_seen_gti[:,index,:]), axis=0)/np.sum(np.array(tab_drawn_gti[:,index,:]),axis=0), color=color, label=f'{int(timebin)}s', ls="--")
        # ax1.plot(tab_amplitude * 1.5,np.sum(np.array(tab_seen_bti[:,index,:]), axis=0)/np.sum(np.array(tab_drawn_bti[:,index,:]),axis=0), color=color, label=f'{int(timebin)}s', ls=":")
        ax2.plot(tab_amplitude*timebin * 1.5,total_ratios, color=color, label=f'{int(timebin)}s')
        ax2.fill_between(tab_amplitude*timebin * 1.5, total_ratios-err_total_ratios, total_ratios+err_total_ratios, color=color, alpha=0.2)
        # ax2.plot(tab_amplitude*timebin * 1.5,
        #          np.sum(np.array(tab_seen_gti[:, index, :]), axis=0) / np.sum(np.array(tab_drawn_gti[:, index, :]),
        #                                                                       axis=0), color=color,
        #          label=f'{int(timebin)}s', ls="--")
        # ax2.plot(tab_amplitude*timebin * 1.5,
        #          np.sum(np.array(tab_seen_bti[:, index, :]), axis=0) / np.sum(np.array(tab_drawn_bti[:, index, :]),
        #                                                                       axis=0), color=color,
        #          label=f'{int(timebin)}s', ls=":")

        # plt.plot(tab_amplitude, gti_percentiles[1], color=color, label=f'{int(timebin)}s')
        # plt.fill_between(tab_amplitude, gti_percentiles[0], gti_percentiles[2], color=color, alpha=0.2)
        # plt.plot(tab_amplitude, bti_percentiles[1], color=color, ls="--")
        # plt.fill_between(tab_amplitude, bti_percentiles[0], bti_percentiles[2], color=color, alpha=0.2)

        # plt.fill_between(tab_amplitude, np.array(tab_result)-np.sqrt(np.array(tab_result))/np.sqrt(n_draws),
        #                  np.array(tab_result) + np.sqrt(np.array(tab_result)) / np.sqrt(n_draws),
        #                  facecolor = color, alpha=0.5)
    ax1.legend(legend_plots, legend_labels)
    ax2.legend(legend_plots, legend_labels)
    ax1.set_ylim((0,1))
    ax2.set_ylim((0,1))
    ax1.set_xlabel(r'Peak amplitude (counts s$^{-1}$)')
    ax1.set_ylabel('Fraction of detected peaks')
    ax2.set_xlabel('Peak amplitude (counts)')
    ax2.set_ylabel('Fraction of detected peaks')
    ax1.set_xscale('log')
    ax2.set_xscale('log')
    plt.tight_layout()
    plt.savefig(data_plots / 'bayes_successrate_timebinning1.png')
    plt.savefig(data_plots / 'bayes_successrate_timebinning1.pdf')
    plt.show()

    plt.figure()
    plots = []
    for index, timebin, tab_amplitude, color in zip(range(len(timebins)), timebins, tab_all_amplitudes, colors):
        gti_ratios = np.array(tab_seen_gti[:, index, :]) / np.array(tab_drawn_gti[:, index, :])
        bti_ratios = np.array(tab_seen_bti[:, index, :]) / np.array(tab_drawn_bti[:, index, :])
        total_ratios = np.nanmean((np.array(tab_seen_bti[:, index, :]) + np.array(tab_seen_gti[:, index, :])) / \
                                  (np.array(tab_drawn_bti[:, index, :]) + np.array(tab_drawn_gti[:, index, :])), axis=0)
        err_total_ratios = np.nanstd((np.array(tab_seen_bti[:, index, :]) + np.array(tab_seen_gti[:, index, :])) / \
                                     (np.array(tab_drawn_bti[:, index, :]) + np.array(tab_drawn_gti[:, index, :])),
                                     axis=0)
        # 1.5 comes from an unknown correction factor, after checking the count difference between true cube and
        # the one with a synthetic peak. Most likely from geometric PSF configuration ?

        a,=plt.plot(tab_amplitude * 1.5,np.sum(np.array(tab_seen_gti[:,index,:])+np.array(tab_seen_bti[:,index,:]), axis=0)/np.sum(np.array(tab_drawn_gti[:,index,:])+np.array(tab_drawn_bti[:,index,:]),axis=0), color=color, label=f'{int(timebin)}s',lw=3)
        b,=plt.plot(tab_amplitude * 1.5,np.sum(np.array(tab_seen_gti[:,index,:]), axis=0)/np.sum(np.array(tab_drawn_gti[:,index,:]),axis=0), color=color, ls="--",lw=2)
        c,=plt.plot(tab_amplitude * 1.5,np.sum(np.array(tab_seen_bti[:,index,:]), axis=0)/np.sum(np.array(tab_drawn_bti[:,index,:]),axis=0), color=color, ls=":",lw=2)
        plots.append(a)
    plt.xlabel(r'Peak amplitude (counts s$^{-1}$)')
    from matplotlib.lines import Line2D
    c=Line2D([], [], linestyle='')
    d,=plt.plot(np.nan, np.nan, color='gray', label='Total',lw=2)
    e,=plt.plot(np.nan, np.nan, color='gray',  ls="--", label='GTI',lw=2)
    f,=plt.plot(np.nan, np.nan, color='gray',  ls=":", label='BTI',lw=2)
    plots+=[c,d,e,f]
    plt.xscale("log")
    plt.legend(plots,  ['5s', '50s', '200s', '', 'Total','GTI','BTI'], labelspacing=.3)

    # plt.figure()
    # for index, timebin, tab_amplitude, color in zip(range(len(timebins)), timebins, tab_all_amplitudes, colors):
    #     gti_ratios =  np.array(tab_seen_gti[:,index,:])/np.array(tab_drawn_gti[:,index,:])
    #     bti_ratios =  np.array(tab_seen_bti[:,index,:])/np.array(tab_drawn_bti[:,index,:])
    #     total_ratios = np.sum(np.array(tab_seen_bti[:,index,:])+np.array(tab_seen_gti[:,index,:]), axis=0)/\
    #                     np.sum(np.array(tab_drawn_bti[:,index,:])+np.array(tab_drawn_gti[:,index,:]), axis=0)
    #     err_total_ratios = np.sqrt(np.sum(np.array(tab_seen_bti[:,index,:])+np.array(tab_seen_gti[:,index,:]), axis=0))/\
    #                     np.sum(np.array(tab_drawn_bti[:,index,:])+np.array(tab_drawn_gti[:,index,:]), axis=0)
    #     gti_percentiles = np.nanpercentile(gti_ratios, (0.16,0.5,0.84), axis=0)
    #     bti_percentiles = np.nanpercentile(bti_ratios, (0.16,0.5,0.84), axis=0)
    #     total_percentiles = np.nanpercentile(total_ratios, (0.16,0.5,0.84), axis=0)
    #
    #     # plt.plot(tab_amplitude*timebin, total_percentiles[1], color=color, label=f'{int(timebin)}s')
    #     # plt.fill_between(tab_amplitude*timebin, total_ratios-err_total_ratios, total_ratios+err_total_ratios, color=color, alpha=0.2)
    #     plt.plot(tab_amplitude*timebin,total_ratios, color=color, label=f'{int(timebin)}s')
    #     plt.fill_between(tab_amplitude*timebin, total_ratios-err_total_ratios, total_ratios+err_total_ratios, color=color, alpha=0.2)
    #     # plt.plot(tab_amplitude*timebin, gti_percentiles[1], color=color, label=f'{int(timebin)}s')
    #     # plt.fill_between(tab_amplitude*timebin, gti_percentiles[0], gti_percentiles[2], color=color, alpha=0.2)
    #     # plt.plot(tab_amplitude*timebin, bti_percentiles[1], color=color, ls="--")
    #     # plt.fill_between(tab_amplitude*timebin, bti_percentiles[0], bti_percentiles[2], color=color, alpha=0.2)
    #
    # plt.legend()
    # plt.ylim((0,1))
    # plt.xlabel('Peak amplitude (counts)')
    # plt.ylabel('Fraction of detected')
    # plt.xscale('log')
    # plt.savefig(data_plots / 'bayes_successrate_timebinning2.png')
    # plt.savefig(data_plots / 'bayes_successrate_timebinning2.pdf')
    # plt.show()
    return tab_all_amplitudes, tab_seen_bti, tab_seen_gti, tab_drawn_bti , tab_drawn_gti

def bayes_false_positive_rate_timebinning(tab_obsids=['0886121001']):
    from tqdm import tqdm
    size_arcsec = 20
    min_energy = 0.2
    max_energy = 12.0

    n_timebins = 3
    n_draws = 500
    timebins = (5,50,200)#np.geomspace(10,1000, n_timebins)
    range_mu, minimum_for_peak, maximum_for_eclipse = load_precomputed_bayes_limits(threshold_sigma=5)
    range_mu_3sig, minimum_for_peak_3sig, maximum_for_eclipse_3sig = load_precomputed_bayes_limits(threshold_sigma=3)
    tab_wrong_positives = np.zeros((len(tab_obsids), n_timebins))
    tab_wrong_positives3sig = np.zeros((len(tab_obsids), n_timebins))

    nbr_pixels = [0,0,0]

    pbar=tqdm(total=len(tab_obsids))
    for obsidx, obsid in enumerate(tab_obsids):
        try:
            observation = Observation(obsid)
            observation.get_files()
            observation.get_events_overlapping_subsets()

            event_list = EventList.from_event_lists(observation.events_overlapping_subsets[0])
            event_list.read()

            img = observation.images[0]
            img.read(wcs_only=True)

            for ind_timebin, timebin in enumerate(timebins):
                data_cube = DataCube(event_list, size_arcsec=size_arcsec, time_interval=timebin)
                cube_mu = calc_cube_mu(data_cube=data_cube, wcs=img.wcs)
                nbr_false_positives = 0
                nbr_false_positives_3sig = 0
                # for _ in tqdm(range(n_draws)):
                for _ in range(int(n_draws/3)):
                    false_cube = np.random.poisson(np.expand_dims(np.nan_to_num(cube_mu),-1),cube_mu.shape+(3,))
                    wrong_peaks = np.sum(false_cube > np.expand_dims(minimum_for_peak(cube_mu),-1))
                    nbr_false_positives+=wrong_peaks
                    wrong_peaks3sig = np.sum(false_cube > np.expand_dims(minimum_for_peak_3sig(cube_mu),-1))
                    nbr_false_positives_3sig+=wrong_peaks3sig
                    nbr_pixels[ind_timebin]+=np.sum(false_cube.shape)
                tab_wrong_positives[obsidx, ind_timebin] = nbr_false_positives
                tab_wrong_positives3sig[obsidx, ind_timebin] = nbr_false_positives_3sig


        except: #To catch observations with no EPIC files
            pass
        print(f'{obsidx+1}/{len(tab_obsids)}')
        pbar.update(1)
    pbar.close()
    for timebinidx, timebin in enumerate(timebins):
        print(f"5sig false positive count for {timebin}s binning: {np.nanmean(tab_wrong_positives[:,timebinidx])}, \
        std {np.nanstd(tab_wrong_positives[:,timebinidx])}, for {nbr_pixels[timebinidx]} pixels")
        print(f'Actual counts: {np.nansum(tab_wrong_positives[:,timebinidx])}')
        print(f'Per pixel rate: {np.nansum(tab_wrong_positives[:,timebinidx])/nbr_pixels[timebinidx]}')
        print(f"3sig false positive count for {timebin}s binning: {np.nanmean(tab_wrong_positives3sig[:,timebinidx])}, \
                std {np.nanstd(tab_wrong_positives3sig[:,timebinidx])}, for {nbr_pixels[timebinidx]} pixels")
        print(f'Actual counts: {np.nansum(tab_wrong_positives3sig[:,timebinidx])}')
        print(f'Per pixel rate: {np.nansum(tab_wrong_positives3sig[:,timebinidx])/nbr_pixels[timebinidx]}')
    return np.nansum(tab_wrong_positives, axis=0), np.nanstd(tab_wrong_positives, axis=0), np.nansum(tab_wrong_positives3sig, axis=0), np.nanstd(tab_wrong_positives3sig, axis=0)


def bayes_eclipse_successrate_depth(base_rate=10., obsids=['0765080801'], time_interval=1000):
    size_arcsec   = 20
    gti_only      = False
    gti_threshold = 0.5
    min_energy    = 0.2
    max_energy    = 12.0

    tab_eclipse_amplitudes = np.linspace(start=0, stop=1, num=20)
    nbr_draws = 5

    tab_result_3sig = []
    tab_result_5sig = []
    for obsid in obsids:

        observation = Observation(obsid)
        observation.get_files()

        event_list = observation.events_processed_pn[0]
        event_list.read()

        img = observation.images[0]
        img.read(wcs_only=True)

        observation.get_events_overlapping_subsets()
        for ind_exp, subset_overlapping_exposures in enumerate(observation.events_overlapping_subsets):
            event_list = EventList.from_event_lists(subset_overlapping_exposures)
            data_cube = DataCube(event_list, size_arcsec=size_arcsec, time_interval=time_interval)
            cube = data_cube.data
            rejected = data_cube.bti_bin_idx
            print('Creating copy of datacube (takes a second)...')
            data_cube2 = data_cube.copy()

            for amplitude in tqdm(tab_eclipse_amplitudes):
                caught_at_amplitude_3sig = 0
                caught_at_amplitude_5sig = 0
                tab_time_peak_fraction = np.random.random(nbr_draws)
                tab_x_pos, tab_y_pos = np.random.randint(5, cube.shape[0] - 5, nbr_draws), np.random.randint(5, cube.shape[1] - 5, nbr_draws)

                data_cube2.data = cube + create_multiple_fake_eclipses(data_cube, tab_x_pos, tab_y_pos, tab_time_peak_fraction, [10] * nbr_draws, [amplitude * base_rate] * nbr_draws, [base_rate] * nbr_draws)
                cube_mu = calc_cube_mu(data_cube=data_cube2, wcs=img.wcs)
                cube_with_eclipse = data_cube2.data
                peaks_3, eclipses_3 = get_cube_masks_peak_and_eclipse(cube_with_eclipse, cube_mu, threshold_sigma=3)
                peaks_5, eclipses_5 = get_cube_masks_peak_and_eclipse(cube_with_eclipse, cube_mu, threshold_sigma=5)
                for x_pos, y_pos in zip(tab_x_pos, tab_y_pos):
                    if np.sum(eclipses_3[x_pos, y_pos])>0:
                        caught_at_amplitude_3sig+=1
                    if np.sum(eclipses_5[x_pos, y_pos])>0:
                        caught_at_amplitude_5sig+=1
                tab_result_3sig.append(caught_at_amplitude_3sig/nbr_draws)
                tab_result_5sig.append(caught_at_amplitude_5sig/nbr_draws)
    plt.figure()
    plt.plot(tab_eclipse_amplitudes, tab_result_3sig, label=r'$3\sigma$')
    plt.plot(tab_eclipse_amplitudes, tab_result_5sig, label=r'$5\sigma$')
    plt.legend()
    plt.xlabel("Relative amplitude of eclipse")
    plt.ylabel("Fraction of detected eclipses")
    plt.savefig(data_plots / 'bayes_eclipse_successrate_depth.png')
    plt.savefig(data_plots / 'bayes_eclipse_successrate_depth.pdf')
    plt.show()

def plot_B_peak():
    """
    Plot the peak Bayes factor for different observed (n) counts as a function of expectation (mu).
    Also plot the 3 and 5 sigma threshold values.
    """
    n_lines_to_plot = 20  # 1 line is drawn for each value of n from 0 to n-1
    colors = plt.cm.winter(np.linspace(start=0, stop=1, num=n_lines_to_plot))

    B_peak_3sig, B_eclipse_3sig = get_bayes_thresholds(3)
    B_peak_5sig, B_eclipse_5sig = get_bayes_thresholds(5)

    mu_lo, mu_hi = 1e-3, 50
    mus = np.geomspace(mu_lo, mu_hi, 1000)

    plt.figure(figsize=(3.5, 3.5))
    for n in range(n_lines_to_plot):
        label = None
        if (n == 0) or (n == n_lines_to_plot-1):  # Label first and last line
            label = f'n={n}'
        plt.plot(mus, B_peak_log(n=n, mu=mus), color=colors[n], label=label)

    plt.axhline(B_peak_3sig, color='red', label=rf'3 $\sigma$ (B={B_peak_3sig:.2f})')
    plt.axhline(B_peak_5sig, color='black', label=rf'5 $\sigma$ (B={B_peak_5sig:.2f})')
    plt.title(f'Peak Bayes factor for n=0-{n_lines_to_plot}')
    plt.xlabel(r'Expected Value $\mu$')
    plt.ylabel(r'$log_{10}$($B_{peak}$)')
    plt.xscale('log')
    plt.tight_layout()
    plt.ylim(0)
    plt.xlim(mu_lo, mu_hi)
    plt.legend()
    plt.savefig(data_plots / 'B_peak.png')
    plt.savefig(data_plots / 'B_peak.pdf')
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

    plt.figure(figsize=(3.5, 3.5))
    for n in range(n_lines_to_plot):
        label = None
        if (n == 0) or (n == n_lines_to_plot-1):  # Label first and last line
            label = f'n={n}'
        plt.plot(mus, B_eclipse_log(n=n, mu=mus), color=colors[n], label=label)

    plt.axhline(B_eclipse_3sig, color='red', label=rf'3 $\sigma$ (B={B_peak_3sig:.2f})')
    plt.axhline(B_eclipse_5sig, color='black', label=rf'5 $\sigma$ (B={B_peak_5sig:.2f})')
    plt.title(f'Eclipse Bayes factor for n=0-{n_lines_to_plot}')
    plt.xlabel(r'Expected Value $\mu$')
    plt.ylabel(r'$log_{10}$($B_{eclipse}$)')
    # plt.xscale('log')
    plt.tight_layout()
    plt.xlim(mu_lo, mu_hi)
    plt.legend()
    plt.savefig(data_plots / 'B_eclipse.png')
    plt.savefig(data_plots / 'B_eclipse.pdf')
    plt.show()

def plot_B_values_3d():
    n_ = np.arange(10)
    mu_ = np.geomspace(1e-3, 5, 1000)
    N, MU = np.meshgrid(n_, mu_)
    B = B_peak_log(n=N, mu=MU)
    B2 = B_eclipse_log(n=N, mu=MU)


    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10,7))

    surf1 = ax.plot_surface(N, MU, B, cmap='hot', linewidth=0, antialiased=False)
    cbar1 = fig.colorbar(surf1, aspect=3, shrink=0.5)
    cbar1.set_label(r'$\mathrm{log}(B_{peak})$')

    surf2 = ax.plot_surface(N, MU, B2, cmap='cool', linewidth=0, antialiased=False)
    cbar2 = fig.colorbar(surf2, aspect=3, shrink=0.5)
    cbar2.set_label(r'$\mathrm{log}(B_{eclipse})$')

    ax.axhline(20, color='red')

    ax.set_xlabel('Observed (n)')
    ax.set_ylabel(r'Expected ($\mu$)')
    ax.set_zlabel('Bayes Factor (B)')
    plt.tight_layout()
    plt.savefig(data_plots / 'B_values_3d.png')
    plt.savefig(data_plots / 'B_values_3d.pdf')
    plt.show()


def plot_B_factor_vs_sigma():
    plt.figure(figsize=(3.5, 3.5))

    xmin, xmax = 1.61, 42

    sigmas_peak = []
    sigmas_eclipse = []
    Bs = np.linspace(xmin, xmax, 500)
    for B in Bs:
        sigmas_peak.append(sigma_equivalent_B_peak(B))
        sigmas_eclipse.append(sigma_equivalent_B_eclipse(B))

    plt.plot(Bs, sigmas_peak, label=r'$\mathrm{B_{peak}}$')
    plt.plot(Bs, sigmas_eclipse, label=r'$\mathrm{B_{Eclipse}}$')
    plt.xlabel(r'Bayes Factor ($\mathrm{log_{10} B})$')
    plt.ylabel(r'Sigma Equivalent ($\sigma$)')
    plt.xlim(xmin, xmax)
    plt.ylim(0, 10)
    plt.legend()
    plt.savefig(data_plots / 'B_factor_vs_sigma.png')
    plt.savefig(data_plots / 'B_factor_vs_sigma.pdf')
    plt.show()


def plot_bayes_limits():
    pbl_3 = PrecomputeBayesLimits(3)
    pbl_5 = PrecomputeBayesLimits(5)

    range_mu  = pbl_3.range_mu

    tab_npeak_3sig = pbl_3.n_peak_threshold(range_mu)
    tab_neclipse_3sig = pbl_3.n_eclipse_threshold(range_mu)

    tab_npeak = pbl_5.n_peak_threshold(range_mu)
    tab_neclipse = pbl_5.n_eclipse_threshold(range_mu)


    plt.figure(figsize=(4, 4))

    plt.plot(range_mu, tab_npeak, ls=':', c='k', label=fr'5$\sigma$', lw=1.0)
    plt.plot(range_mu, tab_neclipse, ls=':', c='k', lw=1.0)

    plt.plot(range_mu, tab_npeak_3sig, ls='--', c='k', label=fr'3$\sigma$', lw=1.0)
    plt.plot(range_mu, tab_neclipse_3sig, ls='--', c='k', lw=1.0)

    plt.fill_between(range_mu, tab_npeak, 1e6, alpha=0.5, color='steelblue', label='Detection Region')
    plt.fill_between(range_mu, tab_npeak_3sig, 1e6, alpha=0.3, color='steelblue')
    plt.fill_between(range_mu, 0, tab_neclipse, alpha=0.5, color='steelblue')
    plt.fill_between(range_mu, 0, tab_neclipse_3sig, alpha=0.3, color='steelblue')


    plt.plot(range_mu, range_mu, label=r'$N=\mu$', color='black')
    #plt.fill_between(range_mu, range_mu-5*np.sqrt(range_mu), range_mu+5*np.sqrt(range_mu), alpha=0.3, label=fr'Naive $5 \sigma$', color='grey')
    #plt.fill_between(range_mu, range_mu-3*np.sqrt(range_mu), range_mu+3*np.sqrt(range_mu), alpha=0.5, label=fr'Naive $3 \sigma$', color='grey')
    plt.yscale('log')
    plt.xscale('log')
    plt.title(r'$B_{peak} = \frac{Q(N+1, \mu)}{e^{-\mu} \mu^{N} / N!} \ \  B_{eclipse} = \frac{P(N+1, \mu)}{e^{-\mu} \mu^{N} / N!}$')
    plt.xlabel(fr'Expected Counts $\mu$')
    plt.ylabel(fr'Observed Counts $N$')
    plt.xlim(min(range_mu), max(range_mu))
    plt.ylim(min(range_mu), max(range_mu))
    plt.text(0.05, 25, s='Significant Peaks', )
    plt.text(35, 0.1, s='Significant\nEclipses')
    #plt.yticks([1, 10, 100, 300], labels=[1, 10, 100, 300])
    plt.yticks([0.01, 0.1, 1, 10, 100, 300], labels=[0.01, 0.1, 1, 10, 100, 300])
    plt.xticks([0.1, 1, 10, 100, 300], labels=[ 0.1, 1, 10, 100, 300])
    plt.xlim(0.01, 300)
    plt.ylim(0.01, 300)
    plt.legend(loc='upper left', fontsize=10, ncol=2, columnspacing=0.8)
    #plt.tight_layout()
    plt.savefig(data_plots / f'bayesfactorlimits_3_5.pdf')
    plt.savefig(data_plots / f'bayesfactorlimits_3_5.png')
    plt.show()

if __name__ == "__main__":
    from exod.utils.plotting import use_scienceplots
    from exod.utils.path import data_processed
    #use_scienceplots()
    #plot_bayes_limits()
    #plot_B_peak()
    #plot_B_eclipse()
    #plot_mu_vs_bayes_factor_for_diff_n()
    #plot_B_factor_vs_sigma()
    #plot_B_values_3d()
    #check_estimate_success()
    #check_eclipse_estimate_success()
    # test_bayes_on_false_cube(size=100)
    #bayes_rate_estimate()
    #bayes_successrate_spacebinning()
    bayes_successrate_timebinning2()
    #bayes_eclipse_successrate_depth()
    # print(bayes_false_positive_rate_timebinning(tab_obsids=os.listdir(data_processed)[:50]))


