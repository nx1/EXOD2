import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from exod.pre_processing.data_loader import DataLoader
from exod.xmm.observation import Observation
from exod.xmm.event_list import EventList
from exod.processing.background_inference import calc_cube_mu
from exod.processing.bayesian_computations import load_precomputed_bayes_limits, get_cube_masks_peak_and_eclipse, \
    B_peak_log, B_eclipse_log, get_bayes_thresholds
from exod.post_processing.estimate_variability_properties import peak_count_estimate, eclipse_count_estimate
from exod.utils.synthetic_data import create_fake_burst, create_multiple_fake_eclipses
import cmasher as cmr


def check_estimate_success():
    """
    If I remember correctly, it is to assess if the count-rate estimation worked,
    independently of the value of the quiescent state mu. So, for a range of mu,
    I add a peak of three different amplitudes, I generate poisson realisations of mu+peak (the crosses),
    and see the value we can estimate for the peak (the envelopes). You have the residuals at the bottom
    It might be an earlier version, and not work anymore. It was just to check the success of the rate estimate function
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
    plt.show()


def plot_some_n_bayes():
    range_n = np.arange(10)
    range_mu = np.geomspace(1e-3, 1e3, 500)
    plt.figure()
    colors = cmr.take_cmap_colors('cmr.ocean',N=len(range_n),cmap_range=(0,0.7))
    for c,n in zip(colors,range_n):
        bayes_peak = [B_peak_log(n=n, mu=mu) for mu in range_mu]#[bayes_factor_peak(mu, n) for mu in range_mu]
        bayes_eclipse = [B_eclipse_log(n=n, mu=mu) for mu in range_mu]
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
    peaks = B_peak_log(estimated, cube) > 5 #cube_n>minimum_for_peak(estimated)
    eclipse = B_eclipse_log(estimated, cube) > 5#cube_n<maximum_for_eclipse(estimated)
    print(np.sum(peaks), np.sum(eclipse))
    return np.sum(peaks), np.sum(eclipse)


def test_on_data(cube, expected, threshold):
    minimum_for_peak, maximum_for_eclipse = load_precomputed_bayes_limits(threshold_sigma=threshold)
    peaks = cube>minimum_for_peak(np.where(expected>0, expected, np.nan))
    eclipse =  cube<maximum_for_eclipse(np.where(expected>0, expected, np.nan))
    return peaks, eclipse


def accepted_n_values():
    """
    Testing function, showing the accepted counts for a range of mu.
    Similar to the pre-compute function
    """
    range_n = np.arange(10)
    range_mu = np.geomspace(1e-3, 1e3, 5000)
    tab_npeak, tab_neclipse = [],[]
    for mu in tqdm(range_mu):
        range_n_peak = np.arange(max(10*mu, 100))
        B_peak = B_peak_log(n=range_n_peak, mu=mu)
        tab_npeak.append(range_n_peak[B_peak>5.94][0])

        range_n_eclipse = np.arange(2*int(mu)+1)
        B_eclipse = B_eclipse_log(n=range_n_eclipse, mu=mu)
        tab_neclipse.append(range_n_eclipse[B_eclipse<5.70][0])
    plt.figure()
    plt.plot(range_mu, range_mu)
    plt.plot(range_mu, range_mu-6*np.sqrt(range_mu), ls='--',c='k')
    plt.plot(range_mu, range_mu+6*np.sqrt(range_mu), ls='--',c='k')

    plt.fill_between(range_mu,tab_neclipse, tab_npeak,alpha=0.5)
    plt.loglog()
    plt.xlabel(r"$\mu$")
    plt.ylabel("Range of accepted # photons")
    plt.show()
# accepted_n_values()

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
    dl = DataLoader(event_list=event_list, time_interval=timebin, size_arcsec=size_arcsec, gti_only=False,
                    min_energy=min_energy, max_energy=max_energy, gti_threshold=gti_threshold, remove_partial_ccd_frames=False)
    dl.run()


    # We create two copies of DataCube()
    # The first (dl.datacube) will be used to create the original + synthetic data
    # we then set the data of the copied cube to this. This is extremely messy but it works for now.
    data_cube = dl.data_cube
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
        dl = DataLoader(event_list=event_list, time_interval=timebin, size_arcsec=size_arcsec, gti_only=False,
                        min_energy=min_energy, max_energy=max_energy, gti_threshold=gti_threshold,
                        remove_partial_ccd_frames=False)
        dl.run()

        data_cube = dl.data_cube
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
                print(f'size_arcsec = {size_arcsec}'
                      f'amplitude = {amplitude} '
                      f'trial={trial}/{n_draws} '
                      f'nbr_caught_gti = {nbr_caught_bti} '
                      f'nbr_caught_bti = {nbr_caught_bti} '
                      f'n_draws_bti = {n_draws_bti} '
                      f'n_draws_gti = {n_draws_gti}')
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

    n_timebins = 2
    n_amplitude = 2
    n_draws = 5
    timebins = np.geomspace(10,1000, n_timebins)
    all_timebin_results = []
    range_mu, minimum_for_peak, maximum_for_eclipse = load_precomputed_bayes_limits(threshold_sigma=5)
    tab_all_amplitudes = []
    for timebin in timebins:
        dl = DataLoader(event_list=event_list, time_interval=timebin, size_arcsec=size_arcsec, gti_only=False,
                        min_energy=min_energy, max_energy=max_energy, gti_threshold=gti_threshold,
                        remove_partial_ccd_frames=False)
        dl.run()

        data_cube = dl.data_cube
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
                print(f'amplitude={amplitude} '
                      f'trial={trial} / {n_draws} '
                      f'nbr_caught_gti = {nbr_caught_gti}'
                      f'nbr_caught_bti = {nbr_caught_bti} '
                      f'n_draws_bti = {n_draws_bti} '
                      f'n_draws_gti = {n_draws_gti}')

                x_pos, y_pos = np.random.randint(5,cube.shape[0]-5),np.random.randint(5,cube.shape[1]-5)
                time_fraction = np.random.random()
                cube_with_peak = cube + create_fake_burst(dl.data_cube, x_pos, y_pos, time_peak_fraction=time_fraction, width_time=timebin/2, amplitude=amplitude)
                data_cube2.data = data_cube.data + create_fake_burst(data_cube=data_cube, x_pos=x_pos, y_pos=y_pos, time_peak_fraction=time_fraction, width_time=timebin / 2, amplitude=amplitude)
                cube_mu = calc_cube_mu(data_cube=data_cube2, wcs=img.wcs)
                cube_with_peak = data_cube2.data
                cube_mu = np.where(cube_mu > 0, cube_mu, np.nan)
                peaks = cube_with_peak > minimum_for_peak(cube_mu)
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



    colors = cmr.take_cmap_colors('cmr.ocean', N=n_timebins, cmap_range=(0,0.7))
    plt.figure()
    for (tab_result_gti,tab_result_bti), timebin, tab_amplitude, color in zip(all_timebin_results, timebins, tab_all_amplitudes, colors):
        plt.plot(tab_amplitude, tab_result_gti, color=color, label=f'{int(timebin)}s')
        plt.plot(tab_amplitude, tab_result_bti, color=color, ls="--")
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
        plt.plot(tab_amplitude*timebin, tab_result_gti, color=color, label=f'{int(timebin)}s')
        plt.plot(tab_amplitude*timebin, tab_result_bti, color=color, ls="--")
        # plt.fill_between(tab_amplitude*timebin, np.array(tab_result)-np.sqrt(np.array(tab_result))/np.sqrt(n_draws),
        #                  np.array(tab_result) + np.sqrt(np.array(tab_result)) / np.sqrt(n_draws),
        #                  facecolor = color, alpha=0.3)
    plt.legend()
    plt.xlabel('Peak count')
    plt.ylabel('Fraction of detected')
    plt.xscale('log')
    plt.show()

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
            dl = DataLoader(event_list=event_list, time_interval=time_interval, size_arcsec=size_arcsec,
                            gti_only=gti_only, min_energy=min_energy, max_energy=max_energy,
                            gti_threshold=gti_threshold)
            dl.run()

            data_cube = dl.data_cube
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

    plt.figure(figsize=(5, 5))
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

    plt.figure(figsize=(5, 5))
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
    plt.show()

if __name__ == "__main__":
    plot_B_peak()
    plot_B_eclipse()
    plot_B_values_3d()
    check_estimate_success()
    check_eclipse_estimate_success()
    plot_some_n_bayes()
    test_bayes_on_false_cube(size=100)
    accepted_n_values()
    bayes_rate_estimate()
    bayes_successrate_spacebinning()
    bayes_successrate_timebinning()
    bayes_eclipse_successrate_depth()
