import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os
from tqdm import tqdm
from exod.pre_processing.data_loader import DataLoader
from exod.xmm.observation import Observation
from exod.processing.experimental.template_based_background_inference import compute_expected_cube_using_templates,compute_likelihood_variability
from exod.processing.experimental.bayesian import load_precomputed_bayes_limits
from exod.utils.synthetic_data import create_fake_burst
from scipy.stats import poisson
import cmasher as cmr




def bayes_successrate_timebinning(obsid='0886121001'):
    size_arcsec = 20
    gti_only = False
    gti_threshold = 0.5
    min_energy = 0.2
    max_energy = 2.0

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

    minimum_for_peak, maximum_for_eclipse = load_precomputed_bayes_limits(threshold=3)

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


