#TODO: KS and Chi2 tests on lightcurves extracted from the source regions
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.pyplot as plt
import cmasher as cmr
from scipy.stats import poisson, kstest, uniform
from statsmodels.tsa.stattools import adfuller
from exod.utils.logger import logger
from exod.utils.path import data_processed, data_results
from exod.processing.experimental.background_estimate import compute_background

def plot_lightcurve_alerts(cube, tab_boundingboxes, time_interval, obsid):
    """
    This function creates a single panel lightcurve of the source region (source + background)
    :param cube: full data cube
    :param tab_boundingboxes: bounding boxes of variable objects, as obtained from extract_variability_regions.py
    :return: nothing, but saves the lightcurve of each source
    """
    color = cmr.take_cmap_colors('cmr.ocean',N=1,cmap_range=(0.3,0.3))[0]
    for ind, source in enumerate(tab_boundingboxes):
        legend_plots=[]
        legend_labels=[]
        lc = np.nansum(cube[source[0]:source[2], source[1]:source[3]], axis=(0,1))
        lc_generated = np.random.poisson(lc, (5000,len(lc)))
        lc_percentiles = np.nanpercentile(lc_generated, (16,84), axis=0)

        plt.figure(figsize=(5,3))
        plt.title(f'obsid={obsid} | src={ind}')
        p1 = plt.step(range(len(lc)+1),list(lc)+[lc[-1]], c=color, where="post")
        p2 = plt.fill(np.NaN, np.NaN, facecolor=color, alpha=0.4)

        # Plot Error regions
        plt.fill_between(range(len(lc)),
                lc_percentiles[0],
                lc_percentiles[1],
                alpha=0.4,
                facecolor=color,
                step="post", label='16 and 84 percentiles')

        legend_plots.append((p1[0],p2[0]))
        legend_labels.append("Source+background")
        plt.ylabel('Variability')
        plt.xlabel('Frame Number')

        plt.legend(legend_plots,legend_labels)
        # plt.savefig(data_results / f'{obsid}' / f'{time_interval}' / f'lc_{ind}.png')

def plot_lightcurve_alerts_with_background(cube, cube_background, cube_background_withsource, tab_boundingboxes):
    """
    This function creates the multi-panel lightcurve of the transient object. It will retrieve the background,
    and use it to compare the (source+background) to the background, and compute the likelihood in each frame
    :param cube: full data cube
    :param cube_background: data cube of de-sourced background estimate
    :param cube_background_withsource: data cube of de-sourced background estimate + constant contribution from the
    sources (i.e. we assume they are constant, take their stacked flux and distribute it over all frames)
    :param tab_boundingboxes: bounding boxes of variable objects, as obtained from extract_variability_regions.py
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



def get_region_lightcurves(cube, df_regions):
    """
    Extract the lightcurves from the variable regions found.

    Parameters
    ----------
    cube
    df_regions

    Returns
    -------
    lcs : List of Lightcurves
    """
    logger.info("Extracting lightcurves from data cube")
    lcs = []
    for i, row in df_regions.iterrows():
        bbox = row['bbox']
        lc = np.sum(cube[bbox[0]:bbox[2], bbox[1]:bbox[3]], axis=(0,1))
        lcs.append(lc)
    return lcs

def calc_KS_probability(lc):
    """
    Calculate the KS Probability assuming the lightcurve was
    created from a possion distribution with the mean of the lightcurve.
    """
    logger.info("Calculating KS Probability, assuming a Poission Distribution")
    lc_mean = mean_of_poisson = np.nanmean(lc)
    N_lc = len(lc)
    result = kstest(lc, [lc_mean] * N_lc)
    expected_distribution = poisson(mean_of_poisson)
    ks_statistic, ks_p_value = kstest(lc, expected_distribution.cdf)
    logger.info(f'lc_mean = {lc_mean}, N_lc = {N_lc}, ks_statistic = {ks_statistic} ks_p_value = {ks_p_value}')
    return ks_statistic, ks_p_value



if __name__=='__main__':
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    from exod.pre_processing.read_events_files import read_EPIC_events_file
    from exod.processing.variability_computation import calc_var_img, convolve_variability
    from exod.post_processing.extract_variability_regions import extract_variability_regions,plot_variability_with_regions, get_regions_sky_position
    from exod.utils.synthetic_data import create_fake_burst
    from matplotlib.colors import LogNorm

    cube, coordinates_XY = read_EPIC_events_file('0831790701', 10, 100, 3,
                                                 gti_only=True, min_energy=0.2, max_energy=2)
    cube += create_fake_burst(cube.shape, 100, time_peak_fraction=0.05,
                                       position=(0.41*cube.shape[0],0.36*cube.shape[1]),
                                       width_time=100, amplitude=1e0, size_arcsec=10)
    variability_map = calc_var_img(cube)
    tab_centersofmass, bboxes = extract_variability_regions(variability_map, 8)
    print(calc_KS_probability(cube))
    # cube, coordinates_XY = read_EPIC_events_file('0831790701', 10, 1000,3,
    #                                             gti_only=True, min_energy=0.2, max_energy=2)
    # cube += create_fake_burst(cube.shape, 1000, time_peak_fraction=0.05,
    #                                    position=(0.41*cube.shape[0],0.36*cube.shape[1]),
    #                                    width_time=100, amplitude=1e0, size_arcsec=10)
    # plot_variability_with_regions(var_img, 8,
    #                                os.path.join(data_processed,'0831790701','plot_test_varregions.png'))
    plot_lightcurve_alerts(cube, bboxes)
    # print(get_regions_sky_position('0831790701', tab_centersofmass, coordinates_XY))
    # cube_background, cube_background_withsource =compute_background(cube)
    # plot_lightcurve_alerts_with_background(cube, cube_background,cube_background_withsource,bboxes)
