import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u
from astropy.table import Table
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from exod.post_processing.make_exod_cat import make_exod_catalogues
from exod.post_processing.hot_regions import rotate_regions_to_detector_coords, \
    plot_regions_detector_coords, hot_regions, plot_hot_regions
from exod.post_processing.crossmatch import crossmatch_unique_regions
import exod.post_processing.simbad_stats as simbad_stats
from exod.processing.bayesian_computations import get_bayes_thresholds
from exod.utils.path import savepaths_combined, data_plots, savepaths_util
from exod.utils.plotting import use_scienceplots, plot_aitoff, plot_aitoff_density
from exod.post_processing.extract_lc_features import extract_lc_features, calc_df_lc_feat_filter_flags, \
    print_df_lc_feat_filter_flag_stats
from exod.post_processing.cluster_regions import ClusterRegions
from exod.utils.simbad_classes import simbad_classifier

def check_results_shape():
    """Check if the length of the results are consistent."""
    print('Checking results shape...')
    print('-------------------------')
    shapes = {}
    for name, path in savepaths_combined.items():
        if name == 'lc':
            continue
        if 'fits' in path.name:
            continue
        if 'alerts' in path.name:
            continue
        print(name, path)
        df = pd.read_csv(path)
        shape = df.shape
        shapes[name] = shape
        print(f'{name:<15}: {shape}')

    assert shapes['regions_unique'][0] == shapes['cmatch_simbad'][0]
    assert shapes['regions_unique'][0] == shapes['cmatch_gaia'][0]
    assert shapes['regions_unique'][0] == shapes['cmatch_om'][0]
    assert shapes['regions_unique'][0] == shapes['cmatch_dr14'][0]
    print('-------------------------\n\n')


def print_event_info(df_evt):
    print(f"Total events processed      = {df_evt['N_events'].sum():,.0f}")
    print(f"Mean events per observation = {df_evt['N_events'].mean():,.0f} - {df_evt['N_events'].quantile(0.16):,} + {df_evt['N_events'].quantile(0.84):,} (16th/84th Percentiles)")
    print(f"Mean observation count rate = {df_evt['mean_rate'].mean():.2f} - {df_evt['mean_rate'].quantile(0.16):.2f} + {df_evt['mean_rate'].quantile(0.84):.2f} ct/s")
    print(f"Mean Exposure Time          = {df_evt['exposure'].mean():,.0f} - {df_evt['exposure'].quantile(0.16):,.0f} + {df_evt['exposure'].quantile(0.84):,.0f} s")
    print('\n')

    print('Instrument Distribution:')
    print(df_evt['instrument'].value_counts(normalize=True) * 100)
    print('\n')
    print('Submode Distribution:')
    print(df_evt['submode'].value_counts(normalize=True) * 100)
    print('\n')

    print('Number of Event Lists Distribution:')
    print(df_evt['N_event_lists'].value_counts(normalize=True) * 100)
    print('\n')


def plot_event_info_mean_rate(df_evt):
    plt.figure()
    plt.hist(df_evt['mean_rate'], bins=np.linspace(0, 200, 100), density=True)
    plt.xlabel('Mean Observation Count Rate (ct/s)')
    plt.ylabel('Normalized Fraction')
    savepath = data_plots / 'mean_rate_hist.pdf'
    print(f'Saving to: {savepath}')
    plt.savefig(savepath)


def plot_mean_events_per_time_frame(df_evt):
    df_dc = pd.read_csv(savepaths_combined['dc_info'])
    plt.figure()
    bins = ['_5_', '_50_', '_200_']
    labs = [r'$t_{\mathrm{bin}}=5$', r'$t_{\mathrm{bin}}=50$', r'$t_{\mathrm{bin}}=200$']
    for i, b in enumerate(bins):
        mask = df_dc['runid'].str.contains(b)
        sub1 = df_dc[mask]
        sub2 = df_evt[mask]
        evts_per_time_bin = sub2['N_events'] / sub1['n_t_bins']
        print(f'Time Binning={b:<5} Events per time frame = {evts_per_time_bin.mean():,.0f} - {evts_per_time_bin.quantile(0.16):,.0f} + {evts_per_time_bin.quantile(0.84):,.0f} (16th/84th Percentiles)')
        plt.hist(np.log10(evts_per_time_bin), bins=np.linspace(0,6.5,100), label=fr'{labs[i]}', histtype='step')
    plt.xlabel(r'$\mathrm{log_{10}}$(Events per time frame)')
    plt.ylabel('Number of Runs')
    plt.legend()

    print(f'Saving to: events_per_time_frame_hist.pdf')
    plt.savefig(data_plots / 'events_per_time_frame_hist.pdf')
    plt.savefig(data_plots / 'events_per_time_frame_hist.png')


def process_evt_info():
    print('Processing EventList Information...')
    print('-----------------------------------')
    df_evt = pd.read_csv(savepaths_combined['evt_info'])
    print_event_info(df_evt)
    plot_event_info_mean_rate(df_evt)
    plot_mean_events_per_time_frame(df_evt)
    print('-----------------------------------\n\n')


def process_data_cube_info():
    print('Processing Data Cube Information...')
    print('-----------------------------------')
    df_dc = pd.read_csv(savepaths_combined['dc_info'])
    df_dc['gti_exposure'] = df_dc['n_gti_bin'] * df_dc['time_interval']
    df_dc['bti_exposure'] = df_dc['n_bti_bin'] * df_dc['time_interval']

    print(f"Total Data Cells    = {df_dc['total_values'].sum():,}")
    print(f"Total GTI Bins      = {df_dc['n_gti_bin'].sum():,}")
    print(f"Total BTI Bins      = {df_dc['n_bti_bin'].sum():,}")
    print(f"Total GTI exposure  = {df_dc['gti_exposure'].sum():,}")
    print(f"Total BTI exposure  = {df_dc['bti_exposure'].sum():,}")
    print(f"Total GTI/BTI ratio = {df_dc['gti_exposure'].sum() / df_dc['bti_exposure'].sum():.2f}")
    print('-----------------------------------\n\n')


def process_run_info():
    print('Processing Run Information...')
    print('-----------------------------')
    df_run_info = pd.read_csv(savepaths_combined['run_info'], dtype={'obsid':'str'})
    vc = df_run_info['n_regions'].value_counts(normalize=True) * 100

    print(f"Total Runs         = {len(df_run_info['obsid']):,}")
    print(f"Total Observations = {df_run_info['obsid'].nunique():,}\n")
    print('Number of Regions per Run:')
    print(vc[vc.index < 11])
    print('\n')
    print('Number of subsets per observation:')
    print(df_run_info['total_subsets'].value_counts(normalize=True))
    print('\n')
    print('-----------------------------\n\n')

def print_number_of_regions_breakdown(df_lc_feat):
    sims = ['_5_0.2_12.0', '_5_0.2_2.0', '_5_2.0_12.0',
            '_50_0.2_12.0', '_50_0.2_2.0', '_50_2.0_12.0',
            '_200_0.2_12.0', '_200_0.2_2.0', '_200_2.0_12.0']

    print('3sig:')
    c = []
    for i, sim in enumerate(sims):
        sub = df_lc_feat[df_lc_feat['runid'].str.contains(sim)]
        c.append(len(sub))
    print(np.array(c).reshape(3, 3))
    print('bottom:', np.array(c).reshape(3, 3).sum(axis=0))
    print('right:', np.array(c).reshape(3, 3).sum(axis=1))
    print('tot', np.array(c).reshape(3, 3).sum())
    print('==========')

    print('5sig:')
    df_5sig = df_lc_feat[df_lc_feat['filt_5sig']]
    c = []
    for i, sim in enumerate(sims):
        sub = df_5sig[df_5sig['runid'].str.contains(sim)]
        c.append(len(sub))
    print(np.array(c).reshape(3, 3))
    print('bottom:', np.array(c).reshape(3, 3).sum(axis=0))
    print('right:', np.array(c).reshape(3, 3).sum(axis=1))
    print('tot', np.array(c).reshape(3, 3).sum())


def print_n_lcs_by_peaks(df_lc_features, sigma=3):
    B_peak_threshold, B_eclipse_threshold = get_bayes_thresholds(sigma)
    mask_peak = df_lc_features['B_peak_log_max'] > B_peak_threshold
    mask_eclipse = df_lc_features['B_eclipse_log_max'] > B_eclipse_threshold

    df_lc_peaks = df_lc_features[mask_peak & ~mask_eclipse]
    df_lc_eclipses = df_lc_features[~mask_peak & mask_eclipse]
    df_lc_eclipses_and_peak = df_lc_features[mask_peak & mask_eclipse]
    df_neither = df_lc_features[~mask_peak & ~mask_eclipse]

    n_peaks_only        = len(df_lc_peaks)
    n_eclipse_only      = len(df_lc_eclipses)
    n_peak_and_eclipses = len(df_lc_eclipses_and_peak)
    n_neither           = len(df_neither)
    print(f'Calculating breakdown of Lightcurves using sigma={sigma}')
    print('---------------------------------------------------')
    print(f'Number of Lightcurves with only peaks    = {n_peaks_only:,} / {len(df_lc_features):,} ({100*n_peaks_only/len(df_lc_features):.2f}%)')
    print(f'Number of Lightcurves with only Eclipses = {n_eclipse_only:,} / {len(df_lc_features):,} ({100*n_eclipse_only/len(df_lc_features):.2f}%)')
    print(f'Number of Lightcurves with both          = {n_peak_and_eclipses:,} / {len(df_lc_features):,} ({100*n_peak_and_eclipses/len(df_lc_features):.2f}%)')
    print(f'Number of Lightcurves with neither       = {n_neither:,} / {len(df_lc_features):,} ({100*n_neither/len(df_lc_features):.2f}%)')
    print('The reason for some lightcurves having neither (at 3sig) is because the threshold value was calculated over each pixel of the data cube, however this B values were then re-calculated for the extracted lightcurves')
    print('---------------------------------------------------\n')
    assert n_peaks_only + n_eclipse_only + n_peak_and_eclipses + n_neither == len(df_lc_features)


def plot_total_counts_hist_full(df_lc_feat):
    plt.figure()
    plt.hist(np.log10(df_lc_feat['n_sum']), bins=100, label='All', histtype='step', color='black')
    labs = [r'$t_{\mathrm{bin}}=5$', r'$t_{\mathrm{bin}}=50$', r'$t_{\mathrm{bin}}=200$']
    tbins = ['_5_', '_50_', '_200_']
    for i, sim in enumerate(tbins):
        sub = df_lc_feat[df_lc_feat['runid'].str.contains(sim)]
        plt.hist(np.log10(sub['n_sum']), bins=100, label=f'{labs[i]}', histtype='step')

    plt.xlabel(r'Total LC Counts $log_{10}(N_{\mathrm{tot}})$')
    plt.ylabel('Number of Regions')
    plt.legend()
    print(f'Saving to: total_lc_counts_hist.png')
    plt.savefig(data_plots / 'total_lc_counts_hist.png')
    plt.savefig(data_plots / 'total_lc_counts_hist.pdf')


def plot_B_peak_histogram(df_lc_feat):
    plt.figure()
    col = 'B_peak_log_max'
    xmin = 5.94
    bins = np.linspace(xmin, 20, 100)
    plt.hist(df_lc_feat[df_lc_feat[col] < np.inf][col], bins=bins, label='All', histtype='step', color='black')
    labs = [r'$t_{\mathrm{bin}}=5$', r'$t_{\mathrm{bin}}=50$', r'$t_{\mathrm{bin}}=200$']
    tbins = ['_5_', '_50_', '_200_']
    for i, sim in enumerate(tbins):
        sub = df_lc_feat[df_lc_feat['runid'].str.contains(sim)]
        sub = sub[sub['B_peak_log_max'] < np.inf]
        plt.hist(sub['B_peak_log_max'], bins=bins, label=f'{labs[i]}', histtype='step')

    plt.xlabel(r'Maximum value of $\mathrm{log10}(B_{peak})$ in LC')
    plt.ylabel('Number of Regions')
    plt.xlim(xmin, 20)
    plt.legend()
    print(f'Saving to: B_peak_histogram.png')
    plt.savefig(data_plots / 'B_peak_histogram.png')
    plt.savefig(data_plots / 'B_peak_histogram.pdf')


def plot_B_eclipse_histogram(df_lc_feat):
    plt.figure()
    col = 'B_eclipse_log_max'
    xmin = 5.7
    bins = np.linspace(xmin, 20, 100)
    plt.hist(df_lc_feat[df_lc_feat[col] < np.inf][col], bins=bins, label='All', histtype='step', color='black')
    labs = [r'$t_{\mathrm{bin}}=5$', r'$t_{\mathrm{bin}}=50$', r'$t_{\mathrm{bin}}=200$']
    tbins = ['_5_', '_50_', '_200_']
    for i, sim in enumerate(tbins):
        sub = df_lc_feat[df_lc_feat['runid'].str.contains(sim)]
        sub = sub[sub['B_eclipse_log_max'] < np.inf]
        plt.hist(sub['B_eclipse_log_max'], bins=bins, label=f'{labs[i]}', histtype='step')

    plt.xlabel(r'Maximum value of $\mathrm{log10}(B_{eclipse})$ in LC')
    plt.ylabel('Number of Regions')
    plt.xlim(xmin, 20)
    plt.legend()
    print(f'Saving to: B_eclipse_histogram.png')
    plt.savefig(data_plots / 'B_eclipse_histogram.png')
    plt.savefig(data_plots / 'B_eclipse_histogram.pdf')


def plot_B_values_all_regions(df_lc_feat):
    plt.figure()
    plt.hist(df_lc_feat[df_lc_feat['B_peak_log_max']<np.inf]['B_peak_log_max'], bins=np.linspace(0,40,100), histtype='step', label=r'$B_{peak}$', color='green', lw=1.0)
    plt.hist(df_lc_feat[df_lc_feat['B_eclipse_log_max']<np.inf]['B_eclipse_log_max'], bins=np.linspace(0,40,100), histtype='step', label=r'$B_{eclipse}$', color='blue', lw=1.0)
    plt.xlabel(r'Maximum value of $log_{10}$(B) in LC')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.xlim(0,40)
    print(f'Saving to: B_values_distribution_all_regions.png')
    plt.savefig(data_plots / 'B_values_distribution_all_regions.png')
    plt.savefig(data_plots / 'B_values_distribution_all_regions.pdf')
    #plt.show()


def plot_total_counts_hist_small_n(df_lc_feat):
    plt.figure()
    bins = np.arange(0, 202, 1)
    plt.hist(df_lc_feat['n_sum'], bins=bins, label='All', histtype='step', color='black')
    labs = [r'$t_{\mathrm{bin}}=5$', r'$t_{\mathrm{bin}}=50$', r'$t_{\mathrm{bin}}=200$']
    tbins = ['_5_', '_50_', '_200_']
    for i, sim in enumerate(tbins):
        sub = df_lc_feat[df_lc_feat['runid'].str.contains(sim)]
        plt.hist(sub['n_sum'], bins=bins, label=f'{labs[i]}', histtype='step')

    plt.xlabel(r'Total LC Counts ($N_{\mathrm{tot}})$')
    plt.ylabel('Number of Regions')
    plt.legend()
    plt.xlim(0, 200)
    print(f'Saving to: total_lc_counts_hist_small_n.png')
    plt.savefig(data_plots / 'total_lc_counts_hist_small_n.png')
    plt.savefig(data_plots / 'total_lc_counts_hist_small_n.pdf')


def plot_N_isolated_flares_vs_N_max(df_lc_feat):
    hi = 20
    n_maxs = np.arange(0, hi, 1)

    n_isolated = []
    n_not_isolated = []

    for n_max in n_maxs:
        sub = df_lc_feat[df_lc_feat['n_max'] > n_max]
        n_isolated.append(len(sub[sub['n_max_isolated_flare'] == True]))
        n_not_isolated.append(len(sub[sub['n_max_isolated_flare'] == False]))

    plt.figure()
    plt.plot(n_maxs, n_isolated, label='Isolated Flares', color='red')
    plt.plot(n_maxs, n_not_isolated, label='Not Isolated Flares', color='green')
    plt.xlabel('N max (1 bin)')
    plt.ylabel('Number of Lightcurves')
    plt.legend(loc='lower right')
    plt.xlim(0, hi - 1)
    plt.ylim(0)
    plt.grid()
    print('Saving to: number_of_isolated_flares.png')
    plt.savefig(data_plots / 'number_of_isolated_flares.png')
    plt.savefig(data_plots / 'number_of_isolated_flares.pdf')
    #plt.show()


def plot_B_peak_threshold_vs_N_reg(df_lc_feat):
    Bs = np.linspace(-5, 100,1000)
    count1 = []
    count2 = []

    for B in Bs:
        sub1 = df_lc_feat[df_lc_feat['B_peak_log_max'] > B]
        sub2 = df_lc_feat[df_lc_feat['B_eclipse_log_max'] > B]
        count1.append(len(sub1))
        count2.append(len(sub2))

    plt.figure()
    plt.plot(Bs, count1, label=r'$B_{Peak}$', color='green')
    plt.plot(Bs, count2, label=r'$B_{eclipse}$', color='blue')
    plt.xlabel(r'$log_{10}B_{Peak}$ Threshold')
    plt.ylabel(r'Number of Sources')
    plt.axvline(6.4, color='red', lw=1.0)
    plt.axvline(5.5, color='red', lw=1.0)
    plt.xlim(0,100)
    plt.grid()
    plt.legend()
    print('Saving to: B_peak_threshold_vs_N_reg.png')
    plt.savefig(data_plots / 'B_peak_threshold_vs_N_reg.png')
    plt.savefig(data_plots / 'B_peak_threshold_vs_N_reg.pdf')


def plot_n_regions_against_n_max_filter(df_lc_feat):

    df_5sec = df_lc_feat[df_lc_feat['runid'].str.contains('_5_')]
    df_50sec = df_lc_feat[df_lc_feat['runid'].str.contains('_50_')]
    df_200sec = df_lc_feat[df_lc_feat['runid'].str.contains('_200_')]

    dfs = {'All':df_lc_feat,
           r'$t_{\mathrm{bin}}$ = 5':df_5sec,
           r'$t_{\mathrm{bin}}$ = 50':df_50sec,
           r'$t_{\mathrm{bin}}$ = 200':df_200sec}

    colors = ['black','C0','C1','C2']

    plt.figure()
    i=0
    for k, df_ in dfs.items():
        ns = np.arange(0,25,1)
        count = []
        for n in ns:
            sub = df_[df_['n_max'] > n]
            count.append(len(sub))

        plt.plot(ns, count, label=k, color=colors[i])
        i+=1
    #plt.xticks(ns)
    plt.xlabel('N max (1 bin)')
    plt.ylabel('number of sources')
    plt.legend(loc='upper right')
    plt.xlim(0, 24)
    plt.grid()
    print(f'Saving to: source_against_n_counts_filter.png')
    plt.savefig(data_plots / 'source_against_n_counts_filter.png')
    plt.savefig(data_plots / 'source_against_n_counts_filter.pdf')
    # plt.show()


def print_significant_bins_stats(df_lc_feat):
    df_lc_feat['has_3_sig_gti_peak'] = df_lc_feat['n_3_sig_peak_bins_gti'] > 0
    df_lc_feat['has_5_sig_gti_peak'] = df_lc_feat['n_5_sig_peak_bins_gti'] > 0
    df_lc_feat['has_3_sig_bti_peak'] = df_lc_feat['n_3_sig_peak_bins_bti'] > 0
    df_lc_feat['has_5_sig_bti_peak'] = df_lc_feat['n_5_sig_peak_bins_bti'] > 0

    df_lc_feat['has_3_sig_gti_eclipse'] = df_lc_feat['n_3_sig_eclipse_bins_gti'] > 0
    df_lc_feat['has_5_sig_gti_eclipse'] = df_lc_feat['n_5_sig_eclipse_bins_gti'] > 0
    df_lc_feat['has_3_sig_bti_eclipse'] = df_lc_feat['n_3_sig_eclipse_bins_bti'] > 0
    df_lc_feat['has_5_sig_bti_eclipse'] = df_lc_feat['n_5_sig_eclipse_bins_bti'] > 0

    n_3_sig_gti_peak_detections = df_lc_feat['has_3_sig_gti_peak'].sum()
    n_5_sig_gti_peak_detections = df_lc_feat['has_5_sig_gti_peak'].sum()
    n_3_sig_bti_peak_detections = df_lc_feat['has_3_sig_bti_peak'].sum()
    n_5_sig_bti_peak_detections = df_lc_feat['has_5_sig_bti_peak'].sum()

    n_3_sig_gti_eclipse_detections = df_lc_feat['has_3_sig_gti_eclipse'].sum()
    n_5_sig_gti_eclipse_detections = df_lc_feat['has_5_sig_gti_eclipse'].sum()
    n_3_sig_bti_eclipse_detections = df_lc_feat['has_3_sig_bti_eclipse'].sum()
    n_5_sig_bti_eclipse_detections = df_lc_feat['has_5_sig_bti_eclipse'].sum()


    print('Fraction of GTI to BTI detections')
    print('---------------------------------')
    print(f'Number of regions with GTI peak detections:    {n_3_sig_gti_peak_detections:,} (3sig) {n_5_sig_gti_peak_detections:,} (5sig)')
    print(f'Number of regions with BTI peak detections:    {n_3_sig_bti_peak_detections:,} (3sig) {n_5_sig_bti_peak_detections:,} (5sig)')
    print(f'Number of regions with GTI eclipse detections: {n_3_sig_gti_eclipse_detections:,} (3sig) {n_5_sig_gti_eclipse_detections:,} (5sig)')
    print(f'Number of regions with BTI eclipse detections: {n_3_sig_bti_eclipse_detections:,} (3sig) {n_5_sig_bti_eclipse_detections:,} (5sig)')
    # print(f'Number of regions with GTI & BTI detections: {} (3sig) {} (5sig)')
    print('---------------------------------')


    len_tot            = df_lc_feat['len'].sum()
    n_peak_3sig_tot    = df_lc_feat['n_3_sig_peak_bins'].sum()
    n_eclipse_3sig_tot = df_lc_feat['n_3_sig_eclipse_bins'].sum()
    n_peak_5sig_tot    = df_lc_feat['n_5_sig_peak_bins'].sum()
    n_eclipse_5sig_tot = df_lc_feat['n_5_sig_eclipse_bins'].sum()

    perc_3_sig_peak    = (n_peak_3sig_tot / len_tot) * 100
    perc_3_sig_eclipse = (n_eclipse_3sig_tot / len_tot) * 100
    perc_5_sig_peak    = (n_peak_5sig_tot / len_tot) * 100
    perc_5_sig_eclipse = (n_eclipse_5sig_tot / len_tot) * 100

    perc_3_sig_peak_gti    = (df_lc_feat['n_3_sig_peak_bins_gti'].sum() / n_peak_3sig_tot) * 100
    perc_3_sig_peak_bti    = (df_lc_feat['n_3_sig_peak_bins_bti'].sum() / n_peak_3sig_tot) * 100
    perc_3_sig_eclipse_gti = (df_lc_feat['n_3_sig_eclipse_bins_gti'].sum() / n_eclipse_3sig_tot) * 100
    perc_3_sig_eclipse_bti = (df_lc_feat['n_3_sig_eclipse_bins_bti'].sum() / n_eclipse_3sig_tot) * 100

    perc_5_sig_peak_gti    = (df_lc_feat['n_5_sig_peak_bins_gti'].sum() / n_peak_5sig_tot) * 100
    perc_5_sig_peak_bti    = (df_lc_feat['n_5_sig_peak_bins_bti'].sum() / n_peak_5sig_tot) * 100
    perc_5_sig_eclipse_gti = (df_lc_feat['n_5_sig_eclipse_bins_gti'].sum() / n_eclipse_5sig_tot) * 100
    perc_5_sig_eclipse_bti = (df_lc_feat['n_5_sig_eclipse_bins_bti'].sum() / n_eclipse_5sig_tot) * 100

    print('Significant Bins Statistics:')
    print('----------------------------')
    print(f"Total Number of 3 sigma peak bins    = {n_peak_3sig_tot :,} / {len_tot :,} ({perc_3_sig_peak:.2f}%) (gti = {df_lc_feat['n_3_sig_peak_bins_gti'].sum():,} ({perc_3_sig_peak_gti:.2f}%)) (bti = {df_lc_feat['n_3_sig_peak_bins_bti'].sum():,} ({perc_3_sig_peak_bti:.2f}%))")
    print(f"Total Number of 3 sigma eclipse bins = {n_eclipse_3sig_tot:,} / {len_tot:,} ({perc_3_sig_eclipse:.2f}%) (gti = {df_lc_feat['n_3_sig_eclipse_bins_gti'].sum():,} ({perc_3_sig_eclipse_gti:.2f}%)) (bti = {df_lc_feat['n_3_sig_eclipse_bins_bti'].sum():,} ({perc_3_sig_eclipse_bti:.2f}%))")
    print(f"Total Number of 5 sigma peak bins    = {n_peak_5sig_tot:,} / {len_tot:,} ({perc_5_sig_peak:.2f}%) (gti = {df_lc_feat['n_5_sig_peak_bins_gti'].sum():,} ({perc_5_sig_peak_gti:.2f}%)) (bti = {df_lc_feat['n_5_sig_peak_bins_bti'].sum():,} ({perc_5_sig_peak_bti:.2f}%))")
    print(f"Total Number of 5 sigma eclipse bins = {n_eclipse_5sig_tot:,} / {len_tot:,} ({perc_5_sig_eclipse:.2f}%) (gti = {df_lc_feat['n_5_sig_eclipse_bins_gti'].sum():,} ({perc_5_sig_eclipse_gti:.2f}%)) (bti = {df_lc_feat['n_5_sig_eclipse_bins_bti'].sum():,} ({perc_5_sig_eclipse_bti:.2f}%))")
    print('----------------------------\n\n')
    input()



def process_lc_features(clobber=True):
    df_lc_features = extract_lc_features(clobber=clobber)
    df_lc_features = calc_df_lc_feat_filter_flags(df_lc_features)
    print_df_lc_feat_filter_flag_stats(df_lc_features)
    print_number_of_regions_breakdown(df_lc_features)
    print_n_lcs_by_peaks(df_lc_features, sigma=3)
    print_n_lcs_by_peaks(df_lc_features, sigma=5)
    print_significant_bins_stats(df_lc_features)
    plot_total_counts_hist_full(df_lc_features)
    plot_total_counts_hist_small_n(df_lc_features)
    plot_B_eclipse_histogram(df_lc_features)
    plot_B_peak_histogram(df_lc_features)
    plot_B_values_all_regions(df_lc_features)
    plot_N_isolated_flares_vs_N_max(df_lc_features)
    plot_B_peak_threshold_vs_N_reg(df_lc_features)
    plot_n_regions_against_n_max_filter(df_lc_features)


def plot_xmm_dr14_flux_comparison(tab_xmm_cmatch):
    tab = tab_xmm_cmatch
    tab = tab[tab['SEP_ARCSEC'] < 30]
    v = tab['SC_EP_8_FLUX']
    tab_xmm = Table.read(savepaths_util['4xmm_dr14_slim'])
    tab_xmm_var = tab_xmm[tab_xmm['SC_VAR_FLAG']]

    plt.figure()
    plt.hist(np.log10(tab_xmm['SC_EP_8_FLUX']), bins=np.linspace(-16, -8, 100), density=True, histtype='step', label='XMM DR14')
    plt.hist(np.log10(tab_xmm_var['SC_EP_8_FLUX']), bins=np.linspace(-16, -8, 100), density=True, histtype='step', label='XMM DR14 Variable')
    plt.hist(np.log10(v), bins=np.linspace(-16, -8, 100), density=True, histtype='step', label='EXOD Detections', color='red')
    plt.xlabel('log$_{10}$ 0.2-12.0 keV Flux (erg s$^{-1}$ cm${^-2}$)')
    plt.ylabel('Normalized Fraction')
    plt.legend()
    print(f'Saving to: Flux_comparison.pdf')
    plt.savefig(data_plots / 'Flux_comparison.pdf')
    plt.savefig(data_plots / 'Flux_comparison.png')


def plot_cmatch_seperations(dfs_cmatch):
    plt.figure()
    # for k, tab in dfs_cmatch.items():
    # Plot the labels in same order as plot.
    for k in ['GAIA DR3', 'XMM OM', 'XMM DR14', 'SIMBAD', 'GLADE+']:
        tab = dfs_cmatch[k]
        N = len(tab[tab['SEP_ARCSEC'] < 20])
        percent = N / len(tab) * 100
        plt.hist(tab['SEP_ARCSEC'], bins=np.linspace(0, 20, 75), histtype='step',
                 label=f'{k} ({N}) {percent:.0f}\\%')
    plt.xlim(0, 20)
    plt.xlabel('Separation (arcsec)')
    plt.ylabel('Number of Sources')
    plt.legend(fontsize=8, markerfirst=False)
    plt.tight_layout()
    print(f'Saving to: crossmatch_sep.pdf')
    plt.savefig(data_plots / 'crossmatch_sep.pdf')
    plt.savefig(data_plots / 'crossmatch_sep.png')


def plot_cmatch_offset_scatter(tab_cmatch_xmm):
    ra = tab_cmatch_xmm['RA_OFFSET'] * 3600
    dec = tab_cmatch_xmm['DEC_OFFSET'] * 3600

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(ra, dec, marker='.', s=0.25, color='grey', rasterized=True)
    ax.scatter(ra.mean(), dec.mean(), color='red', marker='+', s=300, label='Mean')

    ext = 200
    ax.set_xlim(-ext, ext)
    ax.set_ylim(-ext, ext)
    ax.axvline(0, color='black', ls='dashed', lw=1.0)
    ax.axhline(0, color='black', ls='dashed', lw=1.0)
    ax.set_xlabel('RA Offset (arcsec)')
    ax.set_ylabel('Dec Offset (arcsec)')
    ax.legend(loc='upper left')

    ax_inset = inset_axes(ax, width="30%", height="30%", loc='upper right')

    zoom_ext = 5
    ax_inset.scatter(ra, dec, marker='.', s=0.25, color='grey', rasterized=True)
    ax_inset.set_xlim(-zoom_ext, zoom_ext)
    ax_inset.set_ylim(-zoom_ext, zoom_ext)
    ax_inset.axvline(0, color='black', ls='dashed', lw=1.0)
    ax_inset.axhline(0, color='black', ls='dashed', lw=1.0)
    ax_inset.scatter(ra.mean(), dec.mean(), color='red', marker='+', s=100)

    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    ax_inset.text(-4.8, 4.0, s='5" Zoom')
    mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="blue", color='red')
    print(f'Saving to: dr14_offsets.png')
    plt.savefig(data_plots / 'dr14_offsets.pdf')
    plt.savefig(data_plots / 'dr14_offsets.png', dpi=150)


def plot_simbad_types_bar(df_cmatch_simbad):
    vc = df_cmatch_simbad['main_type'].value_counts()
    print('Distribution of Simbad Types:')
    print(vc)
    print('\n')
    fig, ax = plt.subplots(figsize=(8, 30))
    vc.sort_values().plot(kind='barh', ax=ax)
    plt.savefig(data_plots / 'simbad_types_bar.pdf')

    df_cmatch_simbad['otype_sub'] = [simbad_classifier.get(otype) for otype in df_cmatch_simbad['main_type']]
    vc2 = df_cmatch_simbad['otype_sub'].value_counts()
    print('Distribution of Simbad Types (Subtypes):')
    print(vc2)
    print(df_cmatch_simbad['otype_sub'].value_counts(normalize=True))
    print('\n')

    fig, ax = plt.subplots()
    vc2.sort_values().plot(kind='barh', ax=ax)
    plt.savefig(data_plots / 'simbad_types_sub_bar.pdf')


def plot_gaia_hr_diagram(tab):
    sub = tab[~np.isnan(tab['BP-RP'])]
    # sub = sub[sub['SEP_ARCSEC'] < 10]
    fig, ax = plt.subplots(figsize=(3.5, 5))
    ax.scatter(sub['BP-RP'], sub['Gmag'], marker='x', s=0.5, label=f'GAIA DR3 : {len(sub)}')
    # ax.set_xlim(0,7)
    # ax.set_ylim(-5.5,15.5)
    ax.invert_yaxis()
    plt.legend()
    ax.set_ylabel(r'G Band Magnitude $\mathrm{M_G}$')
    ax.set_xlabel(r'$\mathrm{G_{BP} - G_{RP}}$')
    print(f'Saving to: gaia_hr_diagram.png')
    plt.savefig(data_plots / 'gaia_hr_diagram.png')
    plt.savefig(data_plots / 'gaia_hr_diagram.pdf')


def print_cmatch_numbers(dfs_cmatch):
    print('Number of sources by seperation for each crossmatch:')
    print('=====================================================')
    for k, v in dfs_cmatch.items():
        n = len(v)
        n_match_20 = len(v[v["SEP_ARCSEC"] < 20])
        n_match_10 = len(v[v["SEP_ARCSEC"] < 10])
        print(f'{k:<9} : Seperation<20" = {n_match_20:<6} ({100 * (n_match_20 / n):.2f}%) Seperation<10" = {n_match_10:<6} ({100 * (n_match_10 / n):.2f}%)  (Total Rows = {n:,})')
    print('=====================================================')
    print('\n\n')

def print_chime_frb_counts():
    df_chime = Table.read(savepaths_util['CHIME_FRB']).to_pandas()
    df_cmatch_chime = pd.read_csv(savepaths_combined['cmatch_chime'])

    n_chime_rows = len(df_chime)
    n_unique_chime_sources = df_chime['tns_name'].nunique()
    n_unique_regions = df_cmatch_chime['region_num'].nunique()
    n_unique_chime_names = df_cmatch_chime['tns_name'].nunique()

    print(f'Number of rows in CHIME catalogue           = {n_chime_rows}')
    print(f'Number of unique sources in CHIME catalogue = {n_unique_chime_sources}')
    print(f'Number of unique EXOD Regions               = {n_unique_regions}')
    print(f'Number of unique CHIME/FRB sources          = {n_unique_chime_names}')


def print_xmm_dr14_cmatch_stats(df_cmatch_xmm_dr14):
    print('XMM DR14 Crossmatch Statistics:')
    print('===============================')

    df = df_cmatch_xmm_dr14
    mask = df['SEP_ARCSEC'] < 20
    df_l_20 = df[mask]
    df_g_20 = df[~mask]

    df_l_20_transient = df_l_20[df_l_20['SC_VAR_FLAG'] == True]

    sep_mean = df_l_20['SEP_ARCSEC'].mean()
    sep_std = df_l_20['SEP_ARCSEC'].std()

    ra_offset_mean = df_l_20['RA_OFFSET'].mean() * 3600
    ra_offset_std = df_l_20['RA_OFFSET'].std() * 3600

    dec_offset_mean = df_l_20['DEC_OFFSET'].mean() * 3600
    dec_offset_std = df_l_20['DEC_OFFSET'].std() * 3600

    print(f'Information for XMM DR14 Crossmatches with seperations < 20"')
    print(f'Seperation = {sep_mean:.2f} +- {sep_std:.2f} arcseconds')
    print(f'RA Offset  = {ra_offset_mean:.2f} +- {ra_offset_std:.2f} arcseconds')
    print(f'DEC Offset = {dec_offset_mean:.2f} +- {dec_offset_std:.2f} arcseconds')



    print(f'Number of XMM DR14 sources within 20"            = {len(df_l_20):,} / {len(df):,} ({100 * len(df_l_20) / len(df):.2f}%)')
    print(f'Number of XMM DR14 sources outside 20"           = {len(df_g_20):,} / {len(df):,} ({100 * len(df_g_20) / len(df):.2f}%)')
    print(f'Number of XMM DR14 Transient Sources within 20"  = {len(df_l_20_transient):,} / {len(df):,} ({100 * len(df_l_20_transient) / len(df):.2f}%)')
    print(f'Number of XMM DR14 Transient Sources outside 20" = {len(df_g_20[df_g_20["SC_VAR_FLAG"] == True]):,} / {len(df_g_20):,} ({100 * len(df_g_20[df_g_20["SC_VAR_FLAG"] == True]) / len(df_g_20):.2f}%)')
    print('===============================\n\n')


def plot_om_ab_magnitudes(df_cmatch_om):
    cols = ['UVW2mAB', 'UVM2mAB', 'UVW1mAB', 'UmAB', 'BmAB', 'VmAB']
    labels = ['UVW2', 'UVM2', 'UVW1', 'U', 'B', 'V']

    plt.figure()
    for i, c in enumerate(cols):
        sub = df_cmatch_om[~df_cmatch_om[c].isnull()]
        plt.hist(sub[c], bins=np.linspace(10, 25, 100), label=f'{labels[i]} {(len(sub))}', histtype='step', lw=1.0)
    plt.legend(markerfirst=False)
    plt.xlabel('AB Magnitude')
    plt.ylabel('Number of Regions')
    print(f'Saving to: OM_magnitudes.png')
    plt.savefig(data_plots / 'OM_magnitudes.png')
    plt.savefig(data_plots / 'OM_magnitudes.pdf')
    # plt.show()


def process_regions(clobber=True):
    df_regions = pd.read_csv(savepaths_combined['regions'])
    cr = ClusterRegions(df_regions)
    df_regions_unique = cr.df_regions_unique

    plot_aitoff(ra_deg=df_regions_unique['ra_deg'], dec_deg=df_regions_unique['dec_deg'], savepath=data_plots / 'unique_regions_aitoff.pdf')
    plot_aitoff_density(ra_deg=df_regions_unique['ra_deg'], dec_deg=df_regions_unique['dec_deg'], savepath=data_plots / 'unique_regions_aitoff_density.pdf')

    dfs_cmatch = crossmatch_unique_regions(df_regions_unique.reset_index(), clobber=clobber)

    print_cmatch_numbers(dfs_cmatch)
    print_xmm_dr14_cmatch_stats(dfs_cmatch['XMM DR14'])

    plot_cmatch_seperations(dfs_cmatch)
    plot_xmm_dr14_flux_comparison(dfs_cmatch['XMM DR14'])
    plot_cmatch_offset_scatter(dfs_cmatch['XMM DR14'])
    plot_simbad_types_bar(dfs_cmatch['SIMBAD'])
    plot_gaia_hr_diagram(dfs_cmatch['GAIA DR3'])
    plot_om_ab_magnitudes(dfs_cmatch['XMM OM'])

    df_reg_rotated = rotate_regions_to_detector_coords(df_regions, clobber=clobber)
    plot_regions_detector_coords(df_reg_rotated)
    plot_hot_regions(df_reg_rotated, hot_regions)


def main(clobber=True):
    print('EXOD POST PROCESSING')
    print('====================')
    import matplotlib.pyplot as plt
    use_scienceplots()
    check_results_shape()
    process_evt_info()
    process_data_cube_info()
    process_run_info()
    process_regions(clobber=clobber)
    process_lc_features(clobber=clobber)
    print_chime_frb_counts()
    make_exod_catalogues()
    simbad_stats.main()
    plt.show()


if __name__ == "__main__":
    main(clobber=False)
