import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u

from exod.utils.path import savepaths_combined, data_plots
from exod.utils.plotting import use_scienceplots, plot_aitoff, plot_aitoff_density
from exod.post_processing.extract_lc_features import extract_lc_features
from exod.post_processing.cluster_regions import cluster_regions, get_unique_regions
from exod.xmm.bad_obs import obsids_to_exclude


def print_event_info(df_evt):
    print(df_evt['instrument'].value_counts(normalize=True) * 100)
    print(df_evt['submode'].value_counts(normalize=True) * 100)
    print(df_evt['N_event_lists'].value_counts(normalize=True) * 100)
    print(f'Total Exposure processed = {df_evt['exposure'].sum():.2e} s')
    print(f'Total Events processed = {df_evt['N_events'].sum():.2e}')


def plot_event_info_mean_rate(df_evt):
    plt.figure()
    plt.hist(df_evt['mean_rate'], bins=np.linspace(0,200,100), density=True)
    plt.xlabel('Mean Observation Count Rate (ct/s)')
    plt.ylabel('Normalized Fraction')
    savepath = data_plots / 'mean_rate_hist.pdf'
    print(f'Saving to: {savepath}')
    plt.savefig(savepath)

def process_evt_info():
    print('Processing Event Information...')
    df_evt = pd.read_csv(savepaths_combined['evt_info'])
    print_event_info(df_evt)
    plot_event_info_mean_rate(df_evt)
    print('='*80)

def process_data_cube_info():
    print('Processing Data Cube Information...')
    df_dc = pd.read_csv(savepaths_combined['dc_info'])
    print(f'Total Number of Data Cells processed = {df_dc['total_values'].sum():.2e}')
    df_dc['gti_exposure'] = df_dc['n_gti_bin'] * df_dc['time_interval']
    df_dc['bti_exposure'] = df_dc['n_bti_bin'] * df_dc['time_interval']
    print(f'Total GTI exposure = {df_dc['gti_exposure'].sum():.2e}')
    print(f'Total BTI exposure = {df_dc['bti_exposure'].sum():.2e}')
    print(f'Total GTI/BTI ratio = {df_dc['gti_exposure'].sum() / df_dc['bti_exposure'].sum():.2f}')
    print('='*80)

def process_run_info():
    print('Processing Run Information...')
    df_run_info = pd.read_csv(savepaths_combined['run_info'], dtype={'obsid':'str'})
    vc = df_run_info['n_regions'].value_counts(normalize=True) * 100

    print(f'Total Runs: {len(df_run_info['obsid'])}')
    print(f'Total Observations Run: {df_run_info['obsid'].nunique()}')
    print('Number of Regions per Run:')
    print(vc[vc.index < 11])
    print('Number of subsets per observation')
    print(df_run_info['total_subsets'].value_counts(normalize=True))
    print('='*80)


def calc_df_lc_feat_filter_flags(df_lc_feat):
    print('Calculating Light Curve Feature Filter Flags...')
    # Filter flag for regions that have less than 5 counts maximum in 5 second binning
    df_lc_feat['filt_tbin_5_n_l_5'] = (df_lc_feat['runid'].str.contains('_5_')) & (df_lc_feat['n_max'] < 5)

    # Filter flag for runids with more than 20 detected regions
    vc_runid = df_lc_feat['runid'].value_counts()
    df_lc_feat['filt_many_detections'] = df_lc_feat['runid'].isin(vc_runid.index[vc_runid > 20])

    # Filter flag for 5 sigma detections
    df_lc_feat['filt_5sig'] = (df_lc_feat['B_peak_log_max'] > 13.2) | (df_lc_feat['B_eclipse_log_max'] > 12.38)

    # Filter flag for excluded obsids
    df_lc_feat['obsid'] = df_lc_feat['runid'].str.extract(r'(\d{10})')
    df_lc_feat['filt_exclude_obsid'] = df_lc_feat['obsid'].isin(obsids_to_exclude)

    # Print the number of each flag
    flag_cols = ['n_max_isolated_flare', 'n_max_first_bin', 'n_max_last_bin', 'filt_tbin_5_n_l_5', 'filt_5sig', 'filt_exclude_obsid']
    for col in flag_cols:
        num = len(df_lc_feat[df_lc_feat[col] == True])
        print(f'{col:<20} : {num}')

    return df_lc_feat


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
    bins = np.linspace(0, 20, 100)
    plt.hist(df_lc_feat[df_lc_feat[col] < np.inf][col], bins=bins, label='All', histtype='step', color='black')
    labs = [r'$t_{\mathrm{bin}}=5$', r'$t_{\mathrm{bin}}=50$', r'$t_{\mathrm{bin}}=200$']
    tbins = ['_5_', '_50_', '_200_']
    for i, sim in enumerate(tbins):
        sub = df_lc_feat[df_lc_feat['runid'].str.contains(sim)]
        sub = sub[sub['B_peak_log_max'] < np.inf]
        plt.hist(sub['B_peak_log_max'], bins=bins, label=f'{labs[i]}', histtype='step')

    plt.xlabel(r'Maximum value of $\mathrm{log10}(B_{peak})$ in LC')
    plt.ylabel('Number of Regions')
    plt.legend()
    print(f'Saving to: B_peak_histogram.png')
    plt.savefig(data_plots / 'B_peak_histogram.png')
    plt.savefig(data_plots / 'B_peak_histogram.pdf')


def plot_B_eclipse_histogram(df_lc_feat):
    plt.figure()
    col = 'B_eclipse_log_max'
    bins = np.linspace(0, 20, 100)
    plt.hist(df_lc_feat[df_lc_feat[col] < np.inf][col], bins=bins, label='All', histtype='step', color='black')
    labs = [r'$t_{\mathrm{bin}}=5$', r'$t_{\mathrm{bin}}=50$', r'$t_{\mathrm{bin}}=200$']
    tbins = ['_5_', '_50_', '_200_']
    for i, sim in enumerate(tbins):
        sub = df_lc_feat[df_lc_feat['runid'].str.contains(sim)]
        sub = sub[sub['B_peak_log_max'] < np.inf]
        plt.hist(sub['B_peak_log_max'], bins=bins, label=f'{labs[i]}', histtype='step')

    plt.xlabel(r'Maximum value of $\mathrm{log10}(B_{eclipse})$ in LC')
    plt.ylabel('Number of Regions')
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
    plt.show()



def process_lc_features():
    df_lc_features = extract_lc_features(clobber=False)
    df_lc_features = calc_df_lc_feat_filter_flags(df_lc_features)
    print_number_of_regions_breakdown(df_lc_features)
    plot_total_counts_hist_full(df_lc_features)
    plot_total_counts_hist_small_n(df_lc_features)
    plot_B_eclipse_histogram(df_lc_features)
    plot_B_peak_histogram(df_lc_features)
    plot_B_values_all_regions(df_lc_features)
    plot_N_isolated_flares_vs_N_max(df_lc_features)
    plot_B_peak_threshold_vs_N_reg(df_lc_features)
    plot_n_regions_against_n_max_filter(df_lc_features)


def process_regions():
    df_regions = pd.read_csv(savepaths_combined['regions'])
    df_regions['cluster_label'] = cluster_regions(df_regions, clustering_radius=20 * u.arcsec)
    df_regions_unique = get_unique_regions(df_regions, clustering_radius=20 * u.arcsec)
    plot_aitoff(ra_deg=df_regions_unique['ra_deg'], dec_deg=df_regions_unique['dec_deg'],
                savepath=data_plots / 'unique_regions_aitoff.pdf')
    plot_aitoff_density(ra_deg=df_regions_unique['ra_deg'], dec_deg=df_regions_unique['dec_deg'],
                        savepath=data_plots / 'unique_regions_aitoff_density.pdf')


def main():
    use_scienceplots()
    for k,v in savepaths_combined.items():
        print(f'{k:<15} {v.exists()}')

    process_evt_info()
    process_data_cube_info()
    process_run_info()
    process_regions()
    process_lc_features()

    plt.show()



if __name__ == "__main__":
    main()