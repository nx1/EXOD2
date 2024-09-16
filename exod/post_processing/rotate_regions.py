import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from astropy.table import Table

from exod.processing.coordinates import rotate_XY
from exod.utils.path import savepaths_combined, data_plots, data_combined, data_util
from exod.xmm.bad_obs import obsids_to_exclude


# These hot regions in in the coordinate system obtained from using
# rotate_regions_to_detector_coords()
# they were calculated with interactive_scatter_with_rectangles() is provided in utils.plotting
#(x,y,w,h) the x,y position is defined from the upper left corner NOT THE CENTER!
# These regions were calculated using the central pivot point: (25719, 25719)
hot_regions_5s = [(  9536.02,  13601.21,  588.91,  600.74),  # 1
                  (-12633.42,  10837.71,  399.99,  585.98),  # 2
                  (-16385.08,   6131.49,  399.74,  675.02),  # 3
                  (  2326.62,   7454.90,  677.92,  823.48),  # 4
                  (-15344.51,   1375.52,  203.55,  515.18),  # 5
                  (-14184.01,   -636.23,  341.17,  594.07),  # 6
                  (  2821.32,   5936.04,  763.81,  841.08),  # 7
                  (-11605.95,   8403.36,  394.28,  537.49),  # 8
                  (-12244.45,   7892.09,  336.98,  616.15),  # 9
                  (-11022.03,   7131.74,  402.47,  694.80),  # 10
                  (-11063.07,  12741.59,  426.35,  544.85),  # 11
                  (  1320.53,    555.66,  367.11,  544.62),  # 12
                  (  8468.08,  -2409.62,  243.78,  417.05),  # 13
                  ( 10592.09,  -2256.85,  332.32, 1273.23),  # 14
                  (  6629.57,  -9682.14,  331.13,  550.60),  # 15
                  (  1034.69, -10353.72,  471.91, 1011.96),  # 16
                  (-12251.02, -13606.78,  578.03, 1034.96),  # 17
                  (-11801.35, -16049.66,  338.45,  567.11),  # 18
                  ( -1257.66,  -9323.60,  557.89, 1019.55),  # 19
                  (-10351.30, -11477.23,  695.08,  706.53),  # 20
                  ( -5641.13,  -2382.69,  456.12,  891.63),  # 21
                  (  3093.50,  -2880.54,  369.33,  602.12),  # 22
                  ( -5273.71,  -4844.67,  230.06,  708.37),  # 23
                  ( -6213.36,  -4207.98,  258.50,  501.76),  # 24
                  ( -9871.15,  -2761.79,  568.74, 1024.15),  # 25
                  ( -9829.22,  -3528.73,  448.17,  565.85),  # 26
                  ( -9935.49,  13451.52,  446.67,  492.18),  # 27
                  ( -9822.95,  10157.08,  388.11,  655.86),  # 28
                  ( -9313.35,  10880.79,  462.36,  673.95),  # 29
                  ( -8152.75,  11015.41,  431.12,  817.62),  # 30
                  ( 10832.99,  -8089.06,  370.52,  780.13),  # 31
                  (  7547.46,  -4793.52,  396.51,  398.04),  # 32
                  (  4804.20,  -5489.96,  341.22,  823.52),  # 33
                  ( 11291.97,  11334.69,  296.41,  600.34),  # 34
                  (-15119.26,   6126.88,  365.35,  544.16)]  # 35

hot_regions_50s = [( 9564.06,  13704.37,  551.40,   664.78), # 1
                   ( 2386.98,   7606.88,  552.53,   633.13), # 2
                   ( 2861.12,   6057.22,  562.43,   661.06), # 3
                   (-1245.22,  -9222.14,  721.99,   801.65), # 4
                   (12197.35, -11713.11, 4911.11, 24088.55)] # 5 (right side of MOS)

hot_regions_200s = [(  9556.45, 13422.16,  592.68, 1095.46),  # 1
                    (  2372.75,  7611.68,  547.97,  766.90),  # 2
                    (-15047.51,  1284.05, 8869.69,  746.79)]  # 3 (readout streak)

hot_regions = {'_5_'   : hot_regions_5s,
               '_50_'  : hot_regions_50s,
               '_200_' : hot_regions_200s}


def get_pointing_angle(obsid, tab_xmm_obslist):
    """
    Get the pointing angle for a given observation ID, reads from the xmm_obslist table.
    http://xmmssc.irap.omp.eu/Catalogue/4XMM-DR14/4xmmdr14_obslist.fits
    """
    angle = tab_xmm_obslist[tab_xmm_obslist['OBS_ID'] == obsid]['PA_PNT'].value[0]
    return angle

def rotate_regions_to_detector_coords(clobber=True):
    print('Rotating regions to detector coordinates...')
    savepath = data_combined / 'transients_rotated.csv'
    if savepath.exists() and not clobber:
        print(f'{savepath} already exists.')
        df_regions_rotated = pd.read_csv(savepath)
        print(df_regions_rotated)
        return df_regions_rotated
    df_regions = pd.read_csv(savepaths_combined['regions'])
    tab_xmm_obslist = Table.read(data_util / '4xmmdr14_obslist.fits')
    df_regions['obsid'] = df_regions['runid'].str.extract(r'(\d{10})')
    all_res = []
    for obsid in tqdm(df_regions['obsid'].unique()):
        try:
            angle = get_pointing_angle(obsid, tab_xmm_obslist)
            df_transients = df_regions[df_regions['obsid'] == obsid]
        except IndexError:
            print(f'Error with {obsid} no pointing angle found. Skipping...')
        except Exception as e:
            print(f'Error with {obsid} {e} {type(e).__name__}')
            continue
        for i, row in df_transients.iterrows():
            X_EPIC, Y_EPIC = rotate_XY(row['X'], row['Y'], angle)
            res = {
                'obsid': obsid,
                'runid': row['runid'],
                'label': row['label'],
                'angle': angle,
                'X': row['X'],
                'Y': row['Y'],
                'X_EPIC': X_EPIC,
                'Y_EPIC': Y_EPIC
            }
            all_res.append(res)
    df_regions_rotated = pd.DataFrame(all_res)
    df_regions_rotated['EXOD_DETID'] = df_regions_rotated['runid'] + '_' + df_regions_rotated['label'].astype(str)
    df_regions_rotated.to_csv(savepath, index=False)
    return df_regions_rotated


def plot_regions_detector_coords(df_regions_rotated):
    df_regions_rotated = df_regions_rotated[~df_regions_rotated['obsid'].isin(obsids_to_exclude)]

    subs = ['_5_0.2_12.0', '_50_0.2_12.0', '_200_0.2_12.0']
    labels = [r'$t_{\mathrm{bin}}=5$~s', r'$t_{\mathrm{bin}}=50$~s', r'$t_{\mathrm{bin}}=200$~s']

    width = 9.0
    fig, ax = plt.subplots(1, 3, figsize=(width, width / 3))
    for i, s in enumerate(subs):
        sub = df_regions_rotated[df_regions_rotated['runid'].str.contains(s)]
        lab = fr'{labels[i]} | $N_{{\mathrm{{reg}}}}$={len(sub)}'
        ax[i].scatter(sub['X_EPIC'], sub['Y_EPIC'], s=1.0, alpha=0.15, color='black', marker='.', label=lab, rasterized=True)
        ax[i].set_xlim(-17500, 17500)
        ax[i].set_ylim(-17500, 17500)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].legend(loc='lower right')
    plt.subplots_adjust(wspace=0, hspace=0)
    print('Saving to: 02_12_spatial_dist.png')

    plt.savefig(data_plots / '02_12_spatial_dist.png', dpi=300)
    plt.savefig(data_plots / '02_12_spatial_dist.pdf', dpi=300)
    # plt.show()


def plot_hot_regions(df_regions_rotated, hot_regions):
    """
    Plot the hot regions and label them.
    """
    # Plot the hot regions.
    for k, v in hot_regions.items():
        sub = df_regions_rotated[df_regions_rotated['runid'].str.contains(k)]

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(sub['X_EPIC'], sub['Y_EPIC'], s=1.0, marker='.', color='black', alpha=0.2)
        for i, (x, y, w, h) in enumerate(v):
            mask = (sub['X_EPIC'] < x + w) & (sub['X_EPIC'] > x) & (sub['Y_EPIC'] < y + h) & (sub['Y_EPIC'] > y)
            sub2 = sub[mask]
            n_reg = len(sub2)
            n_obs = len(sub2['obsid'].unique())

            ax.scatter(sub2['X_EPIC'], sub2['Y_EPIC'], s=2, alpha=1.0, color='red', label=f'{i} : $N_{{reg}}$={n_reg} $N_{{obs}}$={n_obs}')
            offset = 1500
            ax.text(x - offset, y + offset, s=f'{i}', color='black', fontsize=12,
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.1'))
        ax.legend(ncol=3, bbox_to_anchor=(1, 0, 0, 1))
        ax.set_xlim(-17500, 17500)
        ax.set_ylim(-17500, 17500)
        plt.savefig(data_plots / f'hot_regions{k}s.png')
        plt.savefig(data_plots / f'hot_regions{k}s.pdf')
        print(f'Saving to: hot_regions{k}s.png')
        # plt.show()


if __name__ == "__main__":
    df_regions_rotated = rotate_regions_to_detector_coords(clobber=False)
    # df_regions_rotated = rotate_regions_to_detector_coords()
    plot_regions_detector_coords(df_regions_rotated)
    plot_hot_regions(df_regions_rotated, hot_regions)






