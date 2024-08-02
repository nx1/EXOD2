import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from astropy.table import Table

from exod.processing.coordinates import rotate_XY
from exod.utils.path import savepaths_combined, data_plots, data_combined, data_util
from exod.xmm.bad_obs import obsids_to_exclude


def get_pointing_angle(obsid, tab_xmm_obslist):
    """
    Get the pointing angle for a given observation ID, reads from the xmm_obslist table.
    http://xmmssc.irap.omp.eu/Catalogue/4XMM-DR14/4xmmdr14_obslist.fits
    """
    angle = tab_xmm_obslist[tab_xmm_obslist['OBS_ID'] == obsid]['PA_PNT'].value[0]
    return angle

def rotate_regions_to_detector_coords(clobber=True):
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
    print(df_regions_rotated)
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
        ax[i].scatter(sub['X_EPIC'], sub['Y_EPIC'], s=1.0, alpha=0.15, color='black', marker='.', label=lab,
                      rasterized=True)
        ax[i].set_xlim(-17500, 17500)
        ax[i].set_ylim(-17500, 17500)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].legend(loc='lower right')
    plt.subplots_adjust(wspace=0, hspace=0)
    print('Saving to: 02_12_spatial_dist.png')

    plt.savefig(data_plots / '02_12_spatial_dist.png', dpi=300)
    plt.savefig(data_plots / '02_12_spatial_dist.pdf', dpi=300)
    plt.show()
    

if __name__ == "__main__":
    df_regions_rotated = rotate_regions_to_detector_coords(clobber=False)
    # df_regions_rotated = rotate_regions_to_detector_coords()
    plot_regions_detector_coords(df_regions_rotated)


