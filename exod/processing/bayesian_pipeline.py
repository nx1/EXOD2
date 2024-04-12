from exod.processing.bayesian_computations import precompute_bayes_1000, load_precomputed_bayes1000, \
    load_precomputed_bayes_limits, precompute_bayes_limits, get_cube_masks_peak_and_eclipse, B_peak, B_eclipse, \
    PrecomputeBayesLimits
from exod.utils.plotting import cmap_image
from exod.utils.logger import logger
from exod.pre_processing.data_loader import DataLoader
from exod.utils.util import save_df, save_info
from exod.xmm.event_list import EventList
from exod.xmm.observation import Observation
from exod.processing.template_based_background_inference import compute_expected_cube_using_templates
from exod.processing.coordinates import get_regions_sky_position, calc_df_regions

from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, BoundaryNorm
import pandas as pd
from astropy.visualization import ImageNormalize, SqrtStretch


def plot_region_lightcurve(df_lcs, i, savepath=None):
    """Plot the ith region lightcurve."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax2 = ax.twiny()
    t0 = df_lcs['time'] - df_lcs['time'].min()
    ax.step(t0, df_lcs[f'n_{i}'], where='post', color='black', lw=1.0, label='Observed (n)')
    ax.step(t0, df_lcs[f'mu_{i}'], where='post', color='red', lw=1.0, label=r'Expected ($\mu$)')

    ax2.step(range(len(df_lcs[f'n_{i}'])), df_lcs[f'n_{i}'], where='post', color='none', lw=1.0)

    ax.set_title(f'Detected Region #{i}')
    ax.set_ylabel('Counts (N)')
    ax.set_xlabel('Time (s)')
    ax.set_xlim(t0.min(), t0.max())
    ax2.set_xlabel('Window/Frame Number')
    ax.legend(loc='upper right')
    plt.tight_layout()

    if savepath:
        logger.info(f'Saving lightcurve plot to: {savepath}')
        plt.savefig(savepath)


def plot_detection_image(df_regions, image_eclipse, image_n, image_peak, savepath=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    # Map the peaks=1 and eclipses=2
    image_combined = np.zeros_like(image_peak)
    image_combined[image_peak > 0] = 1
    image_combined[image_eclipse > 0] = 2
    image_combined[(image_peak > 0) & (image_eclipse > 0)] = 3
    c_0 = 'none'
    c_peak = 'cyan'
    c_eclipse = 'lime'
    c_both = 'blue'
    cmap = ListedColormap(colors=[c_0, c_peak, c_eclipse, c_both])
    norm = BoundaryNorm(boundaries=[0, 1, 2, 3, 4], ncolors=4)
    norm2 = ImageNormalize(stretch=SqrtStretch())
    im1 = ax.imshow(image_n.T, cmap=cmap_image(), norm=norm2, interpolation='none', origin='lower')
    ax.imshow(image_combined.T, cmap=cmap, norm=norm, interpolation='none', origin='lower')
    cbar = plt.colorbar(im1, ax=ax, shrink=0.75)
    cbar.set_label('Total Counts')
    ax.scatter(df_regions['weighted_centroid-0'], df_regions['weighted_centroid-1'], marker='+', s=10, color='white')
    for i, row in df_regions.iterrows():
        x_cen = row['centroid-0']
        y_cen = row['centroid-1']

        width = row['bbox-2'] - row['bbox-0']
        height = row['bbox-3'] - row['bbox-1']

        x_pos = x_cen - width / 2
        y_pos = y_cen - height / 2

        rect = patches.Rectangle(xy=(x_pos, y_pos),
                                 width=width,
                                 height=height,
                                 linewidth=1,
                                 edgecolor='white',
                                 facecolor='none')

        plt.text(x_pos + width, y_pos + height, str(i), c='white')
        ax.add_patch(rect)
    lab_kwargs = {'markeredgecolor': None, 'marker': 's', 'markersize': 10, 'ls': 'none'}
    legend_labels = [
        plt.Line2D([0], [0], label='Peak', markerfacecolor=c_peak, **lab_kwargs),
        plt.Line2D([0], [0], label='Eclipse', markerfacecolor=c_eclipse, **lab_kwargs),
        plt.Line2D([0], [0], label='Peak & Eclipse', markerfacecolor=c_both, **lab_kwargs)
    ]
    ax.legend(handles=legend_labels)

    plt.tight_layout()
    if savepath:
        logger.info(f'Saving Image to {savepath}')
        plt.savefig(savepath)
    # plt.show()


def plot_lc_pixel(cube_mu, cube_n, time_interval, x, y):
    cube_mu_xy = cube_mu[x, y]
    cube_data_xy = cube_n.data[x, y]
    # Plot lightcurves
    lw = 1.0
    frame_axis = np.arange(cube_mu.shape[2])  # Frame Number
    time_axis = frame_axis * time_interval  # Zero'ed Time
    time_axis2 = cube_n.bin_t[:-1]  # Observation Time
    mu_3sig, n_peak_3sig, n_eclipse_3sig = load_precomputed_bayes_limits(threshold_sigma=3)
    mu_5sig, n_peak_5sig, n_eclipse_5sig = load_precomputed_bayes_limits(threshold_sigma=5)
    fig, ax = plt.subplots(2, 1, figsize=(15, 5), gridspec_kw={'height_ratios': [10, 1]}, sharex=True)
    ax[0].fill_between(time_axis, n_eclipse_5sig(cube_mu_xy), n_peak_5sig(cube_mu_xy), alpha=0.3, facecolor='blue', label=r'5 $\sigma$')
    ax[0].fill_between(time_axis, n_eclipse_3sig(cube_mu_xy), n_peak_3sig(cube_mu_xy), alpha=0.5, facecolor='blue', label=r'3 $\sigma$')
    ax[0].step(time_axis, cube_data_xy, color='black', where='mid', lw=lw, label=r'Observed ($n$)')
    ax[0].step(time_axis, cube_mu_xy, color='red', where='mid', lw=lw, label=r'Expected ($\mu$)')
    ax[1].step(time_axis, cube_mu_xy, color='red', where='mid', lw=lw, label=r'Expected ($\mu$)')
    ax2 = ax[0].twiny()
    ax2.plot(frame_axis, cube_data_xy, color='none')
    ax2.set_xlabel("Frame #")
    ax[0].legend()
    ax[1].set_xlabel("Time (s)")
    ax[0].set_ylabel("Counts")
    ax[0].set_ylim(0)
    ax[0].set_xlim(np.min(time_axis), np.max(time_axis))
    plt.subplots_adjust(hspace=0)
    plt.suptitle(f'Lightcurve for pixel x={x} y={y}')


def get_region_lightcurves(df_regions, cube_n, cube_mu, savepath=None):
    if len(df_regions) == 0:
        logger.info('No Regions found, No lightcurves produced.')
        return None
    lcs = [pd.DataFrame({'time' : cube_n.bin_t[:-1]}),
           pd.DataFrame({'bti'  : cube_n.bti_bin_idx_bool[:-1]}),
           pd.DataFrame({'bccd' : cube_n.bccd_bin_idx_bool})]
    for i, row in df_regions.iterrows():
        xlo, xhi = row['bbox-0'], row['bbox-2']
        ylo, yhi = row['bbox-1'], row['bbox-3']

        lc_n         = extract_lc_from_cube(cube_n.data, xhi, xlo, yhi, ylo, dtype=np.int32)
        lc_mu        = extract_lc_from_cube(cube_mu, xhi, xlo, yhi, ylo, dtype=np.float32)
        lc_B_peak    = B_peak(n=lc_n, mu=lc_mu)
        lc_B_eclipse = B_eclipse(n=lc_n, mu=lc_mu)

        lcs.append(pd.DataFrame({f'n_{i}': lc_n}))
        lcs.append(pd.DataFrame({f'mu_{i}': lc_mu}))
        lcs.append(pd.DataFrame({f'B_peak_{i}': lc_B_peak}))
        lcs.append(pd.DataFrame({f'B_eclipse_{i}': lc_B_eclipse}))
    df_lcs = pd.concat(lcs, axis=1)
    if savepath:
        logger.info(f'Saving lightcurve plot to: {savepath}')
        plt.savefig(savepath)
    return df_lcs


def extract_lc_from_cube(data_cube, xhi, xlo, yhi, ylo, dtype=np.int32):
    data = data_cube[xlo:xhi, ylo:yhi]
    lc = np.nansum(data, axis=(0, 1), dtype=dtype)
    return lc


def run_pipeline(obsid,
                 size_arcsec=20,
                 time_interval=5,
                 gti_only=False,
                 remove_partial_ccd_frames=True,
                 gti_threshold=1.5,
                 min_energy=0.2,
                 max_energy=10.0,
                 clobber=False,
                 precomputed_bayes_limit=PrecomputeBayesLimits(threshold_sigma=3)):
    precomputed_bayes_limit.load()
    observation = Observation(obsid)
    observation.filter_events(clobber=clobber)
    observation.create_images(clobber=clobber)
    observation.get_files()
    observation.get_events_overlapping_subsets()

    for i_subset, subset_overlapping_exposures in enumerate(observation.events_overlapping_subsets):
        savedir = observation.path_results / f'subset_{i_subset}'
        savedir.mkdir(exist_ok=True)
        
        event_list = EventList.from_event_lists(subset_overlapping_exposures)
        dl = DataLoader(event_list=event_list, time_interval=time_interval, size_arcsec=size_arcsec,
                        gti_only=gti_only, min_energy=min_energy, max_energy=max_energy,
                        gti_threshold=gti_threshold, remove_partial_ccd_frames=remove_partial_ccd_frames)
        dl.run()

        img = observation.images[0]
        img.read(wcs_only=True)

        cube_n = dl.data_cube

        # DataCube(cube_n.data[:,:,0:60]).video()

        # cube_n.video()
        cube_mu = compute_expected_cube_using_templates(cube_n, wcs=img.wcs)

        # cube_mask_peaks, cube_mask_eclipses = get_cube_masks_peak_and_eclipse(cube_n=cube_n.data, cube_mu=cube_mu, threshold_sigma=threshold_sigma)
        cube_mask_peaks, cube_mask_eclipses = precomputed_bayes_limit.get_cube_masks_peak_and_eclipse(cube_n=cube_n.data, cube_mu=cube_mu)

        image_n       = np.nansum(cube_n.data, axis=2)        # Total Counts.
        image_peak    = np.nansum(cube_mask_peaks, axis=2)    # Each Pixel is the number of peaks in cube_n
        image_eclipse = np.nansum(cube_mask_eclipses, axis=2) # Each Pixel is the number of eclipses in cube_n

        image_mask_combined = (image_peak > 0) | (image_eclipse > 0)
        df_reg = calc_df_regions(image=image_n, image_mask=image_mask_combined)
        df_sky = get_regions_sky_position(data_cube=cube_n, df_regions=df_reg, wcs=img.wcs)
        df_regions = pd.concat([df_reg, df_sky], axis=1).reset_index(drop=True)
        df_lcs = get_region_lightcurves(df_regions, cube_n, cube_mu)

        # Plot Lightcurves for each region
        # for i in df_regions.index:
        #     plot_region_lightcurve(df_lcs=df_lcs, i=i, savepath=savedir / f'lc_{i}.png')

        # Plot Lightcurves for each pixel.
        # x_peak, y_peak, t_peak = np.where(cube_mask_peaks)
        # x_eclipse, y_eclipse, t_eclipse = np.where(cube_mask_eclipses)
        # unique_xy = [*(get_unique_xy(x_peak, y_peak)), *(get_unique_xy(x_eclipse, y_eclipse))]
        # for x, y in unique_xy:
        #     plot_lc_pixel(cube_mu, cube_n, time_interval, x, y)

        # Plot Image
        plot_detection_image(df_regions, image_eclipse, image_n, image_peak, savepath=savedir / 'detection_img.png')

        # Save Results
        save_df(df=dl.df_bti, savepath=savedir / 'bti.csv')
        save_df(df=df_lcs, savepath=savedir / 'lcs.csv')
        save_df(df=df_regions, savepath=savedir / 'regions.csv')

        save_info(dictionary=observation.info, savepath=savedir / 'obs_info.csv')
        save_info(dictionary=event_list.info, savepath=savedir / 'evt_info.csv')
        save_info(dictionary=dl.info, savepath=savedir / 'dl_info.csv')
        save_info(dictionary=dl.data_cube.info, savepath=savedir / 'data_cube_info.csv')

        # plt.show()
        plt.close('all')

def main():
    from exod.utils.path import read_observation_ids
    from exod.utils.path import data
    precompute_bayes_limits(threshold_sigma=3)
    precompute_bayes_limits(threshold_sigma=5)
    precompute_bayes_1000()
    load_precomputed_bayes1000()

    obsids = read_observation_ids(data / 'observations.txt')
    # obsids = read_observation_ids(data / 'obs_ccd_check.txt')
    shuffle(obsids)

    # obsids=['0792180301']
    # obsids=['0112570701']
    # obsids=['0810811801']#'0764420101',
    # obsids=['0911990501']

    pre = PrecomputeBayesLimits(threshold_sigma=3)
    pre.load()
    for obsid in obsids:
        # observation='0765080801' #'0886121001' '0872390901',
        run_pipeline(obsid, precomputed_bayes_limit=pre)

if __name__=="__main__":
    main()

