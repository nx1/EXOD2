import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import csv
from matplotlib import pyplot as plt
from astropy.io import ascii, fits
from astropy.table import Table
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from pathlib import Path

from exod.utils.path import data_processed, savepaths_combined, data_plots, data_combined, read_observation_ids, \
    data_util
from exod.xmm.observation import Observation


def rotate_position(X, Y, angle, pivotXY=(25719, 25719)):
    """
    Rotates the positions following the pointing angle of the observation, so that the coordinates are in an EPIC frame.

    Parameters:
        X (float): The X sky coordinate.
        Y (float): The Y sky coordinate.
        angle (float): The pointing angle of the observation (PA_PNT).
        pivotXY (tuple): The pivot point for the rotation.
    """
    X_EPIC = (X - pivotXY[0]) * np.cos(-angle * np.pi / 180) - (Y - pivotXY[1]) * np.sin(-angle * np.pi / 180)
    Y_EPIC = (X - pivotXY[0]) * np.sin(-angle * np.pi / 180) + (Y - pivotXY[1]) * np.cos(-angle * np.pi / 180)
    return X_EPIC, Y_EPIC

def get_pointing_angle(obsid, tab_xmm_obslist):
    #obs = Observation(obsid)
    #obs.get_event_lists_processed()
    #evt = obs.events_processed[0]
    #evt.read(remove_bad_rows=False, remove_borders=False, remove_MOS_central_ccd=False, remove_hot_pixels=False)
    #angle = evt.pnt_angle
    #270.9496
    angle = tab_xmm_obslist[tab_xmm_obslist['OBS_ID'] == obsid]['PA_PNT'].value[0]
    return angle

def get_transients(obsid, df_regions):
    sub = df_regions[df_regions['obsid'] == obsid]
    return sub

def calculate_all_new_positions():
    df_regions = pd.read_csv(savepaths_combined['regions'])
    tab_xmm_obslist = Table.read(data_util / '4xmmdr14_obslist.fits')
    df_regions['obsid'] = df_regions['runid'].str.extract(r'(\d{10})')
    all_res = []
    for obsid in tqdm(df_regions['obsid'].unique()):
        try:
            angle = get_pointing_angle(obsid, tab_xmm_obslist)
            df_transients = get_transients(obsid, df_regions)
        except:
            print(f'Error with {obsid}')
            continue
        for i, row in df_transients.iterrows():
            X_EPIC, Y_EPIC = rotate_position(row['X'], row['Y'], angle)
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
    df_res = pd.DataFrame(all_res)
    print(df_res)
    df_res.to_csv(data_combined / 'transients_rotated.csv', index=False)
    # Save all the new positions.

    fig, ax = plt.subplots(figsize=(10,10))
    ax.hist2d(df_res['X_EPIC'], df_res['Y_EPIC'], bins=(120, 115), norm=LogNorm())
    ax.set_xlabel("EPIC frame X")
    ax.set_ylabel("EPIC frame Y")
    plt.savefig(data_plots / 'transients_hist_rotated.png')
    plt.savefig(data_plots / 'transients_hist_rotated.pdf')

    fig, ax = plt.subplots(figsize=(10,10))
    plt.scatter(df_res['X_EPIC'], df_res['Y_EPIC'], s=1)
    ax.set_xlabel("EPIC frame X")
    ax.set_ylabel("EPIC frame Y")
    plt.savefig(data_plots / 'transients_scatter_rotated.png')
    plt.savefig(data_plots / 'transients_scatter_rotated.pdf')
    plt.show()


if __name__ == "__main__":
    calculate_all_new_positions()


