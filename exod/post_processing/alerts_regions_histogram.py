import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from matplotlib import pyplot as plt
from astropy.io import ascii, fits
from astropy.table import Table
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from pathlib import Path


from exod.utils.path import data_processed

def convert_position(transients_X, transients_Y, obsid):
    """Deprecated, doesn't work as intended"""
    path_to_obsid_data = data_processed / obsid
    files = os.listdir(path_to_obsid_data)
    path_to_eventlists = [path_to_obsid_data/elt for elt in files if 'FILT' in elt]
    tabx, taby, tabrawx, tabrawy = [],[],[],[]
    for eventlistpath in path_to_eventlists:
        event_list = Table(fits.open(eventlistpath)[1].data)['X','Y','RAWX','RAWY']
        tabx+=list(event_list['X'])
        taby+=list(event_list['Y'])
        tabrawx+=list(event_list['RAWX'])
        tabrawy+=list(event_list['RAWY'])
    X_to_RAWX = interp1d(tabx, tabrawx)
    Y_to_RAWY = interp1d(taby, tabrawy)
    print(obsid)
    return X_to_RAWX(transients_X), Y_to_RAWY(transients_Y)

def rotate_position(transients_X, transients_Y, obsid, pivotXY=(25725,25725)):
    """Rotates the positions following the pointing angle of the observation, so that the coordinates are in an EPIC frame"""
    path_to_obsid_data = data_processed / obsid
    files = os.listdir(path_to_obsid_data)
    path_to_image = [path_to_obsid_data/elt for elt in files if 'IMG' in elt][0]
    angle = -fits.open(path_to_image)[0].header['PA_PNT']

    newX = (transients_X-pivotXY[0])*np.cos(angle*np.pi/180) - (transients_Y-pivotXY[1])*np.sin(angle*np.pi/180)
    newY = (transients_X-pivotXY[0])*np.sin(angle*np.pi/180) + (transients_Y-pivotXY[1])*np.cos(angle*np.pi/180)
    return newX, newY, angle

def build_map_transients():
    """Takes the detected alert regions, picks up the corresponding observation and its pointing angle,
    rotates it to be in an EPIC frame, and then stack the positions of all transients. Should allow to assess the
    presence of unexpected spatial structure in the detection, which would be due to instrumental effects"""

    path_to_alerts = '/home/erwan/Documents/EXOD_stuff/df_regions.csv' #To change of course
    results = Table(ascii.read(path_to_alerts))

    #Splits the result table in chunks, one for each observation
    results = results[np.argsort(results['runid'])]
    observations = np.array([elt[:10] for elt in results['runid']])
    splitted_by_obs = np.array_split(results, np.where(observations[:-1]!=observations[1:])[0]+1)

    tab_all_newx = []
    tab_all_newy = []
    tab_all_oldx = []
    tab_all_oldy = []
    for obsid, result_observation in zip(observations, splitted_by_obs):
        if obsid in os.listdir(data_processed):
            tab_all_oldx += [elt['X'] for elt in result_observation]
            tab_all_oldy += [elt['Y'] for elt in result_observation]
            newx, newy, angle=rotate_position(np.array([elt['X'] for elt in result_observation]),
                                       np.array([elt['Y'] for elt in result_observation]), obsid)
            tab_all_newx += list(newx)
            tab_all_newy += list(newy)
    tab_all_newx, tab_all_newy = np.array(tab_all_newx), np.array(tab_all_newy)

    plt.figure()
    plt.hist2d(tab_all_newx, tab_all_newy, bins=(120,115))#, norm=LogNorm())
    plt.xlabel("EPIC frame X'")
    plt.ylabel("EPIC frame Y'")
    c=plt.colorbar()
    c.set_label("Number of variable regions")
    plt.show()

    plt.figure()
    plt.hist2d(tab_all_oldx, tab_all_oldy, bins=(120, 115))#, norm=LogNorm())
    plt.xlabel('Sky frame X')
    plt.ylabel('Sky frame Y')
    c = plt.colorbar()
    c.set_label("Number of variable regions")
    plt.show()

def test_rotation():
    for obsid in os.listdir(data_processed)[:3]:
        path_to_obsid_data = data_processed / obsid
        list_files = os.listdir(path_to_obsid_data)
        path_to_evtlists = [path_to_obsid_data / elt for elt in list_files if 'FILT' in elt]
        tabx, taby= [], []
        for eventlistpath in path_to_evtlists:
            event_list = Table(fits.open(eventlistpath)[1].data)['X', 'Y']
            tabx += list(event_list['X'])
            taby += list(event_list['Y'])
        tabx, taby = np.array(tabx), np.array(taby)
        newX, newY, angle = rotate_position(tabx, taby, obsid)
        fig, (ax1,ax2) = plt.subplots(2,1, figsize=(5,10))
        ax1.hist2d(tabx,taby, bins=(120, 115), norm=LogNorm())
        ax1.set_title("Old coordinates")
        ax2.hist2d(newX,newY, bins=(120, 115), norm=LogNorm())
        ax2.set_title("New rotated coordinates")
        plt.show()


# test_rotation()
build_map_transients()
