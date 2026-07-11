# -*- coding: utf-8 -*-
"""
Created on Fri May 26 13:41:08 2023

@author:    Jonas Hartmann @ Mayor lab @ UCL CDB

@descript:  Functions for loading and preprocessing tracking data in the
            formats received from StarDist and MTrackJ.
"""


# Imports
import warnings
import numpy as np
import pandas as pd


### StarDist

# Loader func (numpy)
def load_stardist_numpy(fpath):
    """Load stardist data from csv into numpy arrays.

    fpath <<- string, path to valid stardist csv file

    d ->> Complete array as resulting from np.loadtxt
    tfyx ->> Subsetted array containing only track, frame, y-, and x-coords
    """
    d = np.loadtxt(fpath, skiprows=4, delimiter=',', dtype=object)
    tfyx = d[:, (2,8,5,4)].astype(float)
    tfyx = [tfyx[tfyx[:,0]==tID][tfyx[tfyx[:,0]==tID, 1].argsort()]  # Sort each track on frames
            for tID in np.unique(tfyx[:,0])]
    tfyx = np.concatenate(tfyx, axis=0)
    if np.isnan(tfyx).any():
        warnings.warn("Found NaN values in tfyx; something is very likely wrong!")
    return d, tfyx


# Loader func (pandas)
def load_stardist_pandas(fpath):
    """Load stardist data from csv into pandas dataframes.

    fpath <<- string, path to valid stardist csv file

    d ->> Complete dataframe as resulting from pd.read_csv
    tfyx ->> Subsetted df containing only track, frame, y-, and x-coords
    """
    df = pd.read_csv(fpath, sep=',', header=0, index_col=2, skiprows=[1,2,3])
    tfyx = df[['FRAME', 'POSITION_Y', 'POSITION_X']]
    tfyx = tfyx.groupby('TRACK_ID').apply(lambda x: x.sort_values('FRAME'))  # Sort each track on frames
    tfyx.index = tfyx.index.droplevel(0)
    tfyx.columns = ['f', 'y', 'x']
    tfyx.index.names = ['track']
    if np.isnan(tfyx.to_numpy(dtype=float)).any():
        warnings.warn("Found NaN values in tfyx; something is very likely wrong!")
    return df, tfyx


### MTrackJ

# Loader func (numpy)
def load_mtrackj_numpy(fpath, frame_time):
    """Load MTrackJ data from csv into numpy arrays.

    fpath <<- string, path to valid MTrackJ csv file
    frame_time <<- Exact duration of a frame in sec (required bc PID!=frame)

    d ->> Complete array as resulting from np.loadtxt
    tfyx ->> Subsetted array containing only track, frame, y-, and x-coords
    """
    d = np.loadtxt(fpath, skiprows=1, delimiter=',', dtype=object)
    tfyx = d[:, (1,5,4,3)].astype(float)
    tfyx[:, 0] -= 1
    tfyx[:, 1] = np.round(tfyx[:, 1] / frame_time).astype(int)
    if np.isnan(tfyx).any():
        warnings.warn("Found NaN values in tfyx; something is very likely wrong!")
    return d, tfyx


# Loader func (pandas)
def load_mtrackj_pandas(fpath, frame_time):
    """Load MTrackJ data from csv into pandas dataframes.

    fpath <<- string, path to valid MTrackJ csv file
    frame_time <<- Exact duration of a frame in sec (required bc PID!=frame)

    d ->> Complete dataframe as resulting from pd.read_csv
    tfyx ->> Subsetted df containing only track, frame, y-, and x-coords
    """
    df = pd.read_csv(fpath, sep=',', header=0, index_col=1)
    tfyx = df[['t [sec]', 'y [micron]', 'x [micron]']].copy()
    tfyx.columns = ['f', 'y', 'x']
    tfyx['f'] = np.round(tfyx['f'] / frame_time).astype(int)
    tfyx.index.names = ['track']
    tfyx.index = tfyx.index - 1
    if np.isnan(tfyx.to_numpy(dtype=float)).any():
        warnings.warn("Found NaN values in tfyx; something is very likely wrong!")
    return df, tfyx
