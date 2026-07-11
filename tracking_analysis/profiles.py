# -*- coding: utf-8 -*-
"""
Created on MON May 29 18:51:17 2023

@author:    Jonas Hartmann @ Mayor lab @ UCL CDB

@descript:  Functions for computing various correlation and 
            similarity profiles across tracks.
"""


# Imports
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import paired_cosine_distances


# Precomputations

def norm_vector(vec):
    """Normalize vector vec of shape (n_vectors, m_dimensions) to magnitude 1.
    Vectors with only 0.0 components will again be set to 0.0 in the output.
    """
    vec_n = vec / np.linalg.norm(vec, axis=1).reshape((-1, 1))
    vec_n[np.isnan(vec_n)] = 0.0 # Handle zero-magnitude cases
    return vec_n


def align_to_start_end_trajectory(df, pos_cols, vec_cols):
    """Reorient movement vectors associated with a tracked object such that 
    they are relative to the straight start-end trajectory of the object, i.e.
    the line between its positions at the first and last time point.

    This allows comparisons of vectors across tracks that point along different 
    angles, e.g. if objects move away from a central starting point in various
    directions.

    NOTE: This currently works only in 2 dimensions, but it would probably not 
    be too hard to generalize it to n dimensions.
    
    Parameters
    ----------
    df : pandas df; index:'track', columns:(f, t, y, x, ...)
        The df containing the tracking data. Must be structured in the standard
        way expected in tracking_analysis pipelines.
    pos_cols : list
        List of columns containing track coordinates, e.g. ['y', 'x']
    vec_cols : list
        List of columns containing vector components, e.g. ['vy', 'vx']

    Returns
    -------
    df : pandas df; index:'track', columns:(f, t, y, x, ...)
        Updated df containing new columns named as in `vec_cols` but with the
        additional suffix '_traj', containing the reoriented vectors.
    """

    # Get start-end trajectory vectors
    startloc = df.groupby('track')[pos_cols].transform("first")
    endloc   = df.groupby('track')[pos_cols].transform("last")
    traj_vec = norm_vector(endloc - startloc)

    # Get angle to x-axis for trajectories
    traj_ang = np.arctan2(traj_vec[pos_cols[0]], traj_vec[pos_cols[1]])

    # Convert from  "-pi --> 0 (x-axis) --> pi"  to  "0 (x-axis) --> 2*pi"
    traj_ang[traj_ang<0] = traj_ang[traj_ang<0] + 2*np.pi

    # Compile rotation matrix
    rotmat = np.array([[np.cos(traj_ang), -np.sin(traj_ang)], 
                       [np.sin(traj_ang),  np.cos(traj_ang)]])

    # Rotate movement vectors onto trajectories
    out_cols = [v+'_traj' for v in vec_cols]
    df[out_cols] = np.einsum('ij,lji->il', df[vec_cols], rotmat)

    # Done
    return df


def align_to_cluster_center_ray(df, pos_cols, cen_cols, vec_cols):
    """Reorient movement vectors associated with a tracked object such that 
    they are relative to the ray starting at the cluster center and passing 
    through the tracked object's current position.

    This allows explicit comparisons of metrics relative to a straight outward
    trajectory within a circular cluster.
    
    Parameters
    ----------
    df : pandas df; index:'track', columns:(f, t, y, x, ...)
        The df containing the tracking data. Must be structured in the standard
        way expected in tracking_analysis pipelines.
    pos_cols : list
        List of columns containing track coordinates, e.g. ['y', 'x']
    cen_cols : list
        List of columns containing cluster center coordinates, e.g. 
        ['cen_y', 'cen_x']
    vec_cols : list
        List of columns containing vector components, e.g. ['vy', 'vx']
    cluster_center : array of length len(pos_cols), optional, default None
        Cluster center coordinates for the input dataset. If None, the cluster
        center is calculated as the centroid over all data points in df.

    Returns
    -------
    df : pandas df; index:'track', columns:(f, t, y, x, ...)
        Updated df containing new columns named as in `vec_cols` but with the
        additional suffix '_ray', containing the reoriented vectors.
    """
    
    # Get center-cell ray vectors
    ray_vec = norm_vector(df[pos_cols] - df[cen_cols].values)

    # Get angle to x-axis for rays
    ray_ang = np.arctan2(ray_vec[pos_cols[0]], ray_vec[pos_cols[1]])

    # Convert from  "-pi --> 0 (x-axis) --> pi"  to  "0 (x-axis) --> 2*pi"
    ray_ang[ray_ang<0] = ray_ang[ray_ang<0] + 2*np.pi

    # Compile rotation matrix
    rotmat = np.array([[np.cos(ray_ang), -np.sin(ray_ang)], 
                       [np.sin(ray_ang),  np.cos(ray_ang)]])

    # Rotate movement vectors onto rays
    # I don't know why element-wise dot products have to be so hard...
    out_cols = [v+'_ray' for v in vec_cols]
    df[out_cols] = np.einsum('ij,lji->il', df[vec_cols], rotmat)
    
    # Done
    return df


# Custom similarity metrics

def zscore_similarity(A, B):
    """Pairwise relative similarity metric for two measurements A and B.
    
    The two populations are individually z-scored, then the absolute difference
    is computed. This absolute difference is now in "unit variance". Because of
    the 68-95-99.7 rule, the vast majority of absolute differences will lie in
    the range [0 sigma, 3 sigma] if both populations are approximately Normal.
    
    Hence, the difference is inverted into a similarity metric as follows:
    
    `similarity = 1.0 - (absolute_difference / 3.0)`
    
    Note the following when working with or interpreting this metric:
    - All resulting values are <= 1.0
    - 1.0 means two measurements are identical
    - If the two input distributions are approximately Normal:
      - ~0.0 means two measurements are as different as typical samples can be
      - <0.0 is therefore from extrema or outliers of the distributions
    - The metric is symmetric on its input; s(a, b) == s(b, a)
    - Comparing metrics computed from different sets of populations is dubious
      
    Parameters
    ----------
    A, B : numpy arrays of individual measurements, matched shape
        The similarity of each element in A to each corresponding element in B
        is computed. Z-scoring is done for the distributions of all elements in
        A and in B separately.
        
    Returns
    -------
    zsim : numpy array of dtype float, same shape as A and B
        Pairwise similarity metric for each pair of measurements in A and B.
    """
    
    # Z-score the two populations
    Az = (A - A.mean()) / A.std()
    Bz = (B - B.mean()) / B.std()
    
    # Compute absolute difference
    absdiff = np.abs(Az - Bz)
    
    # Invert into similarity metric
    return 1.0 - (absdiff / 3.0)


# Correlation and similarity profiles

def col_correlation_profile(
    df, cols_1, cols_2, 
    min_shift, max_shift,
    min_samples=5):
    """Compute time-shifted correlations between two (sets of) columns from a
    dataframe of tracking data, resulting in a temporal correlation profile.

    NOTE: Negative shifts ask how well the current time point's `cols_1` values 
          *are predicted by* past `cols_2` values, whereas positive shifts ask
          how well the current time point's `cols_1` values *predict* future
          `cols_2` values. That is, we're scanning over shifts in `cols_2`.
    
    Parameters
    ----------
    df : pandas df; index:'track', columns:(f, t, y, x, ...)
        The df containing the tracking data. Must be structured in the standard
        way expected in tracking_analysis pipelines.
    cols_1 : str, or list
        Column name(s) to use, e.g. 'v_mag' or ['vy', 'vx']. 
        If multiple columns are given, they are flattened and all values are
        treated as independent observations.
    cols_2 : str, or list of same length as cols_1
        Column name(s) to use, e.g. 'v_mag_itp' or ['vy_itp', 'vx_itp']. 
        If multiple columns are given, they are flattened and all values are
        treated as independent observations.
    min_shift : int
        How many frames "backwards" to calculate correlations.
    max_shift : int
        How many frames "forwards" to calculate correlations.
    min_samples : int, default 5
        The minimum number of matched samples required for a given shift to
        allow calculation of correlations. Cases with fewer samples yield nan.

    Returns
    -------
    shifted_corrs : pandas df; 
        index:'frame', columns:range(min_shift, max_shift+1)
            Pandas df containing the temporal correlation profiles ranging 
            backwards and forwards relative to each cols_1 tp.
    """
    
    # Prepare output array
    shifted_corrs = np.full(
        shape=(df['f'].nunique(), -min_shift+max_shift+1),
        fill_value=np.nan, dtype=float
    )
    
    # Prepare reduced and stacked df with unique colnames
    dfr = df.melt(id_vars='f', value_vars=cols_1, ignore_index=False)
    dfr['c2'] = df.melt(value_vars=cols_2, ignore_index=False)['value']
    dfr.columns = ['f', 'v', 'c1', 'c2']
    dfr.insert(0, 'track', dfr.index.copy())
    dfr.index = ['_'.join([str(t),v]) for t,v in zip(dfr.index, dfr['v'])]
    dfr = dfr.sort_values(['f', 'track'])  # For perf

    # For each pair of frames...
    dfr_by_frame = dfr.groupby("f")
    for i, (f1, group_1) in enumerate(dfr_by_frame):
        for f2, group_2 in dfr_by_frame:

            # If the second group is within profile range...
            shift = f2 - f1
            if (shift >= min_shift) and (shift <= max_shift):

                # Compute correlation on matched samples (if there's enough)
                shifted_corrs[i, shift-min_shift] = group_1['c1'].corr(
                    group_2['c2'], min_periods=min_samples)
    
    # Done
    return pd.DataFrame(
        shifted_corrs,
        index=sorted(df['f'].unique()), 
        columns=range(min_shift, max_shift+1)
        )


def vec_cosinesim_profile(
    df, vecs_1, vecs_2,
    min_shift, max_shift, 
    min_samples=5):
    """Compute time-shifted cosine similarity between two sets of vectors from 
    a dataframe of tracking data, resulting in a temporal similarity profile.

    NOTE: Negative shifts ask how similar the current time point's `vecs_1`
          vectors are to past `vecs_2` vectors, whereas positive shifts ask how
          similar the current time point's `vecs_1` vectors are to future
          `vecs_2` vectors. That is, we're scanning over shifts in `vecs_2`.
    
    TODO: Would be nice to improve performance here, but my several previous
          attempts only ever led to small gains at the cost of a big mess, so
          it is what it is for now. [low priority]
    
    TODO: This could be refactored into a more generic `vec_similarity_profile`
          function akin to `col_similarity_profile`, or the two could even be
          collapsed with vec vs col processing depending on a flag or on the
          selected similarity matrix. [low priority]
    
    Parameters
    ----------
    df : pandas df; index:'track', columns:(f, t, y, x, ...)
        The df containing the tracking data. Must be structured in the standard
        way expected in tracking_analysis pipelines.
    vecs_1 : list
        List of columns containing vector components, e.g. ['vy', 'vx'].
    vecs_2 : list
        List of columns containing vector components, e.g. ['vy_itp', 'vx_itp'].
    min_shift : int
        How many frames "backwards" to calculate similarities.
    max_shift : int
        How many frames "forwards" to calculate similarities.
    min_samples : int, default 5
        The minimum number of matched samples required for a given shift to
        allow calculation of similarities. Cases with fewer samples yield nan.

    Returns
    -------
    shifted_cossims : pandas df; 
        index:'track', columns:range(min_shift, max_shift+1)
            Pandas df containing the temporal similarity metrics (for each cell
            at each tp) ranging backwards and forwards relative to vecs_1 tps.
    shifted_cossims_mean : pandas df; 
        index:'frame', columns:range(min_shift, max_shift+1)
            Pandas df containing the temporal similarity profiles (averaged for
            each frame) ranging backwards and forwards relative to vecs_1 tps.
    """
    
    # Prepare output arrays, prefilled with NaNs
    shifted_cossims = np.full(
        shape=(df.shape[0], -min_shift+max_shift+1),
        fill_value=np.nan, dtype=float)
    shifted_cossims_mean = np.full(
        shape=(df['f'].nunique(), -min_shift+max_shift+1),
        fill_value=np.nan, dtype=float)
    
    # Ensure indexers are in list form...
    if isinstance(vecs_1, str): vecs_1 = [vecs_1]
    if isinstance(vecs_2, str): vecs_2 = [vecs_2]
    if isinstance(vecs_1, tuple): vecs_1 = list(vecs_1)
    if isinstance(vecs_2, tuple): vecs_2 = list(vecs_2)

    # Prepare reduced df (with integer index column)
    dfr = df.loc[:, ['f'] + vecs_1 + vecs_2]
    dfr.insert(1, 'iloc', np.arange(dfr.shape[0]))
    dfr = dfr.sort_values(['f', 'track'])  # For perf
    
    # For each pair of frames...
    dfr_by_frame = dfr.groupby("f")
    for f1, group_1 in dfr_by_frame:
        for f2, group_2 in dfr_by_frame:

            # If the second group is within profile range...
            shift = f2 - f1
            if (shift >= min_shift) and (shift <= max_shift):

                # Compute local similarities between matched cells
                merged = group_1[['iloc'] + vecs_1].merge(
                    group_2[vecs_2], how="inner", on="track")
                if merged.size==0:  # IN VIVO: Skip cases with no matched cells!
                    continue
                shifted_cossims[merged['iloc'], shift-min_shift] = (
                    1 - paired_cosine_distances(merged[vecs_1], merged[vecs_2])
                )

    # Convert to df
    shifted_cossims = pd.DataFrame(
        shifted_cossims,
        index=df.index,
        columns=range(min_shift, max_shift+1))

    # Generate average profiles
    shifted_cossims_by_frame = shifted_cossims.groupby(df['f'].values)
    n_counts = shifted_cossims_by_frame.count()
    shifted_cossims_mean = shifted_cossims_by_frame.mean()
    shifted_cossims_mean[n_counts.values < min_samples] = np.nan

    # Done
    return shifted_cossims, shifted_cossims_mean


def col_similarity_profile(
    df, cols_1, cols_2,
    min_shift, max_shift,
    metric="zscore_similarity",
    min_samples=5):
    """Compute time-shifted similarity metric between two (sets of) columns 
    from a df of tracking data, resulting in a temporal similarity profile.

    NOTE: Negative shifts ask how similar the current time point's `cols_1`
          data are to past `cols_2` data, whereas positive shifts ask how
          similar the current time point's `cols_1` data are to future `cols_2`
          vectors. That is, we're scanning over shifts in `cols_2`.
    
    Parameters
    ----------
    df : pandas df; index:'track', columns:(f, t, y, x, ...)
        The df containing the tracking data. Must be structured in the standard
        way expected in tracking_analysis pipelines.
    cols_1 : str, or list
        Column name(s) to use, e.g. 'v_mag' or ['vy', 'vx']. 
        If multiple columns are given, they are flattened and all values are
        treated as independent observations.
    cols_2 : str, or list of same length as cols_1
        Column name(s) to use, e.g. 'v_mag_itp' or ['vy_itp', 'vx_itp']. 
        If multiple columns are given, they are flattened and all values are
        treated as independent observations.
    min_shift : int
        How many frames "backwards" to calculate similarities.
    max_shift : int
        How many frames "forwards" to calculate similarities.
    metric : str or callable, default "zscore_similarity"
        The similarity metric to be calculated. If "zscore_similarity", the
        function `zscore_similarity` will be used (see there for more info).
        No other metrics are currently pre-implemented, but a callable can be
        passed instead, which must accept two 1D arrays of data of equal shape
        and must return a 1d array of pairwise similarities.
    min_samples : int, default 5
        The minimum number of matched samples required for a given shift to
        allow calculation of similarities. Cases with fewer samples yield nan.

    Returns
    -------
    shifted_sims : pandas df; 
        index:'track', columns:range(min_shift, max_shift+1)
            Pandas df containing the temporal similarity metrics (for each cell
            at each tp) ranging backwards and forwards relative to cols_1 tps.
    shifted_sims_mean : pandas df; 
        index:'frame', columns:range(min_shift, max_shift+1)
            Pandas df containing the temporal similarity profiles (averaged for
            each frame) ranging backwards and forwards relative to cols_1 tps.
    """

    # Select metric function
    if metric == "zscore_similarity":
        metric = zscore_similarity
    
    # Prepare output arrays, prefilled with NaNs
    shifted_sims = np.full(
        shape=(df.shape[0], -min_shift+max_shift+1),
        fill_value=np.nan, dtype=float)
    shifted_sims_mean = np.full(
        shape=(df['f'].nunique(), -min_shift+max_shift+1),
        fill_value=np.nan, dtype=float)
    
    # Prepare reduced and stacked df with unique colnames
    dfr = df.melt(id_vars='f', value_vars=cols_1, ignore_index=False)
    dfr['c2'] = df.melt(value_vars=cols_2, ignore_index=False)['value']
    dfr.columns = ['f', 'v', 'c1', 'c2']
    dfr.insert(0, 'track', dfr.index.copy())
    dfr.insert(1, 'iloc', np.arange(dfr.shape[0]))
    dfr.index = ['_'.join([str(t),v]) for t,v in zip(dfr.index, dfr['v'])]
    dfr = dfr.sort_values(['f', 'track'])  # For perf
    
    # For each pair of frames...
    dfr_by_frame = dfr.groupby("f")
    for f1, group_1 in dfr_by_frame:
        for f2, group_2 in dfr_by_frame:

            # If the second group is within profile range...
            shift = f2 - f1
            if (shift >= min_shift) and (shift <= max_shift):

                # Compute local similarities between matched cells
                merged = group_1[['track', 'iloc', 'c1']].merge(
                    group_2[['track', 'c2']], how="inner", on="track")
                shifted_sims[merged['iloc'], shift-min_shift] = (
                    metric(merged['c1'], merged['c2'])
                )
    
    # Convert to df
    shifted_sims = pd.DataFrame(
        shifted_sims,
        index=df.index,
        columns=range(min_shift, max_shift+1))

    # Generate average profiles
    shifted_sims_by_frame = shifted_sims.groupby(df['f'].values)
    n_counts = shifted_sims_by_frame.count()
    shifted_sims_mean = shifted_sims_by_frame.mean()
    shifted_sims_mean[n_counts.values < min_samples] = np.nan
    
    # Done
    return shifted_sims, shifted_sims_mean

