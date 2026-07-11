# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 23:04:55 2023

@author:    Jonas Hartmann @ Mayor lab @ UCL CDB

@descript:  Tools for operating on data values defined on a point cloud.
"""


# Imports
import numpy as np


def pcl_gaussian_smooth(dists, vals, sigma=1.0, sg_percentile=False):
    """Perform a Gaussian smooth on values defined on the points of a point
    cloud with an arbitrary number of dimensions.
    
    Parameters
    ----------
    dists : ndarray of shape (n_pts, n_pts)
        Squareform of all pairwise distances between points of the point
        cloud, as computed e.g. by `scipy.spatial.distance.pdist`.
    vals : ndarray of shape (n_pts, n_dims)
        Values to be smoothed defined on each point. The different dimensions
        are smoothed independently.
    sigma : numeric, default 1.0
        If `sg_percentile` is False (default), the Gaussian uses this sigma.
        If `sg_percentile` is True, the sigma is instead determined as the X-th
        percentile of all distances in `dists`, where this value is X.
    sg_percentile : bool, default False
        Decide whether `sigma` is used directly or treated as a percentile.
        See `sigma` for more information.
        
    Returns
    -------
    smooth_vals : ndarray of shape (n_pts, n_dims)
        Smoothed values defined on each point.
        
    See also
    --------
    pcl_gaussian_interp : Function for interpolation of values defined on one
        point based on neighbors in another.
    """
    
    # Get sigma from dists (if needed)
    if sg_percentile:
        sigma = np.percentile(dists, sigma)
    
    # Generate Gaussian function for smoothing
    def gaussian_factory(mu, sg):
        gaussian = lambda x : 1 / (sg*np.sqrt(2.0*np.pi)) * np.exp(-1/2*((x-mu)/sg)**2.0)
        return gaussian
    gaussian_func = gaussian_factory(0.0, sigma)

    # Smoothen the distances
    gaussian_dists = gaud = gaussian_func(dists)

    # Use smoothened distances to smoothen values
    smooth_vals = np.empty_like(vals)
    for dim in range(vals.shape[1]):
        smooth_vals[:,dim] = np.sum(gaud*vals[:,dim], axis=1) / np.sum(gaud, axis=1)
    
    # Done
    return smooth_vals


def pcl_gaussian_interp(dists, vals, sigma=1.0, sg_percentile=False):
    """Given a point cloud that has a value associated with each point (pcl1)
    and a point cloud lacking this value (pcl2), interpolate the values for 
    pcl2 as the weighted local average of the values in pcl1, with the weight
    being a Gaussian on the distances.
    
    Parameters
    ----------
    dists : ndarray of shape (pcl1.shape[0], pcl2.shape[0])
        Mutual distances between each combination of points from the two point
        clouds. Can be computed as `dists = dist.cdist(pcl1, pcl2)`, where the
        shape of the point clouds is `(points, coordinates)`.
    vals : ndarray of shape (pcl1.shape[0], n_dims)
        Values defined on points in pcl1, to be interpolated for pcl2. The
        different dimensions are interpolated independently.
    sigma : numeric, default 1.0
        If `sg_percentile` is False (default), the Gaussian uses this sigma.
        If `sg_percentile` is True, the sigma is instead determined as the X-th
        percentile of all distances in `dists`, where this value is X.
    sg_percentile : bool, default False
        Decide whether `sigma` is used directly or treated as a percentile.
        See `sigma` for more information.

    Returns
    -------
    interp_vals : ndarray of shape (pcl2.shape[0], n_dims)
        Interpolated values for pcl2.
        
    See also
    --------
    pcl_gaussian_smooth : Function for Gaussian smoothing of values associated 
        with a single point cloud based on neighborhood.
    """
    
    # Get sigma from dists (if needed)
    if sg_percentile:
        sigma = np.percentile(dists, sigma)
    
    # Generate Gaussian function for smoothing
    def gaussian_factory(mu, sg):
        gaussian = lambda x : 1 / (sg*np.sqrt(2.0*np.pi)) * np.exp(-1/2*((x-mu)/sg)**2.0)
        return gaussian
    gaussian_func = gaussian_factory(0.0, sigma)

    # Smoothen the distances
    gaussian_dists = gaud = gaussian_func(dists)
    
    # Use smoothened distances to interpolate values
    interp_vals = np.empty((dists.shape[1], vals.shape[1]))
    for dim in range(vals.shape[1]):
        interp_vals[:,dim] = np.sum(gaud.T*vals[:,dim], axis=1) / np.sum(gaud, axis=0)
    
    # Done
    return interp_vals
    

def pcl_local_density(
    dists, sigma=1.0, sg_percentile=None, 
    class_labels=None):
    """Use a Gaussian kernel to measure local density in a point cloud,
    optionally class-wise based on labels.
    
    Parameters
    ----------
    dists : ndarray of shape (n_pts, n_pts)
        Squareform of all pairwise distances between points of the point
        cloud, as computed e.g. by `scipy.spatial.distance.pdist`.
    sigma : numeric, default 1.0
        If `sg_percentile` is None (default), the Gaussian uses this sigma.
        If `sg_percentile` is True, the sigma is instead determined as the X-th
        percentile of all distances in `dists`, where this value is X.
    sg_percentile : bool, default None
        Decide whether `sigma` is used directly or treated as a percentile.
        See `sigma` for more information.
    sigma : numeric, default 1.0
        The sigma of the Gaussian.
    class_labels : ndarray of shape (n_pts,), dtype int
        Optionally gives an integer class label to every point in the cloud.
        If not None, local densities at each point are calculated considering
        only other points in the same class.
        
    Returns
    -------
    local_density : ndarray of shape (n_pts, )
        Local density defined on each point.
    """
    
    # Get sigma from dists (if needed)
    if sg_percentile is not None:
        sigma = np.percentile(dists, sigma)
    
    # Generate Gaussian function for smoothing
    def gaussian_factory(mu, sg):
        gaussian = lambda x : 1 / (sg*np.sqrt(2.0*np.pi)) * np.exp(-1/2*((x-mu)/sg)**2.0)
        return gaussian
    gaussian_func = gaussian_factory(0.0, sigma)

    # Apply Gaussian to distances
    gaussian_dists = gaud = gaussian_func(dists)
    
    # Get and return Gaussian local density
    local_density = np.sum(gaud, axis=0)
    
    # Get local class count
    if class_labels is not None:
    
        # Get number of locally relevant classes
        local_class_count = np.array(
            [gaussian_dists[:, class_labels==ci].max(axis=1)
             for ci in np.unique(class_labels)]
        ).sum(axis=0)
        
        # Return class count-normalized local density
        return local_density / local_class_count, local_density, local_class_count
    
    # Return standard local density
    return local_density

