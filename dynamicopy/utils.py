#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Utilitary functions to use inside the module

import numpy as np

### ==================================================== ###
###          VARIABLES GEOGRAPHIC MANIPULATIONS          ###
### ==================================================== ###


def apply_mask_axis(var, mask, axis=-1):
    """Apply a mask on a field on the given axis.

    Parameters
    ----------
    var : np.ndarray
        The field on which to apply the mask
    mask : 1D np.ndarray
        The boolean mask (dimension must fit the dimension of var for the axis)
    axis : int, optional
        axis on which to perform the operation, by default -1

    Returns
    -------
    np.ndarray
        The field but apply for the desired region.
    """
    # Transposition of var so that the lat dimension is the first one
    L = len(np.shape(var))
    if axis < 0:
        axis = L + axis
    axes = np.arange(L)
    axes[0] = axis
    axes[1:axis+1] -= 1
    var_t = np.transpose(var, axes)

    # Apply the mask
    var_t_cut = var_t[mask]

    # Reverse the transposition
    axes = np.arange(L)
    axes[0:axis] = axes[1:axis+1]
    axes[axis] = 0
    var_cut = np.transpose(var_t_cut, axes)

    return var_cut

# ------------- North/South division -------------#


def get_south(var, lat, axis=-2, flip=False):
    """Returns the southern part of a given field.

    Parameters
    ----------
    var : np.ndarray
        The field whose southern part we want
    lat : np.ndarray        
        latitude coordinate 
    axis : int, optional
        latitude axis in var, by default -2
    flip : bool, option
        if True, reverse the data along the latitude index (so that abs(lat) is increasing)

    Returns
    -------
    list of np.ndarray
        Southern part of the field and the corrsponding latitude coordinate array
    """

    mask = lat <= 0.
    lat_south = lat[mask]

    var_south = apply_mask_axis(var, mask, axis)

    if flip:
        var_south = np.flip(var_south, axis)
        lat_south = np.flip(lat_south)

    return lat_south, var_south


def get_north(var, lat, axis=-2, flip=False):
    """Returns the northern part of a given field.

    Parameters
    ----------
    var : np.ndarray
        The field whose northern part we want
    lat : np.ndarray        
        latitude coordinate 
    axis : int, optional
        latitude axis in var, by default -2
    flip : bool, option
        if True, reverse the data along the latitude index (so that abs(lat) is increasing)


    Returns
    -------
    list of np.ndarray
        northern part of the field and the corrsponding latitude coordinate array
    """

    mask = lat >= 0.
    lat_north = lat[mask]

    var_north = apply_mask_axis(var, mask, axis)

    if flip:
        var_north = np.flip(var_north, axis)
        lat_north = np.flip(lat_north)

    return lat_north, var_north

# -------       Basin selection     -------- #


def select_box_indices(var, idx_lat_min, idx_lat_max, idx_lon_min=None, idx_lon_max=None, lat_axis=-2, lon_axis=-1):
    """Extract data in a box from a field. The box is defined by its indices.

    Parameters
    ----------
    var : np.ndarray
        The field in which we exract the box
    idx_lat_min : int >= 0
        index of the minimum latitude   
    idx_lat_max : int >= 0
        index of the maximum latitude
    idx_lon_min : int >= 0, optional
        index of the minimum longitude, by default None
    idx_lon_max : int >= 0, optional
        index of the maximum longitude, by default None
    lat_axis : int, optional
        latitude axis in var, by default -2
    lon_axis : int, optional
        longitude axis in var, by default -1

    Returns
    -------
    np.ndarray
        The field in the box.
    """

    # TODO : Traiter le cas où idx_lon_min>idx_lon_max
    assert(idx_lon_min < idx_lon_max,
           "Cette possibilité n'est pas encore implémentée")

    # Masks
    lat_mask = np.array([False] * np.shape(var)[lat_axis])
    lat_mask[idx_lat_min:idx_lat_max] = True
    var_cut_lat = apply_mask_axis(var, lat_mask, lat_axis)

    lon_mask = np.array([False] * np.shape(var)[lon_axis])
    lon_mask[idx_lon_min:idx_lon_max] = True
    var_box = apply_mask_axis(var_cut_lat, lon_mask, lon_axis)

    return var_box


if __name__ == "__main__":

    import dynamicopy.ncload as ncl
    sp = ncl.var_load("sp", "data_tests/sp.nc")
    lat = ncl.var_load("latitude", "data_tests/sp.nc")
    lon = ncl.var_load("longitude", "data_tests/sp.nc")

    print(np.shape(sp))
