#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Computation tools to manipulate atmospheric variables.

import numpy as np
from dynamicopy import utils


def omega2w(omega, p, T):
    """Convert omega (?) to w (vertical velocity)

    Parameters
    ----------
    omega : np.ndarray
        The ? field
    p : np.ndarray
        Pressure coordinate
    T : np.ndarray
        Temperature field

    Returns
    -------
    np.ndarray
        The vertical velocity field
    """
    omega, p, T = np.array(omega), np.array(p), np.array(T)
    R_gas = 287.058
    g = 9.80665
    rho = p/(R_gas * T)
    w = - omega / (rho*g)
    return w


def hemispheric_mean(var, lat, axis=-1, neg=False):
    """Computes the mean over both hemispheres.

    If assymetric data is given, the mean is computed only on the common latitudes.

    Parameters
    ----------
    var : np.ndarray
        The field that we want to average
    lat : np.ndarray
        The latitude coordinate array
    axis : int, optional
        latitude axis in var, by default -1
    neg : bool, optional
        if True, apply a minus sign to the southern hemisphere (e.g. for vorticity), by default False

    Returns
    -------
    Two np.ndarrays
        The latitude array and the hemispheric mean
    """
    lat_south, var_south = utils.get_south(var, lat, axis, flip=True)
    lat_north, var_north = utils.get_north(var, lat, axis)

    common_lat = [l for l in abs(lat_south) if l in lat_north]
    mask_south = [-l in common_lat for l in lat_south]
    mask_north = [l in common_lat for l in lat_north]

    var_south = utils.apply_mask_axis(var_south, mask_south, axis)
    var_north = utils.apply_mask_axis(var_north, mask_north, axis)

    if neg:
        sign = -1
    else:
        sign = +1

    hemispheric_mean = var_north + sign*var_south

    return common_lat, hemispheric_mean


# Aliases
hemisphericMean = hemispheric_mean

if __name__ == "__main__":

    import dynamicopy.ncload as ncl
    sp = ncl.var_load("sp", "data_tests/sp.nc")
    lat = ncl.var_load("latitude", "data_tests/sp.nc")
    lat -= 17

    print(np.shape(sp))
    print(np.shape(hemispheric_mean(sp, lat, -2)[1]))
