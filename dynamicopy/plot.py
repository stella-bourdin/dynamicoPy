# -*- coding: utf-8 -*-

# Tools for ploting the data from NetCDF file, using only numpy and matplotlib libs.

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


def lon_lat_plot(lon, lat, var, lon_axis=-1, lat_axis=-2, fig_ax=plt.subplots(), title='', cmap="bwr", colorbar_label='', norm=TwoSlopeNorm(vcenter=0), smooth=False):
    """Plot a 2D map of the data

    Parameters
    ----------
    lon : 1D np.ndarray
        longitude coordinate
    lat : 1D np.ndarray
        latitude coordinate
    var : np.ndarray
        field to plot
    lon_axis : int, optional
        axis of longitude in var, by default -1
    lat_axis : int, optional
        axis of latitude in var, by default -2
    fig_ax : (matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot), optional
        2-tuple of fig and ax for the plot, by default plt.subplots()
    title : str, optional
        title of the plot, by default ''
    cmap : str, optional
        color palette, by default "bwr"
    colorbar_label : str, optional
        label of the colorbar, by default ''
    norm : matplotlib.colors.<any>Norm, optional
        normalization for the colormap, by default TwoSlopeNorm(vcenter=0)
    smooth : bool, optional
        if True, contourf is used instead of pcolormesh, by default False

    Returns
    -------
    None
        Plots the map in ax
    """
    # Obtain 2D variable to plot
    L = len(np.shape(var))
    if lon_axis < 0:
        lon_axis = L + lon_axis
    if lat_axis < 0:
        lat_axis = L + lat_axis
    axes_avg = list(range(L))
    axes_avg.remove(lon_axis)
    axes_avg.remove(lat_axis)

    var2D = np.mean(var, axis=tuple(axes_avg))

    # Plotting
    fig, ax = fig_ax
    if not smooth:
        C = ax.pcolormesh(lon, lat, var2D, cmap=cmap, norm=norm)
    else:
        C = ax.contourf(lon, lat, var2D, cmap=cmap, norm=norm)
    fig.colorbar(C, ax=ax, label=colorbar_label,)
    ax.set_title(title)

    # To-Do : Be able to set level when smoothed or even for pcolormesh (Check what I did for the matrix in my article)
    return None


lonLatPlot = lon_lat_plot


if __name__ == "__main__":

    import dynamicopy.ncload as ncl
    u = ncl.var_load("u10", "data_tests/u10.nc")  # [0]
    v = ncl.varLoad("v10", "data_tests/v10.nc")[0]
    lat = ncl.var_load("latitude", "data_tests/u10.nc")
    lon = ncl.varLoad("longitude", "data_tests/u10.nc")
    fig, axs = plt.subplots(2)
    lon_lat_plot(lon, lat, u, lon_axis=-1, lat_axis=-2, fig_ax=(fig, axs[0]),
                 smooth=False, colorbar_label="Velocity (m/s)")
    lon_lat_plot(lon, lat, u, lon_axis=-1, lat_axis=-2, fig_ax=(fig, axs[1]),
                 smooth=True, colorbar_label="Velocity (m/s)")
    # plt.show()
