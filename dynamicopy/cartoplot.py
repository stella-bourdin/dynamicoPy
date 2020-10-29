# -*- coding: utf-8 -*-

# Tools for ploting the data from NetCDF file, using only numpy and matplotlib libs.

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import cartopy.crs as ccrs
from plot import var2d

from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh


def lon_lat_plot_map(lon, lat, var, lon_axis=-1, lat_axis=-2, fig_ax=plt.subplots(), title='', cmap="bwr", colorbar_label='', norm=TwoSlopeNorm(vcenter=0), smooth=False, projection=ccrs.Robinson(), set_global=False):
    """Plot a 2D map of the data with cartopy.

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
        axis on latitude in var, by default -2
    fig_ax : [type], optional
        [description], by default plt.subplots()
    title : str, optional
        title of the plot, by default ''
    cmap : str, optional
        color palette, by default "bwr"
    colorbar_label : str, optional
        label to be shown beside the colorbar, by default ''
    norm : matplotlib.colors.<any>Norm, optional
        normalization of the colormap, by default TwoSlopeNorm(vcenter=0)
    smooth : bool, optional
        if True, contourf is used instead of pcolormesh, by default False
    projection : cartopy.crs projection, optional
        projection for the display, by default ccrs.Robinson()
    set_global : bool, optional
        if True, forces to plot the full globe, by default False

    Returns
    -------
    None
        Plots the map in ax
    """
    # Obtain 2D variable to plot
    var2D = var2d(var, lon_axis, lat_axis)

    # Plotting
    fig, ax = fig_ax
    ax = plt.axes(projection=projection)
    if set_global:
        ax.set_global()
    ax.coastlines()

    if not smooth:
        C = ax.pcolormesh(lon, lat, var2D, cmap=cmap,
                          norm=norm, transform=ccrs.PlateCarree())
    else:
        C = ax.contourf(lon, lat, var2D, cmap=cmap, norm=norm,
                        transform=ccrs.PlateCarree())
    fig.colorbar(C, ax=ax, label=colorbar_label,)
    ax.set_ylabel("Latitude (°)")
    ax.set_xlabel("Longitude (°)")
    ax.set_title(title)

    return None


if __name__ == "__main__":

    import dynamicopy.ncload as ncl
    u = ncl.var_load("u10", "data_tests/u10.nc")  # [0]
    v = ncl.varLoad("v10", "data_tests/v10.nc")[0]
    lat = ncl.var_load("latitude", "data_tests/u10.nc")
    lon = ncl.varLoad("longitude", "data_tests/u10.nc")
    #fig, axs = plt.subplots(2)
    lon_lat_plot_map(lon, lat, u, lon_axis=-1, lat_axis=-2,  # fig_ax=(fig, axs[0]),
                     smooth=False, colorbar_label="Velocity (m/s)")
    plt.figure()
    lon_lat_plot_map(lon, lat, u, lon_axis=-1, lat_axis=-2,  # fig_ax=(fig, axs[1]),
                     smooth=True, colorbar_label="Velocity (m/s)")
    # plt.show()
