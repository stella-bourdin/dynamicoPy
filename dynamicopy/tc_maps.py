# -*- coding: utf-8 -*-

"""
This module aims at allow quick diagnostic plots for tropical cyclones.
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import seaborn as sns
import numpy as np
from matplotlib.colors import BoundaryNorm

#TODO : Ajouter les lignes ?
def plot_tracks(
    tracks,
    intensity_col=None,
    color=None,
    increasing_intensity=True,
    projection=ccrs.PlateCarree(central_longitude=180.0),
    fig_ax=None,
    figsize=[12, 8],
    cmap="autumn_r",
):
    """
    Plot tracks from a dataset onto the globe.
    The dataset is assumed loaded with functions of the cyclones module, thus with tis format.

    Parameters
    ----------
    tracks: The TC dataset in the format of the cyclones module.
    intensity_col: Name of the intensity column to plot.
    increasing_intensity: Is intensity increasing with increasing values of intensity_col?
    projection: cartopy.crs projection to use for the plot.
    fig_ax: (fig, ax) couple to use for the plot. If None, a new plot will be created.
    figsize: Size of the new plot if fig_ax = None.
    cmap: Name of the palette to use for intensity.

    Returns
    -------
    A map of the tracks.
    """

    # Plotting
    if fig_ax == None:
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=projection)
    else:
        fig, ax = fig_ax

    ax.coastlines()
    ax.gridlines(draw_labels=False)

    if intensity_col != None :
        if increasing_intensity:
            size_scale = (tracks[intensity_col] - tracks[intensity_col].min()) / (
                        tracks[intensity_col].max() - tracks[intensity_col].min())
        else:
            size_scale = (tracks[intensity_col] - tracks[intensity_col].max()) / (
                    tracks[intensity_col].min() - tracks[intensity_col].max())

        g = sns.scatterplot(
            data=tracks,
            x="lon",
            y="lat",
            hue=intensity_col,
            hue_norm=tuple(np.percentile(tracks[intensity_col], [10,90])),
            ax=ax,
            transform=ccrs.PlateCarree(),
            palette=cmap,
            size=size_scale,
            sizes=(4, 20),
        )
    else :
        g = sns.scatterplot(
            data=tracks,
            x="lon",
            y="lat",
            ax=ax,
            color=color,
            transform=ccrs.PlateCarree(),
        )
    h,l = g.get_legend_handles_labels()
    plt.legend(h[:-6], l[:-6])
    return fig, ax

def density_map(lons, lats, hist,
                ax=None,
                projection = ccrs.PlateCarree(central_longitude=180.),
                cmap = "Reds",
                vmax=None,
                shrink_cbar = 1):
    #if ax == None:

    ax = plt.axes(projection=projection)
    p = ax.pcolormesh(lons, lats, hist.T, transform=ccrs.PlateCarree(), cmap=cmap, shading = "nearest",
                      vmin=0, vmax=vmax)
    plt.colorbar(p, shrink = shrink_cbar)
    ax.coastlines()
    plt.show()
