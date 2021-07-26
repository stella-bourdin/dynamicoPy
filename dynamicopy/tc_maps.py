# -*- coding: utf-8 -*-

"""
This module aims at allow quick diagnostic plots for tropical cyclones.
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import seaborn as sns
import numpy as np

def plot_tracks(
    tracks,
    intensity_col="wind10",
    increasing_intensity=True,
    projection=ccrs.PlateCarree(central_longitude=180.0),
    fig_ax=None,
    figsize=[12, 8],
    cmap="autumn_r",
):
    """
    Function to plot tracks as issued from the cyclones module

    Parameters
    ----------
    cmap
    tracks: pandas DataFrame of tracks points
    id_col: name of the column in track with cyclones id
    id: id of the track(s) to plot
    projection: ccrs projection to use for the plot
    intensity_col: name of the column in tracks to use as color

    Returns
    -------
    None. Produces a plot.
    """

    # Plotting
    if fig_ax == None:
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=projection)
    else:
        fig, ax = fig_ax

    ax.coastlines()
    ax.gridlines(draw_labels=False)

    if increasing_intensity :
        size_scale = (tracks[intensity_col] - tracks[intensity_col].min()) / (tracks[intensity_col].max() - tracks[intensity_col].min())
    else :
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
    h,l = g.get_legend_handles_labels()
    plt.legend(h[:-6], l[:-6])
    return fig, ax
