import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import seaborn as sns
import numpy as np


def ax_med(central_lon = (38 - 7) / 2, ex = [-7, 38, 28, 44, ]):
    ax = plt.axes(projection=ccrs.LambertAzimuthalEqualArea(central_lon))
    ax.coastlines()
    ax.set_extent(ex)
    ax.gridlines()
    return ax

# TODO : Ajouter les lignes ?
def plot_tracks(
    tracks,
    intensity_col=None,
    color=None,
    increasing_intensity=True,
    projection=None,
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
    import cartopy.crs as ccrs
    if projection == None:
        projection = ccrs.PlateCarree(central_longitude=180.0),
    # Plotting
    if fig_ax == None:
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=projection)
    else:
        fig, ax = fig_ax

    ax.coastlines()
    ax.gridlines(draw_labels=True)

    if intensity_col != None:
        if increasing_intensity:
            size_scale = (tracks[intensity_col] - np.nanmin(tracks[intensity_col])) / (
                np.nanmax(tracks[intensity_col]) - np.nanmin(tracks[intensity_col])
            )
        else:
            size_scale = (tracks[intensity_col] - np.nanmax(tracks[intensity_col])) / (
                np.nanmin(tracks[intensity_col]) - np.nanmax(tracks[intensity_col])
            )

        g = sns.scatterplot(
            data=tracks,
            x="lon",
            y="lat",
            hue=intensity_col,
            hue_norm=tuple(np.nanpercentile(tracks[intensity_col], [10, 90])),
            ax=ax,
            transform=ccrs.PlateCarree(),
            palette=cmap,
            size=size_scale,
            sizes=(4, 20),
        )
    else:
        g = sns.scatterplot(
            data=tracks,
            x="lon",
            y="lat",
            ax=ax,
            color=color,
            transform=ccrs.PlateCarree(),
        )
    h, l = g.get_legend_handles_labels()
    plt.legend(h[:-6], l[:-6])

    # plt.show()
    return fig, ax


def plot_tracks_med(
        tracks,
        intensity_col="wind10",
        projection=ccrs.PlateCarree(central_longitude=0.0),
        fig_ax=None,
        figsize=[12, 8],
        cmap='Spectral_r',
):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # Plotting
    if fig_ax == None:
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=projection)
    else:
        fig, ax = fig_ax

    ax.coastlines()
    ax.gridlines(draw_labels=True)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.set_extent([-20, 45, 25, 50], crs=ccrs.PlateCarree())

    for id in tracks.track_id.unique():
        track = tracks[tracks.track_id == id]
        x = track.lon
        y = track.lat
        c = track[intensity_col]

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(10, 25, c.max())
        lc = LineCollection(segments, cmap=cmap,
                            norm=norm, transform=ccrs.PlateCarree())

        lc.set_array(c)
        lc.set_linewidth(2.)
        line = ax.add_collection(lc)

    return fig, ax

def plot_polar(da):
    if np.max(da.az) > 2 * np.pi:
        da["az"] = da.az * np.pi / 180
    da.plot.pcolormesh("az", "r", subplot_kws=dict(projection="polar"))
    plt.show()
