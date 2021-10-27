# -*- coding: utf-8 -*-

"""
This module aims at allow quick diagnostic plots for tropical cyclones.
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import numpy as np
from matplotlib.colors import BoundaryNorm


def density_map(
    lons,
    lats,
    hist,
    ax=None,
    projection=ccrs.PlateCarree(central_longitude=180.0),
    cmap="Reds",
    vmax=None,
    shrink_cbar=1,
):
    # if ax == None:

    ax = plt.axes(projection=projection)
    p = ax.pcolormesh(
        lons,
        lats,
        hist.T,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        shading="nearest",
        vmin=0,
        vmax=vmax,
    )
    plt.colorbar(p, shrink=shrink_cbar)
    ax.coastlines()
    plt.show()
