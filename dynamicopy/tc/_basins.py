# -*- coding: utf-8 -*-

"""
Oceanic basins as defined by Knutson et al. 2020 appendix.
"""

from shapely.geometry import Polygon, MultiPolygon, Point
import matplotlib.pyplot as plt
import pkg_resources
import dynamicopy
import xarray as xr

try:
    import cartopy.crs as ccrs
    import geopandas as gpd
except ImportError:
    pass
import numpy as np
import matplotlib.ticker as mticker

NATL = Polygon(((260, 90), (360, 90), (360, 0), (295, 0), (260, 20)))
ENP = Polygon(((220, 90), (260, 90), (260, 20), (295, 0), (220, 0)))
CP = Polygon(((180, 0), (180, 90), (220, 90), (220, 0)))
WNP = Polygon(((100, 0), (100, 90), (180, 90), (180, 0)))
NI = Polygon(((30, 0), (30, 90), (100, 90), (100, 0)))
MED = Polygon(((0, 0), (0, 90), (30, 90), (30, 0)))
NH = {"NATL": NATL, "ENP": ENP, "CP":CP, "WNP": WNP, "NI": NI, "MED": MED}

SI = Polygon(((20, -90), (20, 0), (90, 0), (90, -90)))
AUS = Polygon(((90, -90), (90, 0), (160, 0), (160, -90)))
SP = Polygon(((160, 0), (160, -90), (295, -90), (295, 0)))
SA1 = Polygon(((295, -90), (295, 0), (360, 0), (360, -90)))
SA2 = Polygon(((20, -90), (20, 0), (0, 0), (0, -90)))
SA = MultiPolygon([SA1, SA2])
SA_plot = Polygon(((295, -90), (295, 0), (380, 0), (380, -90)))
SH = {"SI": SI, "AUS" : AUS, "SP": SP, "SA": SA}

basins = dict(SH, **NH)


def _save_basins_shapefile():
    """
    Code from Stackoverflow :
    https://gis.stackexchange.com/questions/52705/how-to-write-shapely-geometries-to-shapefiles
    """
    from osgeo import ogr
    # Now convert it to a shapefile with OGR
    driver = ogr.GetDriverByName("Esri Shapefile")
    ds = driver.CreateDataSource("dynamicopy/_data/basins.shp")
    layer = ds.CreateLayer("", None, ogr.wkbPolygon)
    # Add one attribute
    layer.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))
    defn = layer.GetLayerDefn()

    for b in basins:
        # Create a new feature (attribute and geometry)
        feat = ogr.Feature(defn)
        feat.SetField("id", 123)

        # Make a geometry, from Shapely object
        geom = ogr.CreateGeometryFromWkb(basins[b].wkb)
        feat.SetGeometry(geom)

        layer.CreateFeature(feat)
        feat = geom = None  # destroy these

    # Save and close everything
    ds = layer = feat = geom = None


def plot_basins(show=True, save=None, fig_ax = None, text = True, coastcolor = "grey", ):
    """
    Plot the basins according to the Knutson definition

    Parameters
    ----------
    show: Use plt.show in the end to display the figure
    save: Name of the file to save. If None, no file is saved.

    Returns
    -------
    A plot.
    """

    if fig_ax == None :
        fig, ax = plt.subplots(
            subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)}
        )
        ax.coastlines(color = coastcolor)
        ax.set_global()
        gl = ax.gridlines(draw_labels=True)
        gl.xlocator = mticker.FixedLocator([20, 30, 100, 180, -160, -100, -65])
        gl.ylocator = mticker.FixedLocator([-90, 0, 20, 90])
        gl.xlines = False
        gl.ylines = False
    else :
        fig, ax = fig_ax

    for basin, name in zip(
        [NATL, ENP, CP, WNP, NI , SI, AUS, SP, SA_plot],
        ["NATL", "ENP", "CP", "WNP", "NI", "SI", "AUS", "SP", "SATL"],
    ):
        ax.plot(
            basin.exterior.xy[0],
            basin.exterior.xy[1],
            transform=ccrs.PlateCarree(),
            color="k",
        )
        if text :
            ax.text(
                basin.centroid.x - 10,
                np.sign(basin.centroid.y) * 25,
                name,
                transform=ccrs.PlateCarree(),
                fontweight="bold",
            )
    if show:
        plt.show()
    if save != None:
        print("Saving in "+save)
        plt.show()
        plt.savefig(save)


# TODO (?) : Définir aussi les régions WMO

def list_in_med(lon, lat, path = dynamicopy.__file__[:-11] + "_data/med_mask.nc"):
    mask = xr.open_dataset(path).load().lsm
    return [point_in_med(lon[i], lat[i], mask) for i in range(len(lon))]

def point_in_med(lon, lat, mask):
    mask_value = mask.sel(longitude = lon, latitude = lat, method = "nearest")
    return int(mask_value.values)