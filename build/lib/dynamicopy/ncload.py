#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Tools to load variables from a NetCDF File.

import numpy as np
from netCDF4 import Dataset


def var_load(varname, file_path, group=None, subgroup=None, silent=True):
    """ Loads a variable from a netCDF file.

    Parameters
    ----------
    varname : str
        The name of the variable in the file;
    file_path : str
        The path (relative or absolute) to the netCDF file;
    group : str, optional
        Group in which the variable is in the netCDF file if necessary (default None);
    subgroup : str, optional
        Subgroup in which the variable is in the netCDF file if necessary (default None);
    silent : bool, optional
        Indicates whether to print infos about the loaded variable (default True);

    Returns
    -------
    np.ndarray
        A numpy array corresponding to the variable in the file;
    """

    if not silent:
        print("Loading " + varname + "...")

    f_in = Dataset(file_path)

    if group == None:
        var = f_in.variables[varname][:]
    else:
        if subgroup == None:
            var = f_in.groups[group].variables[varname][:]
        else:
            var = f_in.groups[group].groups[subgroup].variables[varname][:]
    f_in.close()

    if not(silent):
        print("Variable dimensions: "+str(np.shape(var)))
    return np.array(var)


def get_lon_lat(file_path, lon_name='lon', lat_name='lat'):
    """ Loads longitude and latitude coordinates from a netCDF file

    Parameters
    ----------
    file_path : str
        The path (relative or absolute) to the netCDF file;
    lon_name : str, optional
        name of the longitude coordinate in the file (default 'lon');
    lat_name : str, optional
        name of the latitude coordinate in the file (default 'lat');

    Returns
    -------
    list of np.ndarray
        Arrays of longitude and latitude in the netCDF file;
    """

    f_in = Dataset(file_path)
    lat = f_in[lat_name][:]
    lon = f_in[lon_name][:]
    f_in.close()

    return np.array(lon), np.array(lat)


# Aliases for old users
varLoad = var_load
getLonLat = get_lon_lat


if __name__ == "__main__":
    file = "../data_tests/sp.nc"
    sp = var_load("sp", file)
    lon, lat = get_lon_lat(file, 'longitude', 'latitude')
    lon_bis, lat_bis = getLonLat(file, 'longitude', 'latitude')

    print(lon, lat, sp)
