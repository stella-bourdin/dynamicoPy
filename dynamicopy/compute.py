# -*- coding: utf-8 -*-

# Computation tools to manipulate atmospheric variables.

import numpy as np
import xarray as xr
from .utils_geo import get_south, get_north, apply_mask_axis


### ==================================== ###
###         Derivated variables          ###
### ==================================== ###


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
    rho = p / (R_gas * T)
    w = -omega / (rho * g)
    return w


# -- TODO -- Handle anyD velocity field.


def compute_vort(u, v, lat, lon):
    """Compute vorticity from the horizontal velocity fields

    ! So far, handles only 2D fields.

    Parameters
    ----------
    u : np.ndarray
        zonal wind field
    v : np.ndarrays
        meridional wind field
    lat : 1D np.ndarray
        latitude coordinate of the fields
    lon : 1D np.ndarray
        longitude coordinate of the field

    Returns
    -------
    3 np.ndarrays
        longitude, latitude and vorticity field
    """

    dlon = lon[1] - lon[0]  # resolution in longitude in deg
    dlat = lat[1] - lat[0]  # resolution in latitude in deg
    R = 6371000  # Earth radius
    dy = R * dlat * np.pi / 180  # Vertical size of a cell
    # Horizontal size of a cell depending on latitude
    lat_rad = lat * np.pi / 180  # Latitudes in rad
    r = np.sin(np.pi / 2 - abs(lat_rad)) * R
    dx = r * dlon * np.pi / 180
    dx = np.transpose([(dx[1:] + dx[:-1]) / 2] * (len(lon) - 1))

    # Compute lat and lon corresponding to the vort matrix
    lat_vort = (lat[:-1] + lat[1:]) / 2
    lon_vort = (lon[:-1] + lon[1:]) / 2

    W = np.zeros([len(lat) - 1, len(lon) - 1])  # Initialization of the matrix
    W = (v[:-1, 1:] - v[:-1, :-1]) * 1 / dx[:] - (u[1:, :-1] - u[:-1, :-1]) * 1 / dy

    return lon_vort, lat_vort, W


def get_dx_dy(lon, lat):
    dlon = lon[1] - lon[0]  # resolution in longitude in deg
    dlat = lat[1] - lat[0]  # resolution in latitude in deg
    R = 6371000  # Earth radius
    dy = R * dlat * np.pi / 180  # Vertical size of a cell
    # Horizontal size of a cell depending on latitude
    lat_rad = lat * np.pi / 180  # Latitudes in rad
    r = np.sin(np.pi / 2 - abs(lat_rad)) * R
    dx = r * dlon * np.pi / 180
    dx = np.transpose([(dx[1:] + dx[:-1]) / 2] * (len(lon) - 1))
    return dx, dy


def compute_stretching_xr(
    xarray, u_name="u", v_name="v", lon_name="lon", lat_name="lat"
):
    """Compute stretching deformation (E) from the horizontal velocity fields

    Parameters
    ----------
    xarray : xr.Dataset
        Dataset containing zonal and meridional wind in m/s
    u_name, v_name, lon_name, lat_name : str
        Names of variables in xarray, respectively zonal wind, meridional wind, longitude and latitude

    Returns
    -------
    E : xr.Dataset
        Stretching deformation in s-1
    """

    xarray = xarray.rename(
        {lat_name: "lat", lon_name: "lon", u_name: "u", v_name: "v"}
    ).squeeze()

    lon, lat = xarray.lon.values, xarray.lat.values
    dx, dy = get_dx_dy(xarray.lon.values, xarray.lat.values)

    u = xarray.u
    v = xarray.v
    E = (
        u.sel(lat=lat[:-1], lon=lon[1:]).values
        - u.sel(lat=lat[:-1], lon=lon[:-1]).values
    ) * 1 / dx[:] - (
        v.sel(lat=lat[1:], lon=lon[:-1]).values - u.sel(lat=lat[:-1], lon=lon[:-1])
    ) * 1 / dy
    E = xr.DataArray(E).interp_like(xarray, kwargs={"fill_value": "extrapolate"})
    E.attrs["units"] = "s-1"
    E = E.rename({"lat": lat_name, "lon": lon_name})
    return E


def compute_shearing_xr(xarray, u_name="u", v_name="v", lon_name="lon", lat_name="lat"):
    """Compute shearing deformation (F) from the horizontal velocity fields

    Parameters
    ----------
    xarray : xr.Dataset
        Dataset containing zonal and meridional wind in m/s
    u_name, v_name, lon_name, lat_name : str
        Names of variables in xarray, respectively zonal wind, meridional wind, longitude and latitude

    Returns
    -------
    F : xr.Dataset
        Shearing deformation in s-1
    """

    xarray = xarray.rename(
        {lat_name: "lat", lon_name: "lon", u_name: "u", v_name: "v"}
    ).squeeze()

    lon, lat = xarray.lon.values, xarray.lat.values
    dx, dy = get_dx_dy(xarray.lon.values, xarray.lat.values)

    u = xarray.u
    v = xarray.v
    F = (
        v.sel(lat=lat[:-1], lon=lon[1:]).values
        - u.sel(lat=lat[:-1], lon=lon[:-1]).values
    ) * 1 / dx[:] - (
        u.sel(lat=lat[1:], lon=lon[:-1]).values - u.sel(lat=lat[:-1], lon=lon[:-1])
    ) * 1 / dy
    F = xr.DataArray(F).interp_like(xarray, kwargs={"fill_value": "extrapolate"})
    F.attrs["units"] = "s-1"
    F = F.rename({"lat": lat_name, "lon": lon_name})
    return F


def compute_ObukoWeiss_norm_xr(vort, E, F):
    """Compute the normalized Obuko-Weiss Parameter

    Parameters
    ----------
    vort : xr.Dataset
        Vorticity field
    E : xr.Dataset
        Stretching deformation field
    F : xr.Dataset
        Shearing deformation field

    Returns
    -------
    xr.Dataset
        Normalized Obuko_Weiss parameter field
    """
    OW_n = 1 - ((E * 3600) ** 2 + (F * 3600) ** 2) / ((vort * 3600) ** 2)
    OW_n.attrs["units"] = "1"
    return OW_n


def compute_OWZ_xr(vort, E, F, lat_name="lat"):
    """Compute the Obuko-Weiss-Zeta Parameter (See Tory et al. 2013)

    Parameters
    ----------
    vort : xr.Dataset
        Vorticity field
    E : xr.Dataset
        Stretching deformation field
    F : xr.Dataset
        Shearing deformation field

    Returns
    -------
    xr.Dataset
        OWZ field
    """
    OW_n = compute_ObukoWeiss_norm_xr(vort, E, F)
    f = compute_Coriolis_param(OW_n[lat_name])
    zeros = xr.zeros_like(OW_n)
    OWZ = xr.ufuncs.maximum(OW_n, zeros) * xr.ufuncs.sign(f) * (vort + f)
    OWZ.attrs["units"] = "s-1"
    OWZ = OWZ.rename({'vo':'owz'})
    return OWZ

def compute_alts_xr(vo, E, F) :
    alt1 = (((E * 3600) ** 2 + (F * 3600) ** 2) / ((vo * 3600) ** 2)).rename({'vo':'alt1'})
    alt2 = (((vo * 3600) ** 2) / ((E * 3600) ** 2 + (F * 3600) ** 2)).rename({'vo':'alt2'})
    alt3 = (((vo * 3600) ** 2) - ((E * 3600) ** 2 + (F * 3600) ** 2)).rename({'vo':'alt3'})
    return xr.merge([alt1, alt2, alt3])


def compute_OWZ_from_files(
    u_file,
    v_file,
    vo_file=None,
    OWZ_file=None,
    owz_name='owz',
    u_name="u",
    v_name="v",
    vo_name="vo",
    lon_name="longitude",
    lat_name="latitude",
):
    """

    Parameters
    ----------
    u_file, v_file : str
        Paths to the respective files containing zonal and meridional wind field
    vo_file : str
        Path to the file containing the vorticity field. To be implemented : If None, compute vorticity from u and v.
    OWZ_file : str
        Path to which the OWZ field will be written. If None, not saved.
    owz_name : str
        Name of the owz to be written i the file if applicable.
    u_name, v_name, vo_name, lon_name, lat_name : str
        Names of zonal wind, meridional wind, vorticity, longitude, latitude respectively in the files.

    Returns
    -------
    OWZ : xr.Dataset
        OWZ field
    """
    if vo_file == None:
        # TODO : Implémenter compute_vort_xr et remplir cette partie pour calculer vo à partir de u et v
        pass
    else:
        vo = xr.open_dataset(vo_file).squeeze()
    u = xr.open_dataset(u_file).squeeze()
    v = xr.open_dataset(v_file).squeeze()
    wind = xr.merge([u, v])
    wind = wind.rename({u_name: "u", v_name: "v", lon_name: "lon", lat_name: "lat"})
    vo = vo.rename({vo_name: "vo", lon_name: "lon", lat_name: "lat"})

    E = compute_stretching_xr(wind)
    F = compute_shearing_xr(wind)

    OWZ = compute_OWZ_xr(vo, E, F)
    alts = compute_alts_xr(vo, E, F)
    OW_n = compute_ObukoWeiss_norm_xr(vo, E, F).rename({'vo':'OW_n'})
    OWZ = xr.merge([OWZ, alts, OW_n])
    OWZ = OWZ.rename({"lat": lat_name, "lon": lon_name, 'owz':owz_name})
    for n in OWZ.data_vars :
        OWZ[n] = OWZ[n].astype(np.float32)

    if OWZ_file != None:
        OWZ.to_netcdf(OWZ_file, format="NETCDF4_CLASSIC")
    return OWZ


def compute_stretching(u, v, lat, lon):
    """Compute stretching deformation (E) from the horizontal velocity fields

    ! So far, handles only 2D fields.

    Parameters
    ----------
    u : np.ndarray
        zonal wind field
    v : np.ndarrays
        meridional wind field
    lat : 1D np.ndarray
        latitude coordinate of the fields
    lon : 1D np.ndarray
        longitude coordinate of the field

    Returns
    -------
    3 np.ndarrays
        longitude, latitude and stretching field
    """

    if type(lon) != np.ndarray:
        lon = lon.values
    if type(lat) != np.ndarray:
        lat = lat.values
    dlon = lon[1] - lon[0]  # resolution in longitude in deg
    dlat = lat[1] - lat[0]  # resolution in latitude in deg
    R = 6371000  # Earth radius
    dy = R * dlat * np.pi / 180  # Vertical size of a cell
    # Horizontal size of a cell depending on latitude
    lat_rad = lat * np.pi / 180  # Latitudes in rad
    r = np.sin(np.pi / 2 - abs(lat_rad)) * R
    dx = r * dlon * np.pi / 180
    dx = np.transpose([(dx[1:] + dx[:-1]) / 2] * (len(lon) - 1))

    # Compute lat and lon corresponding to the E matrix
    lat_E = (lat[:-1] + lat[1:]) / 2
    lon_E = (lon[:-1] + lon[1:]) / 2

    # E = np.zeros([len(lat) - 1, len(lon) - 1])  # Initialization of the matrix
    if type(u) != np.ndarray:
        u = u.values
    if type(v) != np.ndarray:
        v = v.values
    E = (u[:-1, 1:] - u[:-1, :-1]) * 1 / dx[:] - (v[1:, :-1] - v[:-1, :-1]) * 1 / dy

    return lon_E, lat_E, E


def compute_shearing(u, v, lat, lon):
    """Compute shearing deformation (F) from the horizontal velocity fields

    ! So far, handles only 2D fields.

    Parameters
    ----------
    u : np.ndarray
        zonal wind field
    v : np.ndarrays
        meridional wind field
    lat : 1D np.ndarray
        latitude coordinate of the fields
    lon : 1D np.ndarray
        longitude coordinate of the field

    Returns
    -------
    3 np.ndarrays
        longitude, latitude and shearing field
    """

    dlon = lon[1] - lon[0]  # resolution in longitude in deg
    dlat = lat[1] - lat[0]  # resolution in latitude in deg
    R = 6371000  # Earth radius
    dy = R * dlat * np.pi / 180  # Vertical size of a cell
    # Horizontal size of a cell depending on latitude
    lat_rad = lat * np.pi / 180  # Latitudes in rad
    r = np.sin(np.pi / 2 - abs(lat_rad)) * R
    dx = r * dlon * np.pi / 180
    dx = np.transpose([(dx[1:] + dx[:-1]) / 2] * (len(lon) - 1))

    # Compute lat and lon corresponding to the F matrix
    lat_F = (lat[:-1] + lat[1:]) / 2
    lon_F = (lon[:-1] + lon[1:]) / 2

    F = np.zeros([len(lat) - 1, len(lon) - 1])  # Initialization of the matrix
    F = (v[:-1, 1:] - v[:-1, :-1]) * 1 / dx[:] + (u[1:, :-1] - u[:-1, :-1]) * 1 / dy

    return lon_F, lat_F, F


def compute_ObukoWeiss(vort, E, F):
    """Compute the Obuko-Weiss Parameter

    Parameters
    ----------
    vort : np.ndarray
        Vorticity field
    E : np.ndarray
        Stretching deformation field
    F : np.ndarray
        Shearing deformation field

    Returns
    -------
    np.ndarray
        The Obuko-Weiss (OW) parameter field
    """
    return vort ** 2 - (E ** 2 + F ** 2)


def compute_ObukoWeiss_norm(vort, E, F):
    """Compute the normalized Obuko-Weiss Parameter

    Parameters
    ----------
    vort : np.ndarray
        Vorticity field
    E : np.ndarray
        Stretching deformation field
    F : np.ndarray
        Shearing deformation field

    Returns
    -------
    np.ndarray
        The normalized Obuko-Weiss (OW) parameter field
    """
    OW = compute_ObukoWeiss(vort, E, F)
    return OW / vort ** 2


def compute_Coriolis_param(lat):
    """Compute the coriolis parameter for a given latitude (array)

    Parameters
    ----------
    lat : float
        latitude (can also be a np.ndarrays of latitudes)

    Returns
    -------
    float
        Coriolis parameter
    """
    W = 7.2921e-5  # Rotation rate of the Earth
    phi = lat * np.pi / 180
    return 2 * W * np.sin(phi)


def compute_OWZ(vort, E, F, lat):
    OW_n = compute_ObukoWeiss_norm(vort, E, F)
    f = compute_Coriolis_param(lat)
    shape = np.shape(vort)
    f = np.transpose(np.array(list(f) * shape[1]).reshape([shape[1], shape[0]]))
    return np.sign(f) * (vort + f) * np.maximum(OW_n, np.zeros(np.shape(OW_n)))


def compute_grad(T, lat, lon):
    """Compute the gradient of a 2D field.

    Parameters
    ----------
    T : 2D np.ndarray
        The field on which to compute the gradient
    lat : 1D np.ndarray
        latitude coordinate of the field
    lon : 1D np.ndarray
        longitude coordinate of the field

    Returns
    -------
    4 np.ndarrays
        longitude and latitude coordinates of the following x and y coordinates of the gradient.
    """
    dlon = lon[1] - lon[0]  # resolution in longitude in deg
    dlat = lat[1] - lat[0]  # resolution in latitude in deg
    R = 6371000  # Earth radius
    dy = R * dlat * np.pi / 180  #

    Gx = np.zeros([len(lat) - 1, len(lon) - 1])
    Gy = np.zeros([len(lat) - 1, len(lon) - 1])
    for i in range(len(lon) - 1):  # i index of longitude
        for j in range(len(lat) - 1):  # j index of latitude
            # Compute dx geometrically
            lat_rad = lat[j] * np.pi / 180  # Current latitude in rad
            # radius of the current longitude circle
            r = np.sin(np.pi / 2 - abs(lat_rad)) * R
            dx = r * dlon * np.pi / 180

            Gx[j, i] = (T[j + 1, i] - T[j, i]) / dx
            Gy[j, i] = (T[j, i + 1] - T[j, i]) / dy

    lat_G = np.array([(lat[j + 1] + lat[j]) / 2 for j in range(len(lat) - 1)])
    lon_G = np.array([(lon[i + 1] + lon[i]) / 2 for i in range(len(lon) - 1)])

    return lon_G, lat_G, np.array(Gx), np.array(Gy)


def compute_EKE(u, v):  # Probably deserves optimization if useful later.
    """Compute Eddy Kinetic Energy from u and v fields.

    Parameters
    ----------
    u : np.ndarray
        zonal wind field
    v : np.ndarray
        meridional wind field

    Returns
    -------
    np.ndarray
        EKE field
    """

    shape = np.shape(u)
    nlon = shape[-1]
    U_zonal = np.mean(u, -1)
    V_zonal = np.mean(v, -1)
    U_zonal_lon = np.transpose([np.transpose(U_zonal) for i in range(nlon)])
    V_zonal_lon = np.transpose([np.transpose(V_zonal) for i in range(nlon)])

    EKE = 0.5 * (
        np.multiply(u - U_zonal_lon, u - U_zonal_lon)
        + np.multiply(v - V_zonal_lon, v - V_zonal_lon)
    )

    return EKE


### ========================================== ###
###         Geographical computations          ###
### ========================================== ###


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
    lat_south, var_south = get_south(var, lat, axis, flip=True)
    lat_north, var_north = get_north(var, lat, axis)

    common_lat = [l for l in abs(lat_south) if l in lat_north]
    mask_south = [-l in common_lat for l in lat_south]
    mask_north = [l in common_lat for l in lat_north]

    var_south = apply_mask_axis(var_south, mask_south, axis)
    var_north = apply_mask_axis(var_north, mask_north, axis)

    if neg:
        sign = -1
    else:
        sign = +1

    hemispheric_mean = var_north + sign * var_south

    return common_lat, hemispheric_mean


# Aliases
hemisphericMean = hemispheric_mean

if __name__ == "__main__":
    pass
