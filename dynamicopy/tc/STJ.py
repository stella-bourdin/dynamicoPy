import xarray as xr
import numpy as np
import pandas as pd


def compute_STJ_latmin(file_ws, file_u, outfile):
    """

    Parameters
    ----------
    outfile: prefix of the output file, to which will be appended _[N/S]H.nc
    file_ws: File of wind speed THAT HAS ALREADY GONE THROUGH ROLLING AVG WITH cdo -runmean
    file_u: File of zonal wind speed THAT HAS ALREADY GONE THROUGH ROLLING AVG WITH cdo -runmean

    *** History of the two files, in the preliminary job ***
    ufile=$FOLDER/ua/gr/files/*/ua_6hrPlevPt_*_highresSST-present_r1i1p1f1_gr_${yr_min}01010600-*.nc
    vfile=$FOLDER/va/gr/files/*/va_6hrPlevPt_*_highresSST-present_r1i1p1f1_gr_${yr_min}01010600-*.nc

    ncks -d plev,25000.0 -d lat,-60.0,60.0 $ufile $TMP/u200_${NAME}_${yr_min}.nc
    ncks -d plev,25000.0 -d lat,-60.0,60.0 $vfile $TMP/v200_${NAME}_${yr_min}.nc

    cdo merge $TMP/u200_${NAME}_${yr_min}.nc $TMP/v200_${NAME}_${yr_min}.nc $TMP/winds200_${NAME}_${yr_min}.nc

    ncap2 -s 'u200_abs=abs(ua)' $TMP/u200_${NAME}_${yr_min}.nc $TMP/u_abs_${NAME}_${yr_min}.nc
    ncap2 -s 'ws200=sqrt(ua^2+va^2)' $TMP/winds200_${NAME}_${yr_min}.nc $TMP/ws_${NAME}_${yr_min}.nc

    cdo -runmean,120 $TMP/u_abs_${NAME}_${yr_min}.nc $TMP/u200_runmean_${NAME}_${yr_min}.nc   # --> This is u_file
    cdo -runmean,120 $TMP/ws_${NAME}_${yr_min}.nc $TMP/ws200_runmean_${NAME}_${yr_min}.nc     # --> This is ws_file

    Returns
    -------

    """

    speed_roll = xr.open_dataset(file_ws).ws200.squeeze()
    u_roll = xr.open_dataset(file_u).u200_abs.squeeze()

    lat_decrease = (u_roll.lat[0] > u_roll.lat[1])

    NH_STJ = ((speed_roll.where(speed_roll.lat >= 0) > 25) & (u_roll.where(u_roll.lat >= 0) > 15))
    if lat_decrease : NH_STJ = NH_STJ.reindex(lat = NH_STJ.lat[::-1]);

    NH_lim = NH_STJ.idxmax("lat")
    NH_lim = NH_lim.where(NH_lim > 0)
    NH_lim["lon"] = np.where(NH_lim.lon > 180, NH_lim.lon - 360, NH_lim.lon)
    NH_lim = NH_lim.sortby("lon")

    NH_lim.to_netcdf(outfile + "_NH.nc", "w")

    SH_STJ = ((speed_roll.where(speed_roll.lat <= 0) > 25) & (u_roll.where(u_roll.lat <= 0) > 15))
    if ~lat_decrease : SH_STJ = SH_STJ.reindex(lat=SH_STJ.lat[::-1]);

    SH_lim = SH_STJ.idxmax("lat")
    SH_lim = SH_lim.where(SH_lim < 0)
    SH_lim["lon"] = np.where(SH_lim.lon > 180, SH_lim.lon - 360, SH_lim.lon)
    SH_lim = SH_lim.sortby("lon")

    SH_lim.to_netcdf(outfile + "_SH.nc", "w")

    return NH_lim, SH_lim

def identify_ET(tracks, NH_lim, SH_lim, lon_name="longitude", minus_3h=True, fill=True):
    """
    Parameters
    ----------
    tracks (pd.Dataframe) : The dataframe containing the tracks data
    NH_lim (xr.DataArray) : The xarray DataArray with the latitude of the NH STJ as a function of lon and time
    SH_lim (xr.DataArray) : The xarray DataArray with the latitude of the SH STJ as a function of lon and time
    lon_name (str) : The name of the longitude coordinate in the NH_lim and SH_lim objects
    minus_3h (bool) : Set to True if you need to offset the NH_lim and SH_lim objects by 3 hours
    fill (bool) : When set to True, all point in a track after the detection of one ET point are considered ET.

    -------
    pd.Dataframe
        tracks with the ET column
    """

    # Pre-treat latitude limits
    ## Fill NAs in the latitude limits with linear interpolation
    NH_lim = NH_lim.interpolate_na(dim=lon_name).interpolate_na(dim="time")
    SH_lim = SH_lim.interpolate_na(dim=lon_name).interpolate_na(dim="time")
    ## Change time to -3h to fit with tracks
    if minus_3h:
        NH_lim["time"] = NH_lim.time - np.timedelta64(3, "h")
        SH_lim["time"] = SH_lim.time - np.timedelta64(3, "h")
    NH_lim = NH_lim.rename({lon_name: "longitude"})
    SH_lim = SH_lim.rename({lon_name: "longitude"})

    # Pre-treat tracks
    tracks = tracks.reset_index(drop=True)
    # tracks["lon"] = np.floor(tracks.lon * 4)/4
    tracks["lon"] = np.where(tracks.lon > 180, tracks.lon - 360, tracks.lon)

    # Detect ET points
    target_lon = xr.DataArray(tracks.lon, dims="points")
    target_time = xr.DataArray(tracks.time, dims="points")
    tracks["lat_STJ_NH"] = NH_lim.sel(
        time=target_time, longitude=target_lon, method="nearest"
    )
    tracks["lat_STJ_SH"] = SH_lim.sel(
        time=target_time, longitude=target_lon, method="nearest"
    )
    tracks["ET"] = (tracks.lat > tracks.lat_STJ_NH) | (tracks.lat < tracks.lat_STJ_SH)

    if fill:
        # Fill trajectories once one point is ET
        first_times = (
            tracks[tracks.ET == True]
            .sort_values("time")
            .groupby("track_id")
            .time.first()
        )
        t_ET = tracks[
            tracks.track_id.isin(first_times.index)
        ]  # Track that are labelled ET at one point
        t_noET = tracks[
            ~tracks.track_id.isin(first_times.index)
        ]  # Track that are not labelled ET at one point
        t_ET = t_ET.merge(first_times, on="track_id")
        t_ET["ET"] = t_ET.time_x >= t_ET.time_y
        t_ET = t_ET.rename(columns={"time_x": "time"}).drop(columns="time_y")
        tracks = pd.concat([t_ET, t_noET])

    return tracks