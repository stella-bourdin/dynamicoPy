import numpy as np
import xarray as xr


def identify_ET(tracks, NH_lim, SH_lim, lon_name="longitude", minus_3h=True, fill=True):
    """

    Parameters
    ----------
    tracks (pd.Dataframe)
    NH_lim (xr.DataArray)
    SH_lim (xr.DataArray)

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

    return tracks


def identify_ET_new(tracks, NH_lim, SH_lim, lon_name="longitude", fill=True):
    """

    Parameters
    ----------
    tracks (pd.Dataframe)
    NH_lim (xr.DataArray)
    SH_lim (xr.DataArray)

    -------
    pd.Dataframe
        tracks with the ET column
    """

    # Pre-treat latitude limits
    ## Fill NAs in the latitude limits with linear interpolation
    NH_lim = NH_lim.interpolate_na(dim=lon_name)
    SH_lim = SH_lim.interpolate_na(dim=lon_name)
    NH_lim = NH_lim.rename({lon_name: "longitude"})
    SH_lim = SH_lim.rename({lon_name: "longitude"})

    # Pre-treat tracks
    tracks = tracks.reset_index(drop=True)
    tracks["lon"] = np.where(tracks.lon > 180, tracks.lon - 360, tracks.lon)

    # Detect ET points
    ##Target longitudes indices
    n_lon = len(NH_lim.longitude)
    m = n_lon / 2 - 1
    a = (n_lon - 1 - m) / 180
    idx_lon = (m + a * tracks.lon).astype(int)
    ## Target time
    T = tracks.time.dt.date
    # T = np.where(T < np.datetime64("1950-01-16"), np.datetime64("1950-01-16"), T)
    # T = np.where(T > np.datetime64("2014-12-17"), np.datetime64("2014-12-17"), T)
    # target_lon = xr.DataArray(tracks.lon, dims="points")
    # target_time = xr.DataArray(tracks.time, dims="points")
    ##TODO: Unsatisying method
    target_time = [
        np.datetime64(t)
        if np.datetime64(t) in NH_lim.time.values
        else np.datetime64("1950-01-16")
        for t in tracks.time.dt.date.astype(str).values
    ]
    target_time = xr.DataArray(target_time, dims="points")
    tracks["lat_STJ_NH"] = NH_lim.sel(time=target_time, longitude=target_lon)
    tracks["lat_STJ_SH"] = SH_lim.sel(time=target_time, longitude=target_lon)
    tracks["ET"] = (tracks.lat > tracks.lat_STJ_NH) | (tracks.lat < tracks.lat_STJ_SH)

    if fill:
        # Fill trajectories once one point is ET
        tracks_ET = (
            tracks.groupby("track_id")["ET"]
            .max()
            .index[tracks.groupby("track_id")["ET"].max()]
        )
        dic = {t: False for t in tracks_ET}
        for row in (
            tracks[tracks.track_id.isin(tracks_ET)].sort_values("time").itertuples()
        ):
            if row.ET == True:
                dic[row.track_id] = True
            if dic[row.track_id] == True:
                tracks.loc[row.Index, "ET"] = True
            else:
                tracks.loc[row.Index, "ET"] = False

    return tracks


def remove_ET(tracks, trop_pts=1):
    """

    Parameters
    ----------
    tracks
    trop_pts (int): Number of trop points

    Returns
    -------

    """
    tracks["trop"] = 1 - tracks.ET
    ET_track_ids = (
        tracks.groupby("track_id")["trop"]
        .sum()
        .index[tracks.groupby("track_id")["trop"].sum() < trop_pts + 1]
    )
    tracks_trop = tracks[~tracks.track_id.isin(ET_track_ids)]
    tracks_ET = tracks[tracks.track_id.isin(ET_track_ids)]
    return tracks_trop, tracks_ET
