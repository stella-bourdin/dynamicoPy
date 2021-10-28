import numpy as np

def identify_ET(tracks, NH_lim, SH_lim):
    """

    Parameters
    ----------
    tracks (pd.Dataframe)
    NH_lim (xr.DataArray)
    SH_lim (xr.DataArray)

    Returns
    -------
    pd.Dataframe
        tracks with the ET column
    """

    # Pre-treat latitude limits
    ## Fill NAs in the latitude limits with linear interpolation
    NH_lim = NH_lim.interpolate_na(dim="longitude")
    SH_lim = SH_lim.interpolate_na(dim="longitude")
    ## Change time to -3h to fit with tracks
    NH_lim["time"] = NH_lim.time - np.timedelta64(3, 'h')
    SH_lim["time"] = SH_lim.time - np.timedelta64(3, 'h')

    # Pre-treat tracks
    tracks = tracks.reset_index(drop=True)
    tracks["lon"] = np.floor(tracks.lon * 4)/4
    tracks["lon"] = np.where(tracks.lon > 180, tracks.lon - 360, tracks.lon)

    # Detect ET points
    idx_lon = [np.where(NH_lim.longitude == tracks.lon.iloc[i])[0][0] for i in range(len(tracks))]
    idx_time = [np.where(NH_lim.time.values == tracks.time.iloc[i])[0][0] for i in range(len(tracks))]
    tracks["lat_STJ_NH"] = [NH_lim[(idx_time[i], idx_lon[i])].values for i in range(len(idx_time))]
    tracks["lat_STJ_SH"] = [SH_lim[(idx_time[i], idx_lon[i])].values for i in range(len(idx_time))]
    tracks["ET"] = (tracks.lat > tracks.lat_STJ_NH) | (tracks.lat < tracks.lat_STJ_SH)

    # Fill trajectories once one point is ET
    tracks_ET = tracks.groupby('track_id')["ET"].max().index[tracks.groupby('track_id')["ET"].max()]
    dic = {t: False for t in tracks_ET}
    for row in tracks[tracks.track_id.isin(tracks_ET)].sort_values("time").itertuples():
        if row.ET == True:
            dic[row.track_id] = True
        if dic[row.track_id] == True:
            tracks.loc[row.Index, "ET"] = True
        else:
            tracks.loc[row.Index, "ET"] = False

    return tracks
