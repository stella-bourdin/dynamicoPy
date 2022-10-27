import numpy as np
import xarray as xr

def remove_ET(tracks, trop_pts=1):
    """
    Parameters
    ----------
    tracks (pd.Dataframe) : The dataframe containing the tracks data
    trop_pts (int): Maximum number of tropical points in an ET track

    Returns
    -------
    tracks_trop (pd.DataFrame) : The tropical trajectories
    tracks_ET (pd.DataFrame) : The ET trajectories
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
