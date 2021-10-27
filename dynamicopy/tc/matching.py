import pandas as pd
import numpy as np
from haversine import haversine_vector, Unit

def match_tracks(tracks1, tracks2, name1="algo", name2="ib"):
    """

    Parameters
    ----------
    tracks1 (pd.DataFrame): the first track dataframe to match
    tracks2 (pd.DataFrame): the second track dataframe to match
    name1 (str): Suffix for the first dataframe
    name2 (str): Suffix for the second dataframe

    Returns
    -------
    pd.DataFrame
        Dataframe containing the matching tracks with
            the id from both datasets
            the number of matching time steps
            the distance between two tracks
    """
    tracks1, tracks2 = (
        tracks1[["track_id", "lon", "lat", "time"]],
        tracks2[["track_id", "lon", "lat", "time"]],
    )
    merged = pd.merge(tracks1, tracks2, on="time")
    X = np.concatenate([[merged.lat_x], [merged.lon_x]]).T
    Y = np.concatenate([[merged.lat_y], [merged.lon_y]]).T
    merged["dist"] = haversine_vector(X, Y, unit=Unit.KILOMETERS)
    merged = merged[merged.dist <= 300]
    temp = (
        merged.groupby(["track_id_x", "track_id_y"])[["dist"]]
        .count()
        .rename(columns={"dist": "temp"})
    )
    matches = (
        merged[["track_id_x", "track_id_y"]]
        .drop_duplicates()
        .join(temp, on=["track_id_x", "track_id_y"])
    )
    maxs = matches.groupby("track_id_x")[["temp"]].max().reset_index()
    matches = maxs.merge(matches)[["track_id_x", "track_id_y", "temp"]]
    dist = merged.groupby(["track_id_x", "track_id_y"])[["dist"]].mean()
    matches = matches.join(dist, on=["track_id_x", "track_id_y"])
    matches = matches.rename(
        columns={"track_id_x": "id_" + name1, "track_id_y": "id_" + name2}
    )
    return matches