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

def merge_duplicates(tracks1, tracks2, matches = None):
    """
    Function to manage cases where tracks from tracks2 match with several tracks in tracks1,
    by merging the duplicates into one track.

    Parameters
    ----------
    tracks1 (pd.Dataframe)
    tracks2 (pd.Dataframe)
    matches (pd.Dataframe): The output from match_tracks on tracks1 and tracks2.
        If None, match_tracks is run on tracks1 and tracks2.

    Returns
    -------
    pd.Dataframe
        The new match dataset
    + tracks1 modified in place
    """
    c1, c2 = matches.columns[:2]
    count = matches.groupby(c2)[[c1]].nunique()
    duplicates = count[(count > 1).values].index
    print("Handled " + str(len(duplicates)) + " duplicates")
    for i in duplicates:
        merge = matches[matches[c2] == i][c1].values
        new_id = '+'.join(merge.astype(str))
        tracks1["track_id"] = tracks1.track_id.replace({old_id: new_id for old_id in merge})
    matches = match_tracks(tracks1, tracks2, c1[3:], c2[3:])
    return matches
