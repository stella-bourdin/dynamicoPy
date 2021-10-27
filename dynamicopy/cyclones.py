import pandas as pd
from datetime import datetime
import pickle as pkl
from dynamicopy.tc.basins import *
from shapely.geometry import Point
from haversine import haversine_vector, Unit

"""
Format for loading the tracks _data : 
track_id    time                lon     lat     hemisphere  basin   season  sshs    slp     wind10  year    month   day     (wind925)
str         np.datetime64[ns]   float   float   str         str     str     int     float   float   int     int     int     (float)

0 <= lon <= 360
"""






def open_TRACKpkl(
    path="",
    NH_seasons=[1980, 2019],
    SH_seasons=[1981, 2019],
):
    """

    Parameters
    ----------
    path
    NH_seasons
    SH_seasons

    Returns
    -------

    """
    with open(path, "rb") as handle:
        tracks = pkl.load(handle)
    tracks = add_season(tracks)
    tracks = tracks[
        ((tracks.season >= NH_seasons[0]) & (tracks.season <= NH_seasons[1]))
        | (tracks.hemisphere == "S")
    ]
    tracks = tracks[
        ((tracks.season >= SH_seasons[0]) & (tracks.season <= SH_seasons[1]))
        | (tracks.hemisphere == "N")
    ]
    return tracks


def load_CNRMtracks(
    file="tests/tracks_CNRM.csv",
    NH_seasons=[1980, 2019],
    SH_seasons=[1981, 2019],
):
    tracks = pd.read_csv(file)
    tracks = tracks.rename(
        columns={
            "ID": "track_id",
            "Date": "time",
            "Longitude": "lon",
            "Latitude": "lat",
            "Pressure": "slp",
            "Wind": "wind10",
        }
    )
    tracks["hemisphere"] = np.where(tracks.lat > 0, "N", "S")
    tracks["basin"] = get_basin(tracks.lon.values, tracks.lat.values)
    tracks["time"] = tracks.time.astype(np.datetime64)
    tracks["year"] = tracks.time.dt.year
    tracks["month"] = tracks.time.dt.month
    tracks["day"] = tracks.time.dt.day
    tracks = add_season(tracks)
    tracks = tracks[
        ((tracks.season >= NH_seasons[0]) & (tracks.season <= NH_seasons[1]))
        | (tracks.hemisphere == "S")
    ]
    tracks = tracks[
        ((tracks.season >= SH_seasons[0]) & (tracks.season <= SH_seasons[1]))
        | (tracks.hemisphere == "N")
    ]
    tracks["sshs"] = sshs_from_pres(tracks.slp)
    return tracks[
        [
            "track_id",
            "time",
            "lon",
            "lat",
            "hemisphere",
            "basin",
            "season",
            "sshs",
            "slp",
            "wind10",
            "year",
            "month",
            "day",
        ]
    ]


def is_leap(yr):
    if yr % 4 == 0:
        if yr % 100 == 0:
            if yr % 400 == 0:
                return True
            else:
                return False
        else:
            return True
    else:
        return False




def sshs_from_wind(wind):
    sshs = np.where(wind <= 60 / 3.6, -1, None)
    sshs = np.where((sshs == None) & (wind < 120 / 3.6), 0, sshs)
    sshs = np.where((sshs == None) & (wind < 150 / 3.6), 1, sshs)
    sshs = np.where((sshs == None) & (wind < 180 / 3.6), 2, sshs)
    sshs = np.where((sshs == None) & (wind < 210 / 3.6), 3, sshs)
    sshs = np.where((sshs == None) & (wind < 240 / 3.6), 4, sshs)
    sshs = np.where((sshs == None) & (~np.isnan(wind)), 5, sshs)
    sshs = np.where(sshs == None, np.nan, sshs)
    return sshs



def get_basin_old(hemisphere, lon, lat):
    basin = np.where((hemisphere == "N") & (lon > 40) & (lon <= 100), "NI", "")
    basin = np.where(
        (hemisphere == "N")
        & (lon > 100)
        & ((lon <= 200) | ((lat >= 35) & (lon <= 250))),
        "WNP",
        basin,
    )
    basin = np.where(
        (hemisphere == "N")
        & (basin != "WNP")
        & (lon > 200)
        & ((lon <= 260) | ((lat <= 15) & (lon <= 290))),
        "ENP",
        basin,
    )
    basin = np.where((hemisphere == "N") & (basin != "ENP") & (lon > 260), "NA", basin)
    basin = np.where((hemisphere == "S") & (lon > 20) & (lon <= 130), "SI", basin)
    basin = np.where((hemisphere == "S") & (lon > 130) & (lon <= 300), "SP", basin)
    basin = np.where((hemisphere == "S") & (basin == ""), "SA", basin)
    return basin





def to_dt(t):
    ts = np.floor((t - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s"))
    return np.array(
        [datetime.utcfromtimestamp(t) if not np.isnan(t) else np.nan for t in ts]
    )


def match_tracks(tracks1, tracks2, name1="algo", name2="ib", maxd=8, mindays=1):
    """

    Parameters
    ----------
    tracks1 (pd.DataFrame): First tracks DataFrame
    tracks2 (pd.DataFrame): Second tracks DataFrame
    name1 (str): name to append corresponding to the first df
    name2 (str): name to append corresponding to the second df
    maxd (numeric): Maximum allowed distance between two tracks
    mindays (int): Minimum number of days in common between two tracks

    Returns
    -------
    pd.DataFrame
        with the track ids of the matching trajectories in tracks1 and tracks2
    """
    tracks1, tracks2 = (
        tracks1[["track_id", "lon", "lat", "time"]],
        tracks2[["track_id", "lon", "lat", "time"]],
    )
    merged = pd.merge(tracks1, tracks2, on="time")
    X = np.concatenate([[merged.lat_x], [merged.lon_x]]).T
    Y = np.concatenate([[merged.lat_y], [merged.lon_y]]).T
    merged["dist"] = haversine_vector(X, Y, unit=Unit.DEGREES)
    dist = merged.groupby(["track_id_x", "track_id_y"])[["dist"]].mean()
    temp = (
        merged.groupby(["track_id_x", "track_id_y"])[["dist"]]
        .count()
        .rename(columns={"dist": "temp"})
    )
    matches = dist.join(temp)
    matches = matches[(matches.dist < maxd) & (matches.temp > mindays * 4)]
    matches = (
        matches.loc[matches.groupby("track_id_x")["dist"].idxmin()]
        .reset_index()
        .rename(columns={"track_id_x": "id_" + name1, "track_id_y": "id_" + name2})
    )
    return matches


def match_william(tracks1, tracks2, name1="algo", name2="ib"):
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

if __name__ == "__main__":
    # t = load_TRACKtracks()
    pass
