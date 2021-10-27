import numpy as np
from ._basins import *
from shapely.geometry import Point

def add_season(tracks):
    """
    Add (or changes) the season columns in a track dataframe.

    Parameters
    ----------
    tracks (pd.DataFrame): The track dataframe

    Returns
    -------
    pd.DataFrame
        The track dataframe with the new season column
    """
    if "season" in tracks.columns:
        tracks = tracks.drop(columns="season")
    group = (
        tracks.groupby(["track_id"])[["year", "month"]]
        .mean()
        .astype(int)
        .join(
            tracks[["track_id", "hemisphere"]].drop_duplicates().set_index("track_id"),
            on="track_id",
        )
    )
    hemi, yr, mth = group.hemisphere.values, group.year.values, group.month.values
    season = np.where(hemi == "N", yr, None)
    season = np.where((hemi == "S") & (mth >= 7), yr + 1, season)
    season = np.where((hemi == "S") & (mth <= 6), yr, season)
    group["season"] = season.astype(int)
    tracks = tracks.join(group[["season"]], on="track_id")
    return tracks

def get_time(year, month, day, hour):
    time = (
        year.astype(str)
        + "-"
        + month.astype(str)
        + "-"
        + day.astype(str)
        + " "
        + hour.astype(str)
        + ":00"
    ).astype(np.datetime64)
    return time

# TODO : Optimiser cette fonction
def get_basin(lon, lat):
    basin = []
    for x, y in zip(lon, lat):
        ok = False
        if y >= 0:
            for b in NH:
                if NH[b].contains(Point(x, y)):
                    basin.append(b)
                    ok = True
                    break
        else:
            for b in SH:
                if SH[b].contains(Point(x, y)):
                    basin.append(b)
                    ok = True
                    break
        if ok == False:
            basin.append(np.nan)
    return basin

def sshs_from_pres(p):
    sshs = np.where(p > 990, -1, None)
    sshs = np.where((sshs == None) & (p >= 980), 0, sshs)
    sshs = np.where((sshs == None) & (p >= 970), 1, sshs)
    sshs = np.where((sshs == None) & (p >= 965), 2, sshs)
    sshs = np.where((sshs == None) & (p >= 945), 3, sshs)
    sshs = np.where((sshs == None) & (p >= 920), 4, sshs)
    sshs = np.where((sshs == None) & (~np.isnan(p)), 5, sshs)
    sshs = np.where(sshs == None, np.nan, sshs)
    return sshs

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