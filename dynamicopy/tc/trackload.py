from .utils import *

"""
Format for loading the tracks _data : 
track_id    time                lon     lat     hemisphere  basin   season  sshs    slp     wind10  year    month   day    hour 
str         np.datetime64[ns]   float   float   str         str     str     int     float   float   int     int     int    int

0 <= lon <= 360
"""

def load_TEtracks(
    file="tests/tracks_ERA5.csv",
    NH_seasons=[1980, 2019],
    SH_seasons=[1981, 2019],
    surf_wind_col="wind10",
    slp_col="slp",
):
    """
    Parameters
    ----------
    file (str): csv file from TempestExtremes StitchNodes output
    NH_season (list of 2 ints): first and last season in the northern hemisphere
    SH_season (list of 2 ints): first and last season in the southern hemisphere
    surf_wind_col (str): Name of the column with the surface wind to output.
    slp_col (str): Name of the column with the sea-level pressure. If None, no sshs computation.

    Returns
    -------
    pd.DataFrame
        Columns as described in the module header
    """

    tracks = pd.read_csv(file)
    tracks = tracks.rename(columns={c: c[1:] for c in tracks.columns[1:]})
    tracks = tracks.rename(columns={surf_wind_col: "wind10", slp_col: "slp"})

    tracks["time"] = get_time(tracks.year, tracks.month, tracks.day, tracks.hour)
    tracks.loc[tracks.lon < 0, "lon"] += 360
    tracks["hemisphere"] = np.where(tracks.lat > 0, "N", "S")
    tracks["basin"] = get_basin(tracks.lon.values, tracks.lat.values)
    tracks = add_season(tracks)
    tracks = tracks[
        ((tracks.season >= NH_seasons[0]) & (tracks.season <= NH_seasons[1]))
        | (tracks.hemisphere == "S")
    ]
    tracks = tracks[
        ((tracks.season >= SH_seasons[0]) & (tracks.season <= SH_seasons[1]))
        | (tracks.hemisphere == "N")
    ]
    tracks[slp_col] /= 100
    if slp_col != None:
        tracks["sshs"] = sshs_from_pres(tracks.slp.values)
    else:
        tracks["sshs"] = np.nan
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
            "hour",
        ]
    ]