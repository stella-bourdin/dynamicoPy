import pandas as pd
import numpy as np
from datetime import datetime


def load_ibtracs(season=None, file=None, pos_lon=True):
    if file == None:
        tracks = pd.read_csv("data/ibtracs_1980-2020_simplified.csv")
    else:
        tracks = pd.read_csv(file)
    if season != None:
        tracks = tracks[tracks.SEASON == season]
    tracks["time"] = tracks.ISO_TIME.astype(np.datetime64)
    if pos_lon:
        tracks.loc[tracks.LON < 0, "LON"] = tracks.loc[tracks.LON < 0, "LON"] + 360
    tracks = tracks[tracks.USA_SSHS >= 0].rename(
        columns={
            "SID": "track_id",
            "SEASON": "season",
            "BASIN": "basin",
            "USA_SSHS": "sshs",
            "LAT": "lat",
            "LON": "lon",
            "PRES": "slp",
            "WIND": "wind",
        }
    )
    return tracks


def load_TEtracks(file="tests/tracks_ERA5.csv", compute_sshs=True, pos_lon=True, surf_wind_col='wind'):
    df = pd.read_csv(file)
    df = df.rename(columns={c: c[1:] for c in df.columns[1:]})
    df["hemisphere"] = np.where(df.lat > 0, "N", "S")
    # Get season
    df = df.join(
        df.groupby("track_id")["year"].mean().astype(int),
        on="track_id",
        rsuffix="season",
    ).rename(columns={"yearseason": "season"})
    df["basin"] = [get_basin(df.lon[i], df.lat[i]) for i in range(len(df))]
    if compute_sshs:
        df["sshs_wind"] = [sshs_from_wind(df[surf_wind_col][i]) for i in range(len(df))]
        df["sshs_pres"] = [sshs_from_pres(df.slp[i] / 100) for i in range(len(df))]
    df["time"] = (
        df["year"].astype(str)
        + "-"
        + df["month"].astype(str)
        + "-"
        + df["day"].astype(str)
        + " "
        + df["hour"].astype(str)
        + ":00"
    ).astype(np.datetime64)
    if pos_lon:
        df.loc[df.lon < 0, "lon"] = df.loc[df.lon < 0, "lon"] + 360
    return df


def sshs_from_wind(wind):
    if wind <= 60 / 3.6:
        return -1
    elif wind <= 120 / 3.6:
        return 0
    elif wind <= 150 / 3.6:
        return 1
    elif wind <= 180 / 3.6:
        return 2
    elif wind <= 210 / 3.6:
        return 3
    elif wind <= 240 / 3.6:
        return 4
    else:
        return 5


def sshs_from_pres(p):
    if p >= 990:
        return -1
    elif p >= 980:
        return 0
    elif p >= 970:
        return 1
    elif p >= 965:
        return 2
    elif p >= 945:
        return 3
    elif p >= 920:
        return 4
    else:
        return 5


def get_basin(lon, lat):
    if lat >= 0:
        if lon <= 40:
            return np.nan
        elif lon <= 100:
            return "NI"
        elif lon <= 200:
            return "WNP"
        elif (lat >= 35) & (lon <= 250):
            return "WNP"
        elif lon <= 260:
            return "ENP"
        elif (lat <= 15) & (lon <= 290):
            return "ENP"
        else:
            return "NA"
    else:
        if lon <= 20:
            return "SA"
        elif lon <= 130:
            return "SI"
        elif lon <= 300:
            return "SP"
        else:
            return "SA"


def to_dt(t):
    ts = np.floor((t - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s"))
    return np.array(
        [datetime.utcfromtimestamp(t) if not np.isnan(t) else np.nan for t in ts]
    )


def compute_dist(
    id_detected,
    id_ref,
    df_detected,
    df_ref,
):
    """
    returns : mean distance, number of matching time steps.
    """
    c_d = df_detected[(df_detected.track_id == id_detected)]
    c_ref = df_ref[(df_ref.track_id == id_ref)]
    t_d = to_dt(c_d.time)
    t_ref = to_dt(c_ref.time)
    mask_d = [t in t_ref for t in t_d]
    mask_ref = [t in t_d for t in t_ref]
    if np.sum(mask_d) > 0:
        path_d = c_d[["lat", "lon"]][mask_d]
        path_ref = c_ref[["lat", "lon"]][mask_ref]
        d = [
            np.sqrt((c[0][0] - c[1][0]) ** 2 + (c[0][1] - c[1][1]) ** 2)
            for c in zip(path_d.values, path_ref.values)
        ]
        return np.mean(d), np.sum(mask_d)
    else:
        return -1, 0


def find_match(id_detected, df_detected, df_ref, mindays=1, maxd=4):
    """

    Parameters
    ----------
    id_detected
    df_detected
    df_ref
    mindays
    maxd

    Returns
    -------

    """
    c = df_detected[(df_detected.track_id == id_detected)]
    candidates = df_ref[
        (df_ref.time >= c.time.min()) & (df_ref.time <= c.time.max())
    ].track_id.unique()
    if len(candidates) < 1:
        return pd.DataFrame({"id_ref": [np.nan], "dist": [np.nan], "temp": [np.nan]})
    matches = pd.DataFrame()
    for candidate in candidates:
        dist, temp = compute_dist(id_detected, candidate, df_detected, df_ref)
        matches = matches.append(
            pd.DataFrame({"id_ref": [candidate], "dist": [dist], "temp": [temp]})
        )
    matches = matches[matches.temp >= mindays * 4]
    matches = matches[matches.dist <= maxd]
    if len(matches) < 1:
        return pd.DataFrame({"id_ref": [np.nan], "dist": [np.nan], "temp": [np.nan]})
    return matches[matches.dist == matches.dist.min()]

def match_tracks(tracks1, tracks2, name1='algo', name2='ib', maxd=8):
    matches=pd.DataFrame()
    for id in tracks1.track_id.unique():
        matches = matches.append(find_match(id,tracks1,tracks2, maxd=maxd).assign(id_detected=id))
    matches.rename(columns={'id_ref':'id_'+name2, 'id_detected':'id_'+name1})
    return matches

if __name__ == "__main__":
    ib = load_ibtracs(1996)
    tracks = load_TEtracks()
    matches = pd.DataFrame()
    for id in tracks.track_id.unique():
        matches = matches.append(find_match(id, tracks, ib, maxd=7).assign(id_era=id))
    matches = matches[~matches.id_ref.isnull()]
