import pandas as pd
import numpy as np
from datetime import datetime


def load_ibtracs(
    season=None, file="data/ibtracs_1980-2020_simplified.csv", pos_lon=True
):
    tracks = pd.read_csv(file, keep_default_na=False) #TODO : Warning avec column 15 have mixed types
    if season != None:
        tracks = tracks[tracks.SEASON == season]
    tracks["time"] = tracks.ISO_TIME.astype(np.datetime64)
    if pos_lon:
        tracks.loc[tracks.LON < 0, "LON"] = tracks.loc[tracks.LON < 0, "LON"] + 360
    tracks["USA_SSHS"] = pd.to_numeric(tracks.USA_SSHS)
    tracks = (
        tracks[tracks.USA_SSHS >= 0]
        .rename(columns={col: col.lower() for col in tracks.columns})
        .rename(columns={"usa_sshs": "sshs", "sid": "track_id", "pres":"slp"})
        .drop(columns="season")
    )
    tracks["basin"] = tracks.basin.replace("EP", "ENP").replace("WP", "WNP")
    tracks["hemisphere"] = np.where(tracks.lat > 0, "N", "S")
    tracks = add_season(tracks)
    tracks["wind"] = tracks.wind.astype(float)
    return tracks


def load_TEtracks(
    file="tests/tracks_ERA5.csv",
    compute_sshs=True,
    pos_lon=True,
    surf_wind_col="wind",
    slp_col="slp",
):
    tracks = pd.read_csv(file)
    tracks = tracks.rename(columns={c: c[1:] for c in tracks.columns[1:]})
    tracks["hemisphere"] = np.where(tracks.lat > 0, "N", "S")
    tracks = add_season(tracks)
    tracks["basin"] = tracks.apply(lambda row: get_basin(row.lon, row.lat), axis =1)
    tracks[slp_col] /= 100
    if compute_sshs:
        tracks["sshs_wind"] = [
            sshs_from_wind(tracks[surf_wind_col][i]) for i in range(len(tracks))
        ]
        tracks["sshs_pres"] = [
            sshs_from_pres(tracks[slp_col][i]) for i in range(len(tracks))
        ]
    tracks["time"] = (
        tracks["year"].astype(str)
        + "-"
        + tracks["month"].astype(str)
        + "-"
        + tracks["day"].astype(str)
        + " "
        + tracks["hour"].astype(str)
        + ":00"
    ).astype(np.datetime64)
    if pos_lon:
        tracks.loc[tracks.lon < 0, "lon"] = tracks.loc[tracks.lon < 0, "lon"] + 360
    return tracks


_TRACK_data_vars = [
    "vor_tracked",
    "lon1",
    "lat1",
    "vor850",
    "lon2",
    "lat2",
    "vor700",
    "lon3",
    "lat3",
    "vor600",
    "lon4",
    "lat4",
    "vor500",
    "lon5",
    "lat5",
    "vor400",
    "lon6",
    "lat6",
    "vor300",
    "lon7",
    "lat7",
    "vor200",
    "lon8",
    "lat8",
    "slp",
    "lon9",
    "lat9",
    "wind925",
    "lon10",
    "lat10",
    "wind10",
]


def load_TRACKtracks(
    file="tests/tr_trs_pos.2day_addT63vor_addmslp_add925wind_add10mwind.tcident.new",
    data_vars=_TRACK_data_vars,
): # TODO : Doc
    """
    Parameters
    ----------
    file
    data_vars

    Returns
    -------

    """
    f = open(file)
    tracks = pd.DataFrame()
    line0 = f.readline()
    line1 = f.readline()
    line2 = f.readline()
    nb_tracks = int(line2.split()[1])
    c=0
    track_id=0
    time_step = []
    lon = []
    lat = []
    data = [[]]
    for line in f:
        if line.startswith("TRACK_ID"):
            data = pd.DataFrame(np.array(data), columns=data_vars[:np.shape(np.array(data))[1]])
            tracks = tracks.append(
                ## <- This part is taking a long time. Idea: Replace with a list and transform into pandas in the end ?
                pd.DataFrame(
                    {
                        "track_id": [track_id] * len(time_step),
                        "time_step": time_step,
                        "lon": lon,
                        "lat": lat,
                    }
                ).join(data)
            )
            c+=1
            season = line.split()[-1][:-6]
            track_id = season + '-' + str(c)
            time_step = []
            lon=[]
            lat=[]
            data=[]

        elif line.startswith("POINT_NUM"):
            pass
        else:
            time_step.append(line.split()[0])
            lon.append(float(line.split()[1]))
            lat.append(float(line.split()[2]))
            rest = line.split()[3:]
            mask = np.array(rest) == "&"
            data.append(np.array(rest)[~mask])

    f.close()
    SH = tracks.lat.mean() < 0
    tracks["year"] = tracks.time_step.str[:4].astype(int)
    tracks["month"] = tracks.time_step.str[-6:-4].astype(int)
    tracks["day"] = tracks.time_step.str[-4:-2].astype(int)
    tracks["hour"] = tracks.time_step.str[-2:].astype(int)
    if SH:
        tracks.loc[tracks.month <= 6, "year"] += 1
    tracks["time"] = (
        tracks["year"].astype(str)
        + "-"
        + tracks["month"].astype(str)
        + "-"
        + tracks["day"].astype(str)
        + " "
        + tracks["hour"].astype(str)
        + ":00"
    ).astype(np.datetime64)
    if SH :
        tracks["hemisphere"] = "S"
        tracks = add_season(tracks, hemisphere="S")
    else :
        tracks['hemisphere'] = 'N'
        tracks = add_season(tracks, hemisphere="N")
    tracks["basin"] = [
        get_basin(tracks.lon.iloc[i], tracks.lat.iloc[i]) for i in range(len(tracks))
    ]
    # TODO: Selectionner un subset de colonnes
    return tracks


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


def add_season(tracks, hemisphere = 'both', hemi_col="hemisphere", yr_col="year", mth_col="month"):
    if (hemisphere == 'both') :
        NH = tracks[tracks[hemi_col] == "N"]
        SH = tracks[tracks[hemi_col] == "S"]
    elif hemisphere == 'N' :
        NH = tracks
        SH = pd.DataFrame()
    elif hemisphere == 'S' :
        NH = pd.DataFrame()
        SH = tracks

    if len(NH)>0 :
        NH = NH.join(
            NH.groupby("track_id")[yr_col].mean().astype(int),
            on="track_id",
            rsuffix="season",
        ).rename(columns={yr_col + "season": "season"})

    if len(SH)>0 :
        track_dates = SH.groupby("track_id")[[yr_col, mth_col]].mean().astype(int)
        for row in track_dates.itertuples():
            if row.month <= 6:
                track_dates.loc[row.Index, "season"] = (
                    str(row.year - 1) + "-" + str(row.year)
                )
            if row.month >= 7:
                track_dates.loc[row.Index, "season"] = (
                    str(row.year) + "-" + str(row.year + 1)
                )
        SH = SH.join(track_dates["season"], on="track_id")



    return NH.append(SH)


def to_dt(t):
    ts = np.floor((t - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s"))
    return np.array(
        [datetime.utcfromtimestamp(t) if not np.isnan(t) else np.nan for t in ts]
    )

# Supprimer ?
def find_match(tc_detected, tracks_ref, mindays=1, maxd=4):
    """

    Parameters
    ----------
    id_detected
    tracks_detected
    tracks_ref
    mindays
    maxd

    Returns
    -------

    """
    candidates = tracks_ref[
        (tracks_ref.time >= tc_detected.time.min()) & (tracks_ref.time <= tc_detected.time.max())
    ].track_id.unique()
    if len(candidates) < 1:
        return pd.DataFrame({"id_ref": [np.nan], "dist": [np.nan], "temp": [np.nan]})
    matches = pd.DataFrame()
    for candidate in candidates:
        tc_candidate = tracks_ref[(tracks_ref.track_id == candidate)][['lon', 'lat', 'time']]
        merged = pd.merge(tc_detected, tc_candidate, on='time')
        dist = np.mean(merged.apply(lambda row: np.sqrt((row.lon_x - row.lon_y)** 2 + (row.lat_x - row.lat_y)**2), axis = 1)) # Compute distance
        temp = len(merged)
        matches = matches.append(
            pd.DataFrame({"id_ref": [candidate], "dist": [dist], "temp": [temp]})
        )
    matches = matches[matches.temp >= mindays * 4]
    matches = matches[matches.dist <= maxd]
    if len(matches) < 1:
        return pd.DataFrame({"id_ref": [np.nan], "dist": [np.nan], "temp": [np.nan]})
    return matches[matches.dist == matches.dist.min()]


def match_tracks(tracks1, tracks2, name1="algo", name2="ib", maxd=8, mindays=1):
    tracks1, tracks2 = tracks1[['track_id', 'lon', 'lat', 'time']], tracks2[['track_id', 'lon', 'lat', 'time']]
    merged = pd.merge(tracks1, tracks2, on='time')
    merged['dist'] = merged.apply(lambda row: np.sqrt((row.lon_x - row.lon_y) ** 2 + (row.lat_x - row.lat_y) ** 2),
                                  axis=1)
    dist = merged.groupby(['track_id_x', 'track_id_y'])[['dist']].mean()
    temp = merged.groupby(['track_id_x', 'track_id_y'])[['dist']].count().rename(columns={'dist': 'temp'})
    matches = dist.join(temp)
    matches = matches[(matches.dist < maxd) & (matches.temp > mindays*4)]
    matches = matches.loc[matches.groupby('track_id_x')['dist'].idxmin()].reset_index().rename(columns={'track_id_x':'track_id_'+name1, 'track_id_y':'track_id_'+name2})
    return matches


if __name__ == "__main__":
    ibtracs_file = '../data/ibtracs_1980-2020_simplified.csv'
    ib = load_ibtracs(file=ibtracs_file)
    #ib = ib[ib.year == 1980]

    UZ_tracks_file = '../tests/tracks_ERA5.csv'
    UZ = load_TEtracks(UZ_tracks_file, surf_wind_col='wind10')
    #UZ = UZ[UZ.year == 1980]

    matches_UZ = match_tracks(UZ, ib, 'UZ', 'ib')
    print(matches_UZ)