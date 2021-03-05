
import pandas as pd
import numpy as np


def load_ibtracs(season = None):
    tracks = pd.read_csv("data/ibtracs_1980-2020_simplified.csv")
    if season != None :
        tracks = tracks[tracks.SEASON == season]
    tracks["time"] = tracks.ISO_TIME.astype(np.datetime64)
    tracks.loc[tracks.LON < 0, 'LON'] = tracks.loc[tracks.LON < 0, 'LON'] + 360
    tracks = tracks[tracks.USA_SSHS >= 0].rename(
        columns={"SID": 'track_id', "SEASON": "season", "BASIN": 'basin', 'USA_SSHS': 'sshs', 'LAT': 'lat',
                 'LON': 'lon', 'PRES': 'slp', 'WIND': 'wind'})
    return tracks

def load_TEtracks(file="tests/tracks_ERA5.csv"):
    df = pd.read_csv(file)
    df = df.rename(columns={c: c[1:] for c in df.columns[1:]})
    df["hemisphere"] = np.where(df.lat > 0, 'N', 'S')
    # Get season
    df = df.join(df.groupby('track_id')['year'].mean().astype(int), on='track_id', rsuffix='season').rename(
        columns={"yearseason": "season"})
    df["basin"] = [get_basin(df.lon[i], df.lat[i]) for i in range(len(df))]
    df["sshs_wind"] = [sshs_from_wind(df.wind[i]) for i in range(len(df))]
    df["sshs_pres"] = [sshs_from_pres(df.slp[i] / 100) for i in range(len(df))]
    df['time'] = (
                df['year'].astype(str) + '-' + df['month'].astype(str) + '-' + df['day'].astype(str) + ' ' +
                df['hour'].astype(str) + ':00').astype(np.datetime64)
    return df

def sshs_from_wind(wind):
    if wind <= 60/3.6 :
        return -1
    elif wind <= 120/3.6 :
        return 0
    elif wind <= 150/3.6 :
        return 1
    elif wind <= 180/3.6 :
        return 2
    elif wind <= 210/3.6 :
        return 3
    elif wind <= 240/3.6 :
        return 4
    else :
        return 5

def sshs_from_pres(p):
    if p >= 990 :
        return -1
    elif p >= 980 :
        return 0
    elif p >= 970 :
        return 1
    elif p >= 965 :
        return 2
    elif p >= 945 :
        return 3
    elif p >= 920 :
        return 4
    else :
        return 5

def get_basin(lon, lat):
    if lat >= 0:
        if lon <= 40:
            return np.nan
        elif lon <= 100:
            return 'NI'
        elif lon <= 200:
            return 'WNP'
        elif (lat >= 35) & (lon <= 250):
            return 'WNP'
        elif lon <= 260:
            return 'ENP'
        elif (lat <= 15) & (lon <= 290):
            return 'ENP'
        else:
            return 'NA'
    else:
        if lon <= 20:
            return 'SA'
        elif lon <= 130:
            return 'SI'
        elif lon <= 300:
            return 'SP'
        else:
            return 'SA'



if __name__ == "__main__":
    ibtracs = load_ibtracs()
