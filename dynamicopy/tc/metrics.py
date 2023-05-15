import numpy as np
import pandas as pd
from haversine import haversine
from .CPS import theta_multitrack
from ._basins import list_in_med
import xarray as xr

def tc_count(tracks):
    """
    Counts the number of tracks per basin and per SSHS category

    Parameters
    ----------
    tracks (pd.DataFrame): A track dataframe

    Returns
    -------
    pd.DataFrame
        A two-entry table with the basins as rows and the categories as columns, values are the counts. Total added.
    """
    tracks = tracks[~tracks.ET].copy()
    storms = tracks.groupby("track_id")[["hemisphere", "basin"]].agg(
        lambda x: x.value_counts().index[0]
    )
    storms["sshs"] = tracks.groupby("track_id")["sshs"].max()
    B = (
        storms.groupby(["sshs", "hemisphere"])[["basin"]]
        .count()
        .reset_index()
        .pivot_table(
            index="hemisphere",
            columns="sshs",
            fill_value=0.0,
            margins=True,
            aggfunc=np.sum,
        )
        .rename(columns={"basin": "count"})
    )
    C = (
        storms.groupby(["sshs", "basin"])[["hemisphere"]]
        .count()
        .reset_index()
        .pivot_table(
            index="basin", columns="sshs", fill_value=0.0, margins=True, aggfunc=np.sum
        )
        .rename(columns={"hemisphere": "count"})
    )
    return (
        B.append(C)
        .drop_duplicates()
        .rename(index={"All": "global"})
        .reindex(["global", "N", "WNP", "ENP", "NI", "NATL", "S", "SP", "SI", "SA"])
    )

def get_freq(tracks):  # TODO : A optimiser
    tracks = tracks[~tracks.ET].copy()
    storms = tracks.groupby("track_id")[["season", "hemisphere", "basin"]].agg(
        lambda x: x.value_counts().index[0]
    )
    basins = (
        storms.reset_index()
        .groupby(["season", "basin"])["track_id"]
        .count()
        .reset_index()
        .pivot_table(index=["basin"], columns="season", fill_value=0)
        .mean(axis=1)
    )
    hemi = (
        storms.reset_index()
        .groupby(["season", "hemisphere"])["track_id"]
        .count()
        .reset_index()
        .pivot_table(index=["hemisphere"], columns="season", fill_value=0)
        .mean(axis=1)
    )
    freq = hemi.append(basins)
    freq.loc["global"] = 0
    if "S" in freq.index:
        freq.loc["global"] += freq.loc["S"]
    if "N" in freq.index:
        freq.loc["global"] += freq.loc["N"]
    return freq.reindex(
        ["global", "N", "WNP", "ENP", "NI", "NATL", "S", "SP", "SI", "SA"]
    )

def prop_intense(freq, sshs_min=4):
    """
    Retrieve the proportion of intense tc among all

    Parameters
    ----------
    freq (pd.Dataframe): output from the get_freq function

    Returns
    -------
    pd.Dataframe
        total freq, intense freq, intense prop
    """
    cat_45_cols = list(freq.columns[:-1] >= sshs_min) + [False]
    freq_45 = freq.loc[:, cat_45_cols].sum(axis=1)
    prop_45 = freq_45 / freq.loc[:, "All"]
    return freq[["All"]].assign(intense=freq_45).assign(prop=prop_45)

def storm_stats(tracks, time_step = 6):
    """
    Statistics about each track

    Parameters
    ----------
    tracks (pd.Dataframe): The track dataframe

    Returns
    -------
    pd.Dataframe
        Grouped dataframe of the initial one
    """

    # Prepare tracks dataset
    tracks = tracks.copy()
    tracks["wind10"] = tracks.wind10.round(2)
    if "ET" in tracks.columns :
        tracks.loc[tracks.ET.isna(), "ET"] = False
    else :
        tracks["ET"] = False

    # Compute track length
    storms = (tracks.groupby(["track_id"])[["time"]].count() / (24/time_step)).reset_index()

    # Intensity stats : Only on tropical part
    tracks = tracks[tracks.ET.astype(int) == 0]

    # Retrieve the line of max wind and min slp for each track
    tracks_wind_climax = tracks[~tracks.wind10.isna()].sort_values("wind10").groupby("track_id").last().reset_index()[["track_id", "wind10", "lat", "time", "hemisphere", "season", "month", "basin",]]
    tracks_slp_climax = tracks[~tracks.slp.isna()].sort_values("slp").groupby("track_id").first().reset_index()[["track_id", "slp", "lat", "time", "sshs"]]

    # Retrieve the line of genesis for each track
    gen = tracks.sort_values("time").groupby("track_id").first().reset_index()[
        ["track_id", "lat", "time", "basin",]]

    # Merge all together
    storms = storms.merge(tracks_wind_climax, on="track_id", suffixes=("", "_wind"), how = "outer").rename(columns = {"lat":"lat_wind"})
    storms = storms.merge(tracks_slp_climax, on="track_id", suffixes=("", "_slp"), how = "outer")
    storms = storms.merge(gen, on="track_id", suffixes=("", "_gen"))

    return storms

def storm_stats_med(tracks, time_step = 6):
    """
    Statistics about each track

    Parameters
    ----------
    tracks (pd.Dataframe): The track dataframe

    Returns
    -------
    pd.Dataframe
        Grouped dataframe of the initial one
    """

    # Prepare tracks dataset
    tracks = tracks.copy()
    tracks["wind10"] = tracks.wind10.round(2)

    # Compute track length
    storms = (tracks.groupby(["track_id"])[["time"]].count() / (24/time_step)).reset_index()

    # Retrieve the line of max wind and min slp for each track
    tracks_wind_climax = tracks[~tracks.wind10.isna()].sort_values("wind10").groupby("track_id").last().reset_index()[["track_id", "wind10", "lat", "time", "hemisphere", "season", "month",]]
    tracks_slp_climax = tracks[~tracks.slp.isna()].sort_values("slp").groupby("track_id").first().reset_index()[["track_id", "slp", "lat", "time",]]

    # Retrieve the line of genesis for each track
    gen = tracks.sort_values("time").groupby("track_id").first().reset_index()[
        ["track_id", "lat", "time",]]

    # Merge all together
    storms = storms.merge(tracks_wind_climax, on="track_id", suffixes=("", "_wind"), how = "outer").rename(columns = {"lat":"lat_wind"})
    storms = storms.merge(tracks_slp_climax, on="track_id", suffixes=("", "_slp"), how = "outer")
    storms = storms.merge(gen, on="track_id", suffixes=("", "_gen"))

    return storms


"""
def storm_stats_med(tracks, time_step = 1):
    """
    Statistics about each track

    Parameters
    ----------
    tracks (pd.Dataframe): The track dataframe

    Returns
    -------
    pd.Dataframe
        Grouped dataframe of the initial one
    """

    # Prepare tracks dataset
    tracks = tracks.sort_values(["track_id", "time"]).copy()
    tracks["wind10"] = tracks.wind10.round(2)
    #if "in_med" not in tracks.columns :
        #tracks["in_med"] = list_in_med(tracks.lon, tracks.lat)
    if "theta" not in tracks.columns :
        tracks["theta"] = theta_multitrack(tracks)

    # Retrieved grouped data
    ## Storm length
    time = (tracks.groupby(["track_id"]).time.max() - tracks.groupby(["track_id"]).time.min()).astype('timedelta64[h]')
    ## Number & proportion of points in med
    N_med = tracks.groupby("track_id").in_med.sum()
    prop_med = N_med / time
    ## Merge
    storms = pd.DataFrame(time).join(N_med.rename("N_med")).join(prop_med.rename("prop_med"))

    # Retrieve the line of max wind and min slp for each track
    tracks_wind_climax = tracks[~tracks.wind10.isna()].sort_values("wind10").groupby("track_id").last().reset_index()[["track_id", "wind10", "lat", "time", "hemisphere", "season", "month", "basin",]]
    tracks_slp_climax = tracks[~tracks.slp.isna()].sort_values("slp").groupby("track_id").first().reset_index()[["track_id", "slp", "lat", "time", "sshs"]]

    # Retrieve the line of genesis for each track
    gen = tracks.sort_values("time").groupby("track_id").first().reset_index()[
        ["track_id", "lat", "lon", "time",]]
    dissip = tracks.sort_values("time").groupby("track_id").last().reset_index()[
        ["track_id", "lat", "lon", "time",]]

    # Compute distance between first and last point
    storms["start_end_dist"] = [haversine((gen.iloc[i].lat,gen.iloc[i].lon), (dissip.iloc[i].lat, dissip.iloc[i].lat))
                      for i in range(len(gen))]
    # TODO : Make sure that order of track_ids is ok

    tracks.loc[:tracks.index[-2], "lon_next"] = tracks.lon[1:].values
    tracks.loc[:tracks.index[-2], "lat_next"] = tracks.lat[1:].values
    tracks.loc[:tracks.index[-2], "theta_next"] = tracks.theta[1:].values
    tracks.loc[:tracks.index[-2], "time_next"] = tracks.time[1:].values
    tracks["tpos"] = tracks.sort_values("time", ascending=False).groupby("track_id").transform("cumcount")
    # Maximum distance between two points
    tracks["dist_next"] = tracks.apply(lambda x: haversine((x.lat, x.lon), (x.lat_next, x.lon_next)), axis=1)
    tracks.loc[tracks.tpos == 0, "dist_next"] = np.nan
    storms["maxdist"] = tracks.groupby("track_id").dist_next.max()
    # Maximum theta between two points
    tracks["dtheta"] = tracks.theta_next - tracks.theta
    tracks["dtheta"] = np.where(tracks.dtheta > 180, 360 - tracks.dtheta, tracks.dtheta)
    tracks.loc[tracks.tpos == 0, "dtheta"] = np.nan
    storms["maxtheta"] = tracks.groupby("track_id").dtheta.max()
    # Maximum gap between two points
    tracks["gap_next"] = (tracks.time_next - tracks.time).astype('timedelta64[h]')
    tracks.loc[tracks.tpos == 0, "gap_next"] = np.nan
    storms["maxgap"] = tracks.groupby("track_id").gap_next.max()

    # Merge all together
    storms = storms.merge(tracks_wind_climax, on="track_id", suffixes=("", "_wind"), how = "outer").rename(columns = {"lat":"lat_wind"})
    storms = storms.merge(tracks_slp_climax, on="track_id", suffixes=("", "_slp"), how = "outer")
    storms = storms.merge(gen, on="track_id", suffixes=("", "_gen"))

    return storms
"""

def propagation_speeds(tracks): # TODO : Probleme quand un point traverse le méridien de greenwhich
    """
    Return the propagation speed of each track

    Parameters
    ----------
    tracks (pd.Dataframe): The track dataframe

    Returns
    -------
    dict
        keys are the track ids, values the arrays corresponding to propagation speed along the trajectory.
    """

    speeds = {}
    for t in tracks.track_id.unique():
        T = tracks[tracks.track_id == t]
        dlon = T[:-1].lon.values - T[1:].lon.values
        dlat = T[:-1].lat.values - T[1:].lat.values
        dist = np.sqrt(dlon ** 2 + dlat ** 2)  # Vérifier les aspects de GCD etc.
        speeds[t] = dist * 100 / 6
    return speeds  # Vitesses en centième de degré par heure

def genesis_points(tracks):
    """
    Return the first point of each track

    Parameters
    ----------
    tracks (pd.Dataframe)

    Returns
    -------
    pd.Dataframe
    """
    return (
        tracks.sort_values("time")
        .groupby("track_id")[
            ["hemisphere", "basin", "season", "month", "time", "lon", "lat"]
        ]
        .first()
    )

def density_map(tracks, bin_size=5, N_seasons = 64) :
    x = np.arange(0, 360+bin_size, bin_size)
    y = np.arange(-90, 90+bin_size, bin_size)
    H, X, Y = np.histogram2d(tracks.lon, tracks.lat, bins = [x, y])
    da = xr.Dataset(dict(hist2d=(["lon", "lat"], np.array(H))), coords = dict(lon=(["lon"], (X[:-1]+X[1:])/2), lat=(["lat"], (Y[:-1]+Y[1:])/2))).hist2d
    da = da/N_seasons
    return da.where(da != 0).T
def density_map_med(tracks, bin_size=2, N_seasons = 64) :
    x = np.arange(-20, 45+bin_size, bin_size)
    y = np.arange(25, 50+bin_size, bin_size)
    H, X, Y = np.histogram2d(tracks.lon, tracks.lat, bins = [x, y])
    da = xr.Dataset(dict(hist2d=(["lon", "lat"], np.array(H))), coords = dict(lon=(["lon"], (X[:-1]+X[1:])/2), lat=(["lat"], (Y[:-1]+Y[1:])/2))).hist2d
    da = da/N_seasons
    return da.where(da != 0).T