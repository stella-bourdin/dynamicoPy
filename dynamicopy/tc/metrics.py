import numpy as np

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

def get_freq(tracks):
    storms = tracks.groupby("track_id")[["season", "hemisphere", "basin"]].agg(
        lambda x: x.value_counts().index[0]
    )
    storms["sshs"] = tracks.groupby("track_id")["sshs"].max()

    SH = (
        storms[storms.hemisphere == "S"] \
        .groupby(["season", "sshs"])[["basin"]] \
        .count() \
        .reset_index() \
        .pivot_table(index=["sshs"], columns="season", fill_value=0) \
        .melt(ignore_index=False) \
        .iloc[:, 2:] \
        .groupby("sshs") \
        .mean() \
        .assign(basin="S") \
        .reset_index() \
    )

    NH = (
        storms[storms.hemisphere == "N"]
        .groupby(["season", "sshs"])[["basin"]]
        .count()
        .reset_index()
        .pivot_table(index=["sshs"], columns="season", fill_value=0)
        .melt(ignore_index=False)
        .iloc[:, 2:]
        .groupby("sshs")
        .mean()
        .assign(basin="N")
        .reset_index()
    )

    basins = (
        storms
        .groupby(["season", "sshs", "basin"])
        .count()
        .reset_index()
        .pivot_table(index=["basin", "sshs"], columns="season", fill_value=0)
        .melt(ignore_index=False)
        .iloc[:, 2:]
        .groupby(["sshs", "basin"])
        .mean()
        .reset_index()
    )

    freq = (
        SH.append(NH)
        .append(basins)
        .pivot_table(
            index="basin", columns="sshs", fill_value=0.0, margins=True, aggfunc=np.sum
        )
        .drop("All")
    )
    freq.loc["global"] = freq.loc["S"] + freq.loc["N"]
    freq.columns = freq.columns.get_level_values(1)
    return freq.reindex(
        ["global", "N", "WNP", "ENP", "NI", "NATL", "S", "SP", "SI", "SA"]
    )

def prop_intense(freq):
    cat_45_cols = list(freq.columns[:-1] >= 4) + [False]
    freq_45 = freq.loc[:, cat_45_cols].sum(axis=1)
    prop_45 = freq_45 / freq.loc[:, "All"]
    return freq[["All"]].assign(intense=freq_45).assign(prop=prop_45)

def storm_stats(tracks):
    storms = (
        tracks.groupby(["track_id"])[["hemisphere", "basin", "season", "month"]]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
    )
    storms = storms.merge(
        tracks.groupby(["track_id"])[["sshs", "wind10"]].max().reset_index()
    )
    storms = storms.merge(tracks.groupby(["track_id"])[["slp"]].min().reset_index())
    storms = storms.merge(
        (tracks.groupby(["track_id"])[["time"]].count() / 4).reset_index()
    )
    storms = storms.merge(
        storms[["track_id", "wind10"]]
        .merge(tracks[["track_id", "wind10", "lat", "time"]])
        .groupby("track_id")
        .agg(lambda t: t.mean())
        .reset_index()
        .rename(columns={"lat": "lat_wind", "time": "time_wind"}),
        how="outer",
    )
    storms = storms.merge(
        storms[["track_id", "slp"]]
        .merge(tracks[["track_id", "slp", "lat", "time"]])
        .groupby("track_id")
        .agg(lambda t: t.mean())
        .reset_index()
        .rename(columns={"lat": "lat_slp", "time": "time_slp"}),
        how="outer",
    )
    return storms