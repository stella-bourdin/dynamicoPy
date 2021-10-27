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