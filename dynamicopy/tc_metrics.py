import numpy as np

"""
This module implements functions to compute standard metrics over tc tracks datasets (observed or detected)
"""

def tc_count(tracks):
    storms = tracks.groupby("track_id")[["hemisphere", "basin"]].max()  # Récupérer la fonction pour le plus de point
    return pd.Series([len(storms)], ["global"]).append(
        storms.groupby("hemisphere")["basin"]
        .count()
        .append(storms.groupby("basin")["hemisphere"].count())
    )

def freq(tracks):
    storms = tracks.groupby("track_id")[
        ["year", "season", "hemisphere", "basin"]].max()  # Récupérer la fonction pour le plus de point
    return pd.Series([storms.groupby("year")['hemisphere'].count()], ["global"]).append(
        storms.groupby(["season", "hemisphere"])["basin"].count().groupby("hemisphere").mean()
            .append(storms.groupby(["season", "basin"])["hemisphere"].count().groupby("basin").mean())
    )