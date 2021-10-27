import numpy as np

def add_season(tracks):
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
    # _ = np.where(
    #    (hemi == "S"),
    #    np.core.defchararray.add(season.astype(str), np.array(["-"] * len(season))),
    #    season,
    # ).astype(str)
    # season = np.where(
    #    (hemi == "S"), np.core.defchararray.add(_, (season + 1).astype(str)), season
    # )
    group["season"] = season.astype(int)
    tracks = tracks.join(group[["season"]], on="track_id")
    return tracks