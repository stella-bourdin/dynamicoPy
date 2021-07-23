import numpy as np

"""
This module implements functions to compute standard metrics over tc tracks datasets (observed or detected)
"""

def tc_count(tracks):
    storms = tracks.groupby("track_id")[["hemisphere", "basin"]].agg(lambda x:x.value_counts().index[0])
    storms["sshs"] = tracks.groupby("track_id")["sshs"].max()
    B = storms.groupby(["sshs", "hemisphere"])[["basin"]].count().reset_index().pivot_table(index = "hemisphere", columns="sshs", fill_value = 0.0, margins=True, aggfunc=np.sum).rename(columns={"basin":"count"})
    C = storms.groupby(["sshs", "basin"])[["hemisphere"]].count().reset_index().pivot_table(index = "basin", columns="sshs", fill_value = 0.0, margins = True, aggfunc=np.sum).rename(columns={"hemisphere":"count"})
    return B.append(C).drop_duplicates().rename(index={"All":"global"}).reindex(['global', 'N', 'WNP', 'ENP', 'NI', 'NATL', 'S', 'SP', 'SI', 'SA'])

def get_freq(tracks):
    storms = tracks.groupby("track_id")[["season", "hemisphere", "basin"]].agg(lambda x:x.value_counts().index[0])
    storms["sshs"] = tracks.groupby("track_id")["sshs"].max()

    SH = storms[storms.hemisphere == "S"].groupby(["season", "sshs"])[["basin"]].count().reset_index(
        ).pivot_table(index=["sshs"], columns ="season", fill_value = 0).melt(ignore_index=False).iloc[:,1:].groupby("sshs").mean().assign(basin="S").reset_index()
    basins_S = storms[storms.hemisphere == "S"].groupby(["season", "sshs", "basin"]).count().reset_index(
        ).pivot_table(index=["basin", "sshs"], columns ="season", fill_value = 0).melt(ignore_index=False).iloc[:,1:].groupby(["sshs", "basin"]).mean().reset_index()

    NH = storms[storms.hemisphere == "N"].groupby(["season", "sshs"])[["basin"]].count().reset_index(
    ).pivot_table(index=["sshs"], columns="season", fill_value=0).melt(ignore_index=False).iloc[:, 1:].groupby(
        "sshs").mean().assign(basin="N").reset_index()
    basins_N = storms[storms.hemisphere == "N"].groupby(["season", "sshs", "basin"]).count().reset_index(
        ).pivot_table(index=["basin", "sshs"], columns ="season", fill_value = 0).melt(ignore_index=False).iloc[:,1:].groupby(["sshs", "basin"]).mean().reset_index()

    freq = SH.append(basins_S).append(NH).append(basins_N).pivot_table(index="basin", columns="sshs", fill_value=0.0, margins=True, aggfunc=np.sum).drop("All")
    freq.loc["global"] = freq.loc["S"] + freq.loc["N"]
    freq.columns = freq.columns.get_level_values(1)
    return freq.reindex(['global', 'N', 'WNP', 'ENP', 'NI', 'NATL', 'S', 'SP', 'SI', 'SA'])

def prop_intense(freq):
    cat_45_cols = list(freq.columns[:-1]>=4) + [False]
    freq_45 = freq.loc[:,cat_45_cols].sum(axis = 1)
    prop_45 = freq_45 / freq.loc[:,"All"]
    return freq[["All"]].assign(intense= freq_45).assign(prop=prop_45)

