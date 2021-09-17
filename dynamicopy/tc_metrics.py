import numpy as np
from .utils import hist2d

"""
This module implements functions to compute standard metrics over tc tracks datasets (observed or detected).

In all functions, tracks represent a dataset of TC points as issued by the load_*track of the cyclones module.
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

def storm_stats(tracks):
    storms = tracks.groupby(['track_id'])[['hemisphere', 'basin', 'season', 'month']].agg(lambda x: x.value_counts().index[0]).reset_index()
    storms = storms.merge(
            tracks.groupby(['track_id'])[["sshs", "wind10"]].max().reset_index())
    storms = storms.merge(tracks.groupby(['track_id'])[["slp"]].min().reset_index())
    storms = storms.merge((tracks.groupby(['track_id'])[['time']].count()/4).reset_index())
    tracks[["ACE"]] = tracks[["wind10"]]**2 * 1e-4
    tracks[["PDI"]] = tracks[["wind10"]] ** 3
    storms = storms.merge(tracks.groupby(['track_id'])[["ACE", "PDI"]].sum().reset_index())
    storms = storms.merge(storms[["track_id", 'wind10']].merge(tracks[["track_id", "wind10", "lat", "time"]]).groupby("track_id").agg(lambda t: t.mean()).reset_index().rename(columns = {"lat":"lat_wind", "time":"time_wind"}), how = "outer")
    storms = storms.merge(storms[["track_id", 'slp']].merge(tracks[["track_id", "slp", "lat", "time"]]).groupby("track_id").agg(lambda t: t.mean()).reset_index().rename(columns = {"lat":"lat_slp", "time":"time_wind"}), how = "outer")
    return storms

def genesis_points(tracks):
    return tracks.sort_values("time"
                              ).groupby("track_id")[["hemisphere", "basin", "season", "month", "time", "lon", "lat"]
                                ].first()

def propagation_speeds(tracks):
    speeds = {}
    for t in tracks.track_id.unique():
        T = tracks[tracks.track_id == t]
        dlon = T[:-1].lon.values - T[1:].lon.values
        dlat = T[:-1].lat.values - T[1:].lat.values
        dist = np.sqrt(dlon ** 2 + dlat ** 2) # Vérifier les aspects de GCD etc.
        speeds[t] = dist * 100 / 6
    return speeds #Vitesses en centième de degré par heure

def u10_map(tracks, resolution = 8):
    tracks["lon_block"] = (tracks.lon // resolution) * resolution + resolution/2
    lon_blocks = np.arange(resolution/2, 360.1, resolution)
    tracks["lat_block"] = (np.abs(tracks.lat) // resolution * np.sign(tracks.lat)) * resolution + resolution/2
    lat_blocks = np.array(list(np.flip(-1*np.arange(resolution/2, 90, resolution))) + list(np.arange(resolution/2, 90, resolution)))
    map = tracks.groupby(["lon_block", "lat_block"])[["wind10"]].max(
        ).pivot_table(index="lat_block", columns="lon_block", fill_value = 0).loc[:,"wind10"]
    for l in lon_blocks[~np.isin(lon_blocks, map.columns)] :
        map[l] = 0
    for l in lat_blocks[~np.isin(lat_blocks, map.index)] :
        map.loc[l] = 0
    map = map.sort_index().reindex(sorted(map.columns), axis = 1)
    return map

def slp_map(tracks, resolution = 8):
    tracks["lon_block"] = (tracks.lon // resolution) * resolution + resolution/2
    lon_blocks = np.arange(resolution/2, 360.1, resolution)
    tracks["lat_block"] = (np.abs(tracks.lat) // resolution * np.sign(tracks.lat)) * resolution + resolution/2
    lat_blocks = np.array(list(np.flip(-1*np.arange(resolution/2, 90, resolution))) + list(np.arange(resolution/2, 90, resolution)))
    map = tracks.groupby(["lon_block", "lat_block"])[["slp"]].min(
        ).pivot_table(index="lat_block", columns="lon_block", fill_value = 0).loc[:,"slp"]
    for l in lon_blocks[~np.isin(lon_blocks, map.columns)] :
        map[l] = 0
    for l in lat_blocks[~np.isin(lat_blocks, map.index)] :
        map.loc[l] = 0
    map = map.sort_index().reindex(sorted(map.columns), axis = 1)
    return map

#TODO : u10 et slp sont biaisés par un important nombre de couples (0,0)
def spatial_correlations(tracks1, tracks2, method="pearson", res=8):
    """
    Metric implemented from Zarzycki et al. 2021

    Parameters
    ----------
    tracks1
    tracks2
    method

    Returns
    -------

    """
    lons, lats, hist_track_1 = hist2d(tracks1,resolution=res)
    lons, lats, hist_track_2 = hist2d(tracks2,resolution=res)
    lons, lats, hist_ACE_1 = hist2d(tracks1[~tracks1.ACE.isna()], weights = tracks1[~tracks1.ACE.isna()].ACE, resolution=res)
    lons, lats, hist_ACE_2 = hist2d(tracks2[~tracks2.ACE.isna()], weights = tracks2[~tracks2.ACE.isna()].ACE,resolution=res)
    map_u10_1 = u10_map(tracks1, resolution=res)
    map_u10_2 = u10_map(tracks2, resolution=res)
    map_slp_1 = slp_map(tracks1, resolution=res)
    map_slp_2 = slp_map(tracks2, resolution=res)
    lons, lats, hist_gen_1 = hist2d(genesis_points(tracks1), resolution = res)
    lons, lats, hist_gen_2 = hist2d(genesis_points(tracks2), resolution = res)
    if method == "pearson" :
        f = pearsonr
    else :
        f = spearmanr
    track_correlation = f(hist_track_1.flatten(), hist_track_2.flatten())[0]
    ACE_correlation = f(hist_ACE_1.flatten(), hist_ACE_2.flatten())[0]
    u10_correlation = f(map_u10_1.values.flatten(), map_u10_2.values.flatten())[0]
    slp_correlation = f(map_slp_1.values.flatten(), map_slp_2.values.flatten())[0]
    gen_correlation = f(hist_gen_1.flatten(), hist_gen_2.flatten())[0]
    return {"track":track_correlation, "ACE":ACE_correlation, "u10":u10_correlation,
            "slp":slp_correlation, "gen":gen_correlation}

# TODO : Implementer pour LMI ?
def temporal_correlation(tracks1, tracks2, method = "peearson") :
    ss1 = storm_stats(tracks1)
    ss2 = storm_stats(tracks2)
    count_1 = ss1.groupby("month")['track_id'].count()
    count_2 = ss2.groupby("month")['track_id'].count()
    tcd_1 = ss1.groupby("month")['time'].sum()
    tcd_2 = ss2.groupby("month")['time'].sum()
    ace_1 = ss1.groupby("month")['ACE'].sum()
    ace_2 = ss2.groupby("month")['ACE'].sum()

    if method == "pearson" :
        f = pearsonr
    else :
        f = spearmanr
    count_corr = f(count_1, count_2)
    tcd_corr = f(tcd_1, tcd_2)
    ace_corr = f(ace_1, ace_2)
    return {"count":count_corr, "tcd":tcd_corr, "ACE":ace_corr}