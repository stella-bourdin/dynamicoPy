import pandas as pd
import pickle as pkl
import pkg_resources
import numpy as np
from .utils import *


def _clean_ibtracs(
    raw_file="tests/ibtracs.since1980.list.v04r00_05092021.csv",
    csv_output="dynamicopy/_data/ibtracs.since1980.cleaned.csv",
    pkl_output="dynamicopy/_data/ibtracs.pkl",
):
    """
    Function used to post-treat ibtracs _data into a lighter file

    Parameters
    ----------
    raw_file: The csv file from the ibtracs database
    csv_output: The name of the csv file to be saved
    pkl_output: The name of the pickle file to be saved

    Returns
    -------
    pd.DataFrame
        The cleaned ibtracs dataset
    + saves csv and pkl file
    """

    ib = pd.read_csv(
        raw_file,
        na_values=["", " "],
        header=0,
        skiprows=[1],
        usecols=[
            "SID",
            "SEASON",
            "BASIN",
            "SUBBASIN",
            "ISO_TIME",
            "NATURE",
            "TRACK_TYPE",
            "LON",
            "LAT",
            "USA_SSHS",
            "WMO_WIND",
            "USA_WIND",
            "TOKYO_WIND",
            "CMA_WIND",
            "REUNION_WIND",
            "BOM_WIND",
            "NADI_WIND",
            "WELLINGTON_WIND",
            "WMO_PRES",
            "USA_PRES",
            "TOKYO_PRES",
            "CMA_PRES",
            "HKO_PRES",
            "NEWDELHI_PRES",
            "REUNION_PRES",
            "BOM_PRES",
            "NADI_PRES",
            "WELLINGTON_PRES",
        ],
        converters={
            "SID": str,
            "SEASON": int,
            "BASIN": str,
            "SUBBASIN": str,
            "LON": float,
            "LAT": float,
        },
        parse_dates=["ISO_TIME"],
    )
    ib = ib[~ib.TRACK_TYPE.str.startswith("spur")]
    ib = ib[ib.SEASON < 2020]
    ib = ib[(ib.SEASON > 1980) | (ib.LAT > 0)]
    ib["WIND10"] = np.where(
        ~ib.WMO_WIND.isna(),
        ib.WMO_WIND,
        ib[
            ["TOKYO_WIND", "REUNION_WIND", "BOM_WIND", "NADI_WIND", "WELLINGTON_WIND"]
        ].mean(axis=1, skipna=True),
    )
    ib["WIND10"] = np.where(
        ib.WIND10.isna(), ib.USA_WIND / 1.12, ib.WIND10
    )  # Conversion rate in the IBTrACS doc
    ib["WIND10"] = np.where(
        ib.WIND10.isna(), ib.CMA_WIND / 1.08, ib.WIND10
    )  # Conversion rate determined through a linear regression
    ib["WIND10"] *= 0.514  # Conversion noeuds en m/s
    tcs = (
        ib.groupby("SID")["WIND10"].max()[ib.groupby("SID")["WIND10"].max() >= 17].index
    )
    ib = ib[ib.SID.isin(tcs)]  # Filter tracks not reaching 17 m/s
    tcs = (
        ib.groupby("SID")["ISO_TIME"]
        .count()[ib.groupby("SID")["ISO_TIME"].count() >= 4]
        .index
    )
    ib = ib[ib.SID.isin(tcs)]  # Filter tracks not reaching 17 m/s
    ib["PRES"] = np.where(
        ~ib.WMO_PRES.isna(),
        ib.WMO_PRES,
        ib[
            [
                "USA_PRES",
                "TOKYO_PRES",
                "CMA_PRES",
                "HKO_PRES",
                "NEWDELHI_PRES",
                "REUNION_PRES",
                "BOM_PRES",
                "NADI_PRES",
                "WELLINGTON_PRES",
            ]
        ].mean(axis=1, skipna=True),
    )
    ib = ib.rename(columns={col: col.lower() for col in ib.columns}).rename(
        columns={
            "usa_sshs": "sshs",
            "sid": "track_id",
            "pres": "slp",
            "iso_time": "time",
        }
    )
    ib.loc[ib.lon < 0, "lon"] += 360
    ib["hemisphere"] = np.where(ib.lat > 0, "N", "S")
    ib["basin"] = (
        ib.basin.replace("EP", "ENP").replace("WP", "WNP").replace("NA", "NATL")
    )
    ib["day"] = ib.time.dt.day
    ib["month"] = ib.time.dt.month
    ib["year"] = ib.time.dt.year
    ib = add_season(ib)
    ib["basin"] = get_basin(ib.lon, ib.lat)
    ib = ib[
        [
            "track_id",
            "time",
            "lon",
            "lat",
            "hemisphere",
            "basin",
            "season",
            "sshs",
            "slp",
            "wind10",
            "year",
            "month",
            "day",
        ]
    ]
    # Save
    ib.to_csv(csv_output)
    with open(pkl_output, "wb") as handle:
        pkl.dump(ib, handle)
    return ib


def load_ibtracs():
    """
    Function to load the cleaned ibtracs dataset included in the package

    Returns
    -------
    pd.DataFrame
        The ibtracs dataset
    """
    stream = pkg_resources.resource_stream(
        __name__, "../_data/ibtracs.since1980.cleaned.csv"
    )
    ib = pd.read_csv(
        stream,
        keep_default_na=False,
        index_col=0,
        na_values=["", " "],
        dtype={"slp": float, "wind10": float, "season": str},
        parse_dates=["time"],
    )
    ib["basin"] = get_basin(ib.lon, ib.lat)
    ib["ET"] = False
    return ib
