import pandas as pd
import pickle as pkl
import pkg_resources
import datetime
import numpy as np
from .utils import *
from .lifecycle import identify_lifecycle
import dynamicopy

def _clean_ibtracs(
    raw_file="tests/ibtracs.since1980.list.v04r00_8Jan23.csv",
    csv_output="dynamicopy/_data/ibtracs.since1980.cleaned.csv",
    pkl_output="dynamicopy/_data/ibtracs.pkl",
    six_hourly=True,
    threshold_wind=True,
    max_season=2022,
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
            "NAME",
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
            "USA_RMW",
            "BOM_RMW",
            "REUNION_RMW",
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
    # Remove spur tracks
    ib = ib[~ib.TRACK_TYPE.str.startswith("spur")]

    # Season selection
    ib = ib[ib.SEASON < max_season]
    ib = ib[ib.SEASON > 1980]

    # All about wind
    ## Select from the data & convert
    ### WMO wind if available, with conversion in the basins where it is necessary
    ib["WIND10"] = ib.WMO_WIND
    ib["WIND10"] = np.where(ib.BASIN.isin(["EP", "NA"]), ib.WMO_WIND / 1.12, ib.WIND10)
    ib["WIND10"] = np.where(ib.BASIN.isin(["NI"]), ib.WIND10 / 1.08, ib.WIND10)
    ### Wind from a WMO center reporting 10-minutes averaged wind speed if available
    ib["WIND10"] = np.where(
        ib.WIND10.isna(),
        ib[
            ["TOKYO_WIND", "REUNION_WIND", "BOM_WIND", "NADI_WIND", "WELLINGTON_WIND"]
        ].mean(axis=1, skipna=True),
        ib.WIND10,
    )
    ### Otherwise USA wind if available
    ib["WIND10"] = np.where(
        ib.WIND10.isna(), ib.USA_WIND / 1.12, ib.WIND10
    )  # Conversion rate in the IBTrACS doc
    ### Otherwise CMA wind if available
    ib["WIND10"] = np.where(
        ib.WIND10.isna(), ib.CMA_WIND / 1.08, ib.WIND10
    )  # Conversion rate determined through a linear regression

    ib["WIND10"] *= 0.514  # Conversion noeuds en m/s

    # Select pressure
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

    # Convert RMW from nmiles to km
    ib["USA_RMW"] *= 1.852

    # Rename columns
    ib = ib.rename(columns={col: col.lower() for col in ib.columns}).rename(
        columns={
            "sid": "track_id",
            "pres": "slp",
            "iso_time": "time",
        }
    )

    # Geographical details
    ib.loc[ib.lon < 0, "lon"] += 360
    ib["hemisphere"] = np.where(ib.lat > 0, "N", "S")
    ib["basin"] = get_basin(ib.lon, ib.lat)

    # All about time
    ib["hour"] = ib.time.dt.hour
    ib["day"] = ib.time.dt.day
    ib["month"] = ib.time.dt.month
    ib["year"] = ib.time.dt.year
    ib = add_season(ib)

    ## Filter 6-hourly
    origin = datetime.datetime(1800, 1, 1, 0, 0, 0)
    if six_hourly:
        ib = ib[(ib.time - origin).dt.total_seconds() % (6 * 60 * 60) == 0]

    ## Filter tracks not reaching 16 m/s
    if threshold_wind:
        tcs = (
            ib.groupby("track_id")["wind10"]
            .max()[ib.groupby("track_id")["wind10"].max() >= 16]
            .index
        )
        ib = ib[ib.track_id.isin(tcs)]

    ## Filter tracks lasting for at least 4 time steps
    tcs = (
        ib.groupby("track_id")["time"]
        .count()[ib.groupby("track_id")["time"].count() >= 4]
        .index
    )
    ib = ib[ib.track_id.isin(tcs)]

    # Compute SSHS classification according to Klotzbach
    ib["sshs"] = sshs_from_pres(ib.slp)

    ib = ib[
        [
            "track_id",
            "time",
            "lon",
            "lat",
            "name",
            "hemisphere",
            "basin",
            "season",
            "sshs",
            "nature",
            "slp",
            "wind10",
            "year",
            "month",
            "day",
            "hour",
            "usa_rmw"
        ]
    ]

    ib["ET"] = False

    ib = identify_lifecycle(ib)

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
    return ib
