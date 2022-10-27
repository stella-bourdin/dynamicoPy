import xarray as xr
import numpy as np

def compute_STJ_latmin(file_ws, file_u, outfile):
    """

    Parameters
    ----------
    outfile: prefix of the output file, to which will be appended _[N/S]H.nc
    file_ws: File of wind speed THAT HAS ALREADY GONE THROUGH ROLLING AVG WITH cdo -runmean
    file_u: File of zonal wind speed THAT HAS ALREADY GONE THROUGH ROLLING AVG WITH cdo -runmean

    *** History of the two files, in the preliminary job ***
    ufile=$FOLDER/ua/gr/files/*/ua_6hrPlevPt_*_highresSST-present_r1i1p1f1_gr_${yr_min}01010600-*.nc
    vfile=$FOLDER/va/gr/files/*/va_6hrPlevPt_*_highresSST-present_r1i1p1f1_gr_${yr_min}01010600-*.nc

    ncks -d plev,25000.0 -d lat,-60.0,60.0 $ufile $TMP/u200_${NAME}_${yr_min}.nc
    ncks -d plev,25000.0 -d lat,-60.0,60.0 $vfile $TMP/v200_${NAME}_${yr_min}.nc

    cdo merge $TMP/u200_${NAME}_${yr_min}.nc $TMP/v200_${NAME}_${yr_min}.nc $TMP/winds200_${NAME}_${yr_min}.nc

    ncap2 -s 'u200_abs=abs(ua)' $TMP/u200_${NAME}_${yr_min}.nc $TMP/u_abs_${NAME}_${yr_min}.nc
    ncap2 -s 'ws200=sqrt(ua^2+va^2)' $TMP/winds200_${NAME}_${yr_min}.nc $TMP/ws_${NAME}_${yr_min}.nc

    cdo -runmean,120 $TMP/u_abs_${NAME}_${yr_min}.nc $TMP/u200_runmean_${NAME}_${yr_min}.nc   # --> This is u_file
    cdo -runmean,120 $TMP/ws_${NAME}_${yr_min}.nc $TMP/ws200_runmean_${NAME}_${yr_min}.nc     # --> This is ws_file

    Returns
    -------

    """

    speed_roll = xr.open_dataset(file_ws).ws200.squeeze()
    u_roll = xr.open_dataset(file_u).u200_abs.squeeze()

    lat_decrease = (u_roll.lat[0] > u_roll.lat[1])

    NH_STJ = ((speed_roll.where(speed_roll.lat >= 0) > 25) & (u_roll.where(u_roll.lat >= 0) > 15))
    if lat_decrease : NH_STJ = NH_STJ.reindex(lat = NH_STJ.lat[::-1]);

    NH_lim = NH_STJ.idxmax("lat")
    NH_lim = NH_lim.where(NH_lim > 0)
    NH_lim["lon"] = np.where(NH_lim.lon > 180, NH_lim.lon - 360, NH_lim.lon)
    NH_lim = NH_lim.sortby("lon")

    NH_lim.to_netcdf(outfile + "_NH.nc", "w")

    SH_STJ = ((speed_roll.where(speed_roll.lat <= 0) > 25) & (u_roll.where(u_roll.lat <= 0) > 15))
    if ~lat_decrease : SH_STJ = SH_STJ.reindex(lat=SH_STJ.lat[::-1]);

    SH_lim = SH_STJ.idxmax("lat")
    SH_lim = SH_lim.where(SH_lim < 0)
    SH_lim["lon"] = np.where(SH_lim.lon > 180, SH_lim.lon - 360, SH_lim.lon)
    SH_lim = SH_lim.sortby("lon")

    SH_lim.to_netcdf(outfile + "_SH.nc", "w")

    return NH_lim, SH_lim