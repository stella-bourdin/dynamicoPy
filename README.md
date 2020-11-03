# DynamicoPy

This is a package designed to ease your analysis of climate models data with netCDF file. 

You can install this package with pip : `pip install dynamicopy`
PyPI release here : https://pypi.org/project/dynamicopy/ 

How to use the package ? 
Imagine you have a u.nc netCDF4 file. 

1 / Load the data
`u = var_load('u', 'u.nc')`

2 / Load its coordinates
`lon, lat = get_lon_lat('u.nc')`

3 / Plot it on a map
`lon_lat_plot_map(lon, lat, u)`