"""
Compiles the hydrological database required for modelling streamflow based on
meteorological variables, using era5 data retrieval script and hydrological
data handling functions from the apollo library.

@author: robertrouse
"""

# Import cdsapi and create a Client instance
import os
import glob
import paths
import xarray as xr
import pandas as pd
import numpy as np
from apollo import era5 as er
from apollo import hydropoint as hp
from train_model import load_data

### Specify meteorological variables and spatiotemporal ranges
area = [60, -8, 48, 4]
yyyy = [str(y) for y in range(1979,2022,1)]
mm = [str(m) for m in range(1,13,1)]
dd = [str(d) for d in range(1,32,1)]
hh = [str(t).zfill(2) + ':00' for t in range(0, 24, 1)]
met = ['total_precipitation','snowmelt', 'temperature','u_component_of_wind',
       'v_component_of_wind','relative_humidity']
       #'volumetric_soil_water_layer_1','volumetric_soil_water_layer_2',
       #'volumetric_soil_water_layer_3','volumetric_soil_water_layer_4']


### Download meteorological variable sets from Copernicus Data Store

## Download Rainfall, including snow melt separately (midnight to midnight and shifted daily from 9 to 9)
for yy in yyyy:

    for m in mm:

        filename = paths.WEATHER_UK + '/Rainfall/Rainfall_' + str(yy) + str(m)
        rain_query = er.query(filename, 'reanalysis-era5-single-levels', met[:2],
                                    area, yy, m, dd, hh)

        if not os.path.exists(filename + '.nc'):
            print('downloading ', filename)
            rain_data = er.era5(rain_query).download()

            # Save the hourly precipitation data and aggregate daily (from midnight to midnight)
            er.aggregate_mean(str(rain_query['file_stem']) + '.nc',
                      str(rain_query['file_stem']) + '_aggregated.nc')

"""
    # Shift the daily aggregation (from 9am to 9am)
    if not os.path.exists(filename + '_9to9.nc'):
        print('shifting', filename)
        er.shift_time_9am_to_9am(str(rain_query['file_stem']) + '.nc',
                                 int(yy),
                                 str(rain_query['file_stem']) + '_9to9.nc')
        er.aggregate_mean(str(rain_query['file_stem']) + '_9to9.nc',
                          str(rain_query['file_stem']) + '_aggregated_9to9.nc',
                          shift=9)
"""

# Combine the daily midnight-midnight precipitation
if not os.path.exists(paths.RAINFALL_UK):
    full_rain_data = xr.open_mfdataset(paths.WEATHER_UK + '/Rainfall/Rainfall_*_aggregated.nc', concat_dim='time',
                                       combine='nested')
    full_rain_data.to_netcdf(path=paths.RAINFALL_UK)

"""
# Combine the daily 9 to 9 precipiptation
if not os.path.exists(paths.RAINFALL_UK_SHIFTED):
    full_shifted_rain_data = xr.open_mfdataset(paths.WEATHER_UK + '/Rainfall/Rainfall_*_aggregated_9to9.nc',
                                               concat_dim='time', combine='nested')
    full_shifted_rain_data.to_netcdf(path=paths.RAINFALL_UK_SHIFTED)

# Hourly precipitation for the 9 to 9 shift
if not os.path.exists(paths.RAINFALL_HOURLY_UK_SHIFTED):
    all_files = glob.glob(paths.WEATHER_UK + '/Rainfall/Rainfall_*_9to9.nc')
    filtered_files = [f for f in all_files if 'aggregated' not in f]

    full_rain_data = xr.open_mfdataset(filtered_files, concat_dim='time', combine='nested')
    if not os.path.exists(paths.RAINFALL_HOURLY_UK_SHIFTED):
        full_rain_data.to_netcdf(path=paths.RAINFALL_HOURLY_UK_SHIFTED)
"""

## Download Pressure data (converted to windspeed and humidity later)
for yy in yyyy:
    filename = paths.WEATHER_UK + '/Pressure/Pressure_' + str(yy)

    if not os.path.exists(filename + '1000hPa.nc'):
        print("downloading ", filename)
        pressure_query = er.query(filename, 'reanalysis-era5-pressure-levels', met[2:6],
                          area, yy, mm, dd, '12:00', ['1000'])
        pressure_data = er.era5(pressure_query).download()

full_pressure_data = xr.open_mfdataset(paths.WEATHER_UK + '/Pressure/Pressure_*.nc', concat_dim='time', combine='nested')
if not os.path.exists(paths.PRESSURE_UK):
    full_pressure_data.to_netcdf(path=paths.PRESSURE_UK)

"""
## Download Soil Moisture data (4 different soil layers)
for yy in yyyy:
    filename = paths.SURFACE_UK + '/Soil_Moisture_' + str(yy)

    if not os.path.exists(filename + '.nc'):
        print("downloading ", filename)
        soil_moisture_query = er.query(filename, 'reanalysis-era5-land', met[6:], area, yy, mm, dd, '12:00')
        soil_moisture_data = er.era5(soil_moisture_query).download()

full_soil_moisture_data = xr.open_mfdataset(paths.SURFACE_UK + '/Soil_Moisture_*.nc', concat_dim='time', combine='nested')
if not os.path.exists(paths.SOIL_MOISTURE_UK):
    full_soil_moisture_data.to_netcdf(path=paths.SOIL_MOISTURE_UK)"""

### Produce lumped regression files per catchment
domain_weather = xr.open_mfdataset([paths.RAINFALL_UK_SHIFTED,
                                    paths.PRESSURE_UK])
domain_weather= domain_weather.astype(np.float32)
#surface_data = xr.open_dataset(paths.SOIL_MOISTURE_UK)
#surface_data = surface_data.astype(np.float32)

domain_rain = xr.open_dataset(paths.RAINFALL_UK_SHIFTED)

#domain_rain_HR = xr.open_mfdataset(paths.RAINFALL_UK_SHIFTED_HR)
#domain_rain_hourly = xr.open_mfdataset([paths.RAINFALL_HOURLY_UK_SHIFTED])

db = pd.read_csv(paths.DATA + '/Catchments_Database.csv')

EXT = '_9to9'
OUT_FP = paths.CATCHMENT_BASINS

for i in range(len(db)):
    print('start with ', i)

    db_path = paths.CATCHMENT_BASINS + '/' + str(db.iloc[i,0])

    test = hp.hydrobase(db.iloc[i,0],
                        db_path + '/' + db.iloc[i,3],
                        db_path + '/' + db.iloc[i,4])

    # Normal Era5 data
    cache = test.output_era5_file(domain_weather, surface_data, 28,
                                out_fp=OUT_FP,
                                ext=EXT,
                                interpolation_method='linear',
                                reload=False)

    # Hourly precipitation
    hourly_cache = test.output_hourly_rain_file(domain_rain_hourly,
                                              hours_shift=9,
                                              out_fp=paths.CATCHMENT_BASINS,
                                              ext=EXT,
                                              interpolation_method='linear',
                                              reload=False)

    # Use era5 files as reference to alterate the precipitation with both NRFA and surface interpolated values
    rain_columns = (['Rain'] + ['Rain-' + f'{d + 1}' for d in range(27)] +
                    ['Rain_28_Mu', 'Rain_90_Mu', 'Rain_180_Mu'])
    ref_path = (paths.CATCHMENT_BASINS + '/' + str(db.iloc[i,0]) + '/' +
               str(db.iloc[i,0]) + f"_lumped{EXT}_linear.csv")
    rf_ref = load_data.load_data(ref_path, verbose=False).drop(columns=rain_columns)

    # NRFA precipitation
    rf_nrfa = pd.read_csv(paths.CATCHMENT_BASINS + f'/{str(db.iloc[i,0])}/{str(db.iloc[i,0])}_cdr.csv')
    nrfa_cache = test.output_nrfa_file(rf_ref=rf_ref,
                                       rf_nrfa=rf_nrfa,
                                       out_fp=OUT_FP,
                                       ext=EXT,
                                       reload=False)

    # Surface interpolation
    surf_interp_cache = test.output_surface_interpolated_file(domain_rain=domain_rain,
                                                              rf_ref=rf_ref,
                                                              out_fp=OUT_FP,
                                                              ext=EXT,
                                                              multiplicator=1000*24,
                                                              reload=False)

    # Surface interpolation high resolution data
    surf_interp_cache_HR = test.output_surface_interpolated_file(domain_rain=domain_rain_HR,
                                                              rf_ref=rf_ref,
                                                              out_fp=OUT_FP,
                                                              ext=EXT + '_HR',
                                                              resolution=0.1,
                                                              multiplicator=1000*24,
                                                              reload=False)
