import os
import paths
import pandas as pd
import geopandas as gpd

from shapely import Point
from apollo import metrics as me
from apollo import osgconv as osg


def find_longest_diagonal(polygon):
    box = polygon.minimum_rotated_rectangle
    x, y = box.exterior.coords.xy
    edge_length = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))
    return max(edge_length)


def calculate_latitude(geometry):
    points = geometry.centroid
    lat, lon = osg.BNG_2_latlon(points.x, points.y)
    return lat


def calculate_slope_gradient(gdf, metadata):
    gdf['Cross_distance'] = gdf['Geometry'].apply(lambda geometry: find_longest_diagonal(geometry))
    gdf['Area'] = gdf['Geometry'].apply(lambda geometry: geometry.area)

    gdf['Minimum altitude'] = gdf['Station'].apply(
        lambda station: metadata[metadata['Station number'] == int(station)]['Minimum altitude'].iloc[0])
    gdf['Maximum altitude'] = gdf['Station'].apply(
        lambda station: metadata[metadata['Station number'] == int(station)]['Maximum altitude'].iloc[0])
    gdf['Height difference'] = gdf['Maximum altitude'] - gdf['Minimum altitude']
    gdf['Slope gradient'] = gdf['Height difference'] / gdf['Cross_distance']
    return gdf


def get_characteristics_all_stations(stations_list, input_type='9to9_linear', years_eval=[2010 + i for i in range(10)]):

    new_data_list = []

    # Determine source of station list
    if isinstance(stations_list, str) and os.path.isdir(stations_list):
        # Case: stations_list is a directory path — list subdirectories
        station_nrs = os.listdir(stations_list)
    elif isinstance(stations_list, list):
        # Case: stations_list is already a list of station numbers
        station_nrs = stations_list

    for station_nr in station_nrs:

        df = pd.read_csv(paths.CATCHMENT_BASINS + '/' + str(station_nr) + '/' + str(str(station_nr) + '.csv'))
        name = df.loc[df.index[3]].iloc[2]

        boundary = gpd.read_file \
        (paths.CATCHMENT_BASINS + '/' + str(station_nr) + '/' + str(str(station_nr) + '.shp'))

        df_predictions = pd.read_csv(paths.PREDICTIONS + f"/{input_type}/{station_nr}_{input_type}.csv")
        df_predictions = df_predictions.dropna(subset=['Flow', 'Predicted'])
        df_predictions = df_predictions.loc[:, ~df_predictions.columns.str.contains('^Unnamed')]
        try:
            df_predictions['Date'] = pd.to_datetime(df_predictions['Date'], format='%Y-%m-%d %H:%M:%S').dt.normalize()
        except ValueError:
            df_predictions['Date'] = pd.to_datetime(df_predictions['Date'], format='%Y-%m-%d').dt.normalize()

        df_test = df_predictions[df_predictions['Date'].dt.year.isin(years_eval)]
        NSE = me.R2(df_test['Flow'], df_test['Predicted'])
        HE_mean = ((df_test['Flow'].mean( ) /boundary.geometry.area ) *((10 ^6) ) *86400000).iloc[0]

        new_data_list.append({
            'Station': station_nr,
            'Name': name,
            'NSE': NSE,
            'HE_mean': HE_mean,
            'Geometry': boundary.geometry.iloc[0]
        })

    overview_gdf = gpd.GeoDataFrame(new_data_list)

    """
    metadata = pd.read_csv(paths.DATA + '/Catchments_Database.csv')
    overview_gdf = calculate_slope_gradient(overview_gdf, metadata)
    overview_gdf['90 percentile'] = overview_gdf['Station'].apply(
        lambda station: metadata[metadata['Station number'] == int(station)]['90 percentile'].iloc[0])
    """
    overview_gdf['Latitude'] = overview_gdf['Geometry'].apply(lambda geometry: calculate_latitude(geometry))
    return overview_gdf