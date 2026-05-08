import pandas as pd
import geopandas as gpd

df = pd.read_csv('samples/insitu_mato_grosso_state_(brazil)_land_use_and_land_cover_samples_2k.csv')

df_filtered = df[['longitude', 'latitude', 'label', 'start_date', 'end_date']]
df_filtered['start_date'] = pd.to_datetime(df['start_date'], format='%d-%m-%Y').dt.strftime('%Y-%m-%d')
df_filtered['end_date'] = pd.to_datetime(df['end_date'], format='%d-%m-%Y').dt.strftime('%Y-%m-%d')

df_filtered.to_csv("./samples/samples_mt.csv", index = False)

gdf = gpd.GeoDataFrame(
    df_filtered, 
    geometry=gpd.points_from_xy(df_filtered.longitude, df_filtered.latitude),
    crs="EPSG:4326"
)[['geometry', 'label', 'start_date', 'end_date']]

gdf.to_file("./samples/sample_mt/samples_mt.shp", driver='ESRI Shapefile')
