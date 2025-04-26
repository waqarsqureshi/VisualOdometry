import folium
import pandas as pd
import numpy as np

# Load your trajectory CSV
df = pd.read_csv('./output/trajectory_comparison.csv')

# Load GPS origin from your GPS file
gps_df = pd.read_csv('./data/GPS_20250216_161649_data.csv')
origin_lat = gps_df.iloc[0]['GPS (Lat.) [deg]']
origin_lon = gps_df.iloc[0]['GPS (Long.) [deg]']

# Helper: Convert local meters back to lat/lon
def meters_to_latlon(dx, dz, origin_lat, origin_lon):
    dlat = dx / 111320
    dlon = dz / (40075000 * np.cos(np.radians(origin_lat)) / 360)
    return origin_lat + dlat, origin_lon + dlon

# Create a map centered at start
mymap = folium.Map(location=[origin_lat, origin_lon], zoom_start=18)

# Build lists of VO and GPS points
vo_points = []
gps_points = []

for i in range(len(df)):
    if not np.isnan(df['vo_x'][i]) and not np.isnan(df['vo_z'][i]):
        lat_vo, lon_vo = meters_to_latlon(df['vo_z'][i], df['vo_x'][i], origin_lat, origin_lon)
        vo_points.append((lat_vo, lon_vo))
    if not np.isnan(df['gps_x'][i]) and not np.isnan(df['gps_z'][i]):
        lat_gps, lon_gps = meters_to_latlon(df['gps_z'][i], df['gps_x'][i], origin_lat, origin_lon)
        gps_points.append((lat_gps, lon_gps))


# Draw VO path (green line)
folium.PolyLine(locations=vo_points, color='green', weight=3, opacity=0.7, tooltip='VO Trajectory').add_to(mymap)

# Draw GPS path (blue dashed line)
folium.PolyLine(locations=gps_points, color='blue', weight=2, opacity=0.5, dash_array='5,10', tooltip='GPS Path').add_to(mymap)

# Mark start and end points
folium.Marker(location=vo_points[0], popup='VO Start', icon=folium.Icon(color='green')).add_to(mymap)
folium.Marker(location=vo_points[-1], popup='VO End', icon=folium.Icon(color='red')).add_to(mymap)

folium.Marker(location=gps_points[0], popup='GPS Start', icon=folium.Icon(color='blue')).add_to(mymap)
folium.Marker(location=gps_points[-1], popup='GPS End', icon=folium.Icon(color='cadetblue')).add_to(mymap)

# Save Map
mymap.save('./output/trajectory_map.html')
print("Google Maps visualization saved as './output/trajectory_map.html'. Open it in a browser!")

