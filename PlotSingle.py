import time
import pandas as pd
import folium
import glob
import os
from folium.plugins import TimestampedGeoJson

def process_file(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # Remove unwanted spaces
    
    if '# Timestamp' not in df.columns or 'Longitude' not in df.columns or 'Latitude' not in df.columns:
        print(f"Skipping {file_path}: Required columns not found.")
        return None
    
    df['# Timestamp'] = pd.to_datetime(df['# Timestamp'], format="%d/%m/%Y %H:%M:%S", dayfirst=True)
    
    coordinates = list(zip(df['Longitude'], df['Latitude']))
    timestamps = df['# Timestamp'].astype(str).tolist()
    
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [coordinates[i][0], coordinates[i][1]]},
            "properties": {
                "time": timestamps[i],
                "popup": f"Time: {timestamps[i]} ({os.path.basename(file_path)}), MMSI: {df['MMSI'].iloc[0]}",
                "icon": "marker",
                "iconstyle": {
                    "iconUrl": "https://cdn-icons-png.flaticon.com/512/190/190004.png",
                    "iconSize": [30, 30],
                    "iconAnchor": [15, 15]
                }
            }
        } for i in range(0, len(coordinates))
    ]
    
    return {
        "coordinates": coordinates,
        "features": features,
        "mean_location": [df['Latitude'].mean(), df['Longitude'].mean()],
        "mmsi": df['MMSI'].iloc[0]
    }

def main():
    start_time = time.time()
    
    map_center = [0, 0]
    all_features = []
    
    m = folium.Map(location=[0, 0], zoom_start=5)
    valid_files = 0

    result = process_file('Ships_data/244855000.csv')
    if result:
        folium.PolyLine([(lat, lon) for lon, lat in result['coordinates']],
                        color="blue", weight=2.5, opacity=0.7, popup=f"MMSI: {result['mmsi']}").add_to(m)
        all_features.extend(result['features'])
        map_center[0] += result['mean_location'][0]
        map_center[1] += result['mean_location'][1]
    
    
    TimestampedGeoJson(
        {"type": "FeatureCollection", "features": all_features},
        period="PT2M",
        duration="PT10M",
        add_last_point=True,
        auto_play=True,
        loop=False,
    ).add_to(m)

    # Add ship locations to map
    for feature in all_features:
        folium.Marker(
            location=[feature['geometry']['coordinates'][1], feature['geometry']['coordinates'][0]],
            popup=feature['properties']['popup'],
        ).add_to(m)
    
    # Adjust map center
    if valid_files > 0:
        map_center[0] /= valid_files
        map_center[1] /= valid_files
        m.location = map_center

    m.save('ship_single_sim.html')
    print(f"Execution time: {time.time() - start_time} seconds")

if __name__ == '__main__':
    main()
