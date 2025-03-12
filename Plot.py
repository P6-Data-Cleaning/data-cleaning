import folium
import random
import pandas as pd
import dask.dataframe as dd
import sys
from datetime import datetime
from main import META

def plot(file_path):
    from distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=32, threads_per_worker=4, memory_limit="300GB")
    client = Client(cluster)

    # Create a copy of META without the Timestamp column
    meta_without_timestamp = META.copy()
    if '# Timestamp' in meta_without_timestamp:
        del meta_without_timestamp['# Timestamp']

    # Load the data using Dask with modified dtype and parse_dates
    df = dd.read_csv(file_path, 
                     dtype=meta_without_timestamp,
                     parse_dates=['# Timestamp'])
    
    # Select needed columns and convert to pandas
    needed_cols = ['MMSI', '# Timestamp', 'Latitude', 'Longitude']
    pdf = df[needed_cols].compute()
    
    # Convert timestamp to datetime and extract date part only
    pdf['# Timestamp'] = pd.to_datetime(pdf['# Timestamp']).dt.floor('D')
    
    # Sort values to optimize groupby operation
    pdf = pdf.sort_values(['MMSI', '# Timestamp'])
    
    # Initialize map
    m = folium.Map(location=[0, 0], zoom_start=5)
    
    # Process in chunks by MMSI
    colors = {}
    unique_mmsis = pdf['MMSI'].unique()
    
    # Calculate map center using mean of all coordinates
    map_center = [pdf['Latitude'].mean(), pdf['Longitude'].mean()]
    m.location = map_center
    
    for mmsi in unique_mmsis:
        # Get all data for this MMSI
        vessel_data = pdf[pdf['MMSI'] == mmsi]
        
        # Group by date
        for date, group in vessel_data.groupby('# Timestamp'):
            coordinates = list(zip(group['Longitude'], group['Latitude']))
            
            if not coordinates:
                continue
                
            # Generate consistent color for MMSI
            if mmsi not in colors:
                colors[mmsi] = f'#{random.randint(0, 0xFFFFFF):06x}'
            
            folium.PolyLine(
                [(lat, lon) for lon, lat in coordinates],
                color=colors[mmsi],
                weight=2.5,
                opacity=0.8,
                popup=f"MMSI: {mmsi} - Date: {date.strftime('%Y-%m-%d')} - Points: {len(coordinates)}"
            ).add_to(m)
    
    # Save the map
    m.save('outputs/html/ship_trajectories.html')
    print(f"Map saved with {len(unique_mmsis)} unique ships")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Plot.py <filename>")
        sys.exit(1)

    fileName = sys.argv[1]
    plot(fileName)
