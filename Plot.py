import folium
import random
import pandas as pd
import dask.dataframe as dd

def plot(df):
    # Initialize map
    m = folium.Map(location=[0, 0], zoom_start=5)
    
    # First compute the entire dataframe to avoid column order issues
    full_df = df.compute() # Reset index to avoid issues with groupby
    
    # Then select just the needed columns from the pandas dataframe
    needed_cols = ['MMSI', 'Latitude', 'Longitude']
    pdf = full_df[needed_cols]
    
    # Continue with your aggregation
    agg_computed = pdf.groupby('MMSI').agg({
        'Latitude': ['mean', 'count'],
        'Longitude': 'mean'
    })
    
    # Flatten the MultiIndex columns
    agg_computed.columns = ['mean_lat', 'count', 'mean_lon']
    
    valid_groups = len(agg_computed)
    total_points = agg_computed['count'].sum()
    
    if valid_groups > 0:
        map_center = [agg_computed['mean_lat'].mean(), agg_computed['mean_lon'].mean()]
        m.location = map_center
    
    # Step 2: Process each MMSI separately
    mmsis = agg_computed.index.tolist()
    
    # Process each MMSI using the already computed pandas DataFrame
    for mmsi in mmsis:
        # Filter already computed pandas DataFrame
        group_df = pdf[pdf['MMSI'] == mmsi]
        
        # Plot this group
        coordinates = list(zip(group_df['Longitude'], group_df['Latitude']))
        
        # Skip if no coordinates
        if not coordinates:
            continue
            
        # Color generation
        color = f'#{random.randint(0, 0xFFFFFF):06x}'
        
        folium.PolyLine(
            [(lat, lon) for lon, lat in coordinates], 
            color=color, 
            weight=2.5, 
            opacity=0.8,
            popup=f"MMSI: {mmsi} - {len(coordinates)} points"
        ).add_to(m)
    
    # Save the map
    m.save('ship_trajectories.html')
    print(f"Map saved to 'ship_trajectories.html' with {valid_groups} ships and {total_points} total coordinates")