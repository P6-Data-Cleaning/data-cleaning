import folium
import random
import pandas as pd
import sys

def plot(file_path):
    # Load the data
    df = pd.read_csv(file_path)


    # Initialize map
    m = folium.Map(location=[0, 0], zoom_start=5)
    
    # Then select just the needed columns from the pandas dataframe
    needed_cols = ['MMSI', 'Latitude', 'Longitude']
    pdf = df[needed_cols]
    
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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Plot.py <filename>")
        sys.exit(1)

    fileName = sys.argv[1]
    
    plot(fileName)
    print(f"Saved to ship_trajectories.html")
