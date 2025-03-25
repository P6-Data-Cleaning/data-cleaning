import pandas as pd
import geopandas as gpd
import os
from shapely.geometry import Point, LineString
import numpy as np
from tqdm import tqdm
import numba
from joblib import Parallel, delayed
import multiprocessing
from geopy.distance import geodesic
# Line simplification method

# Define input and output directories
input_directory = r'D:\Paper2\2025_02'
csv_output_directory = r'C:\Users\HU84VR\Downloads\Paper2\data_csv_2025_02'
csv_onehot_output_directory = r'C:\Users\HU84VR\Downloads\Paper2\data_csv_onehot_2025_02'
geojson_output_directory = r'C:\Users\HU84VR\Downloads\Paper2\data_geojson_2025_02'

# Create directories if they don't exist
os.makedirs(csv_output_directory, exist_ok=True)
os.makedirs(csv_onehot_output_directory, exist_ok=True)
os.makedirs(geojson_output_directory, exist_ok=True)

# Function to calculate trajectory length in kilometers
def calculate_trajectory_length(traj_df):
    """Calculate the total length of a trajectory in kilometers"""
    if len(traj_df) <= 1:
        return 0
    
    total_length = 0
    for i in range(len(traj_df) - 1):
        point1 = (traj_df.iloc[i]['lat'], traj_df.iloc[i]['lng'])
        point2 = (traj_df.iloc[i+1]['lat'], traj_df.iloc[i+1]['lng'])
        total_length += geodesic(point1, point2).kilometers
    
    return total_length

# Function to perform Douglas-Peucker line simplification on a trajectory
def simplify_trajectory(traj_df, tolerance=0.0001):
    """
    Apply Douglas-Peucker line simplification algorithm to a trajectory
    
    Args:
        traj_df: DataFrame containing trajectory points
        tolerance: Simplification tolerance in degrees (higher = more simplification)
    
    Returns:
        DataFrame with simplified trajectory points
    """
    if len(traj_df) <= 2:  # Keep trajectories with 2 or fewer points as is
        return traj_df
    
    # Sort by timestamp
    traj_df = traj_df.sort_values(by='timestamp').reset_index(drop=True)
    
    # Create a LineString from the trajectory points
    line = LineString([(row['lng'], row['lat']) for _, row in traj_df.iterrows()])
    
    # Simplify the line using Douglas-Peucker algorithm
    simplified_line = line.simplify(tolerance, preserve_topology=True)
    
    # Extract the coordinates of the simplified line
    simplified_coords = list(simplified_line.coords)
    
    # Create a mask for points to keep
    keep_indices = []
    for lng, lat in simplified_coords:
        # Find the closest point in the original trajectory
        # This ensures we keep the original attributes
        distances = np.sqrt((traj_df['lng'] - lng)**2 + (traj_df['lat'] - lat)**2)
        closest_idx = distances.argmin()
        keep_indices.append(closest_idx)
    
    # Sort indices to maintain temporal order
    keep_indices = sorted(keep_indices)
    
    # Always include first and last points if they're not already included
    if 0 not in keep_indices:
        keep_indices.insert(0, 0)
    if len(traj_df) - 1 not in keep_indices:
        keep_indices.append(len(traj_df) - 1)
    
    # Return the simplified trajectory
    return traj_df.iloc[keep_indices].reset_index(drop=True)

# Function to process a single trajectory (for parallel processing)
def process_trajectory(traj_df):
    # Apply line simplification
    return simplify_trajectory(traj_df)

# Process each CSV file
csv_files = [f for f in os.listdir(input_directory) if f.endswith('.csv')]
for file_name in csv_files:
    print(f"Processing {file_name}...")
    file_path = os.path.join(input_directory, file_name)
    df = pd.read_csv(file_path)

    # Step 1: Extract Class A data
    df_class_a = df[df['Type of mobile'] == 'Class A']

    # Step 2: Keep required attributes
    important_attributes = ['# Timestamp', 'Type of mobile', 'MMSI', 'Latitude', 'Longitude', 'ROT', 'SOG', 'COG',
                            'Draught', 'Ship type', 'Length', 'Width', 'Navigational status']
    df_class_a = df_class_a[important_attributes]
    
    # Step 3: Convert timestamp to Unix format
    df_class_a['# Timestamp'] = pd.to_datetime(df_class_a['# Timestamp'], dayfirst=True)
    df_class_a['timestamp'] = df_class_a['# Timestamp'].astype('int64') // 10**9  # Convert to Unix timestamp
    df_class_a.drop(columns=['# Timestamp'], inplace=True)
    
    # Step 4: Rename columns
    df_class_a.rename(columns={
        'Type of mobile': 'type_mobile',
        'MMSI': 'mmsi',
        'Latitude': 'lat',
        'Longitude': 'lng',
        'ROT': 'rot',
        'SOG': 'sog',
        'COG': 'cog',
        'Draught': 'draught',
        'Ship type': 'ship_type',
        'Length': 'length',
        'Width': 'width',
        'Navigational status': 'nav_status'
    }, inplace=True)
    
    # Step 5: Ensure lat/lng keep full precision
    df_class_a['lat'] = df_class_a['lat'].astype(float)
    df_class_a['lng'] = df_class_a['lng'].astype(float)

    # Step 6: Convert numerical columns to float and round to exactly 2 decimal places
    decimal_columns = ['rot', 'sog', 'cog', 'draught', 'length', 'width']
    df_class_a[decimal_columns] = df_class_a[decimal_columns].astype(float).round(2)
    
    # Step 7: Remove outliers based on lat/lon range
    lon_min, lon_max = 3.5, 17.3
    lat_min, lat_max = 53, 59
    df_class_a = df_class_a[
        (df_class_a['lng'] >= lon_min) & (df_class_a['lng'] <= lon_max) &
        (df_class_a['lat'] >= lat_min) & (df_class_a['lat'] <= lat_max)
    ]

    # Step 8: Remove rows with missing values, Unknown value, Unknown, Undefined
    df_class_a = df_class_a.dropna()
    df_class_a = df_class_a[~df_class_a['ship_type'].isin(['Unknown', 'Undefined'])]
    df_class_a = df_class_a[~df_class_a['nav_status'].isin(['Unknown', 'Unknown value', 'Undefined'])]
    
    # Step 9: Remove "Pilot", "Undefined", "Fishing" ship types and "Moored" or "At anchor" navigational status
    df_class_a = df_class_a[~df_class_a['ship_type'].isin(["Dredging", "Fishing", "Pilot", "Passenger", 
                                                           "Port tender","SAR", "Spare 1", "Spare 2", 
                                                           "Tanker", "Towing", "Tug", "Law enforcement",
                                                           "Undefined", "HSC", "Other"])]
    df_class_a = df_class_a[~df_class_a['nav_status'].isin(["Moored", "At anchor"])]
    
    # Step 10: Sort data in ascending time order for each MMSI
    df_class_a = df_class_a.sort_values(by=['mmsi', 'timestamp'])

    # Step 11: Trajectory simplification using Douglas-Peucker algorithm
    # Use parallel processing to speed up the simplification
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} CPU cores for parallel processing")
    
    # Group trajectories by MMSI
    trajectory_groups = [group for _, group in df_class_a.groupby('mmsi')]
    
    # Process trajectories in parallel
    simplified_trajectories = Parallel(n_jobs=num_cores)(
        delayed(process_trajectory)(traj_df) for traj_df in tqdm(trajectory_groups, desc="Simplifying trajectories")
    )

    # Combine simplified data
    df_cleaned = pd.concat(simplified_trajectories) if simplified_trajectories else pd.DataFrame()
    
    # Sort the final combined data by mmsi and timestamp
    df_cleaned = df_cleaned.sort_values(by=['mmsi', 'timestamp'])
    
    # Step 11.5: Filter out trajectories with length less than 10
    valid_trajectories = []
    for mmsi, traj_df in df_cleaned.groupby('mmsi'):
        traj_length = calculate_trajectory_length(traj_df)
        if traj_length >= 10:  # Only keep trajectories >= 10km
            valid_trajectories.append(traj_df)
    
    # Combine valid trajectories
    df_cleaned = pd.concat(valid_trajectories) if valid_trajectories else pd.DataFrame()
    print(f"Kept {len(valid_trajectories)} trajectories with length >= 10km")

    # Save cleaned data without one-hot encoding
    csv_output_file = os.path.join(csv_output_directory, file_name.replace('.csv', '_cleaned.csv'))
    df_cleaned.to_csv(csv_output_file, index=False, float_format='%.10f')  # Full precision for lat/lng
    print(f"Cleaned data saved to {csv_output_file}")

    # Step 12: One-hot encoding for categorical attributes
    df_onehot = pd.get_dummies(df_cleaned, columns=['type_mobile', 'ship_type', 'nav_status'], dummy_na=True)

    # Convert only one-hot encoded columns (boolean columns) to 1/0
    one_hot_columns = df_onehot.select_dtypes(include=['bool', 'uint8']).columns
    df_onehot[one_hot_columns] = df_onehot[one_hot_columns].astype(int)

    # Save cleaned data with one-hot encoding
    csv_onehot_output_file = os.path.join(csv_onehot_output_directory, file_name.replace('.csv', '_cleaned_onehot.csv'))
    df_onehot.to_csv(csv_onehot_output_file, index=False, float_format='%.10f')  # Full precision for lat/lng
    print(f"One-hot encoded data saved to {csv_onehot_output_file}")

    # Step 13: Convert to GeoJSON for QGIS visualization
    if not df_cleaned.empty:
        geojson_output_file = os.path.join(geojson_output_directory, file_name.replace('.csv', '_cleaned.geojson'))
        
        # Convert to GeoDataFrame with WGS 84 (EPSG:4326)
        gdf = gpd.GeoDataFrame(df_cleaned, 
                               geometry=gpd.points_from_xy(df_cleaned['lng'], df_cleaned['lat']))

        # Set CRS to WGS84 (EPSG:4326)
        gdf.set_crs(epsg=4326, inplace=True)

        # Save as GeoJSON
        gdf.to_file(geojson_output_file, driver="GeoJSON")

        print(f"GeoJSON file saved at: {geojson_output_file}")

print("All processing complete.")
