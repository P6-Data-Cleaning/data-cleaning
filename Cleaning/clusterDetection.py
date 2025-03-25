import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import sys
import os
import dask.dataframe as dd
from distributed import Client, LocalCluster

META = {
    'MMSI': 'int64',
    'Latitude': 'float64',
    'Longitude': 'float64',
    'SOG': 'float64',
    'COG': 'float64',
    'Heading': 'float64',
    'ROT': 'float64',
    'Navigational status': 'object',
    'IMO': 'object',
    'Callsign': 'object',
    'Name': 'object',
    'Ship type': 'object',
    'Cargo type': 'object',
    'Width': 'float64',
    'Length': 'float64',
    'Type of position fixing device': 'object',
    'Draught': 'float64',
    'Destination': 'object',
    'ETA': 'object',
    'Data source type': 'object',
    'A': 'float64',
    'B': 'float64',
    'C': 'float64',
    'D': 'float64'
}

def detect_and_remove_anomalies(file_path, density_threshold, radius_km, output_path=None):
    """
    Detect ships with dense clustering behavior (>10 points in 0.5km radius) and remove them.
    
    Parameters:
    -----------
    file_path : str
        Path to the AIS data CSV file
    output_path : str # Parameters for cluster detection
    density_threshold = 10
    radius_km = 1.0
        Path to save the cleaned data (if None, will generate from input name)
    density_threshold : int
        Minimum number of points in a cluster to consider it anomalous
    radius_km : float
        Radius in kilometers to consider for clustering
        
    Returns:
    --------
    DataFrame with anomalous ships removed
    """
    print(f"Setting up distributed computing environment...")
    cluster = LocalCluster(n_workers=32, threads_per_worker=4, memory_limit="300GB")
    client = Client(cluster)
    
    # Create output path if not provided
    if output_path is None:
        base_name = os.path.basename(file_path)
        name_parts = os.path.splitext(base_name)
        output_path = f"{name_parts[0]}_clusterDetection_cleaned{name_parts[1]}"
    
    # Create a copy of META without the Timestamp column
    meta_without_timestamp = META.copy()
    if '# Timestamp' in meta_without_timestamp:
        del meta_without_timestamp['# Timestamp']
    
    print(f"Loading data from {file_path}...")
    # Load the data using Dask
    df = dd.read_csv(file_path, 
                     dtype=meta_without_timestamp,
                     parse_dates=['# Timestamp'])
    
    # Select needed columns and convert to pandas
    needed_cols = ['MMSI', '# Timestamp', 'Latitude', 'Longitude']
    pdf = df[needed_cols].compute()
    
    print(f"Original dataset: {len(pdf)} records, {pdf['MMSI'].nunique()} unique vessels")
    
    # Function to convert km to radians for DBSCAN with haversine metric
    kms_per_radian = 6371.0
    epsilon = radius_km / kms_per_radian
    
    print(f"DBSCAN parameters: eps={radius_km} km, min_samples=2 (for forming clusters)")
    print(f"Will identify ships with >= {density_threshold} points in {radius_km} km radius")
    
    # Process each ship separately
    unique_mmsis = pdf['MMSI'].unique()
    print(f"Analyzing {len(unique_mmsis)} vessels for anomalous clusters...")
    
    anomalous_mmsis = set()  # Using a set for faster lookups
    
    # Process each ship
    for i, mmsi in enumerate(unique_mmsis):
        if i > 0 and i % 100 == 0:
            print(f"Processed {i}/{len(unique_mmsis)} vessels...")
            
        ship_data = pdf[pdf['MMSI'] == mmsi].copy()
        
        if len(ship_data) < density_threshold:
            # Skip ships with too few points to be anomalous
            continue
        
        # Detect clusters using DBSCAN
        coords = ship_data[['Latitude', 'Longitude']].values
        
        # Use min_samples=2 to form clusters more easily, then check sizes manually
        db = DBSCAN(eps=epsilon, min_samples=2, algorithm='ball_tree', metric='haversine')
        clusters = db.fit_predict(np.radians(coords))
        
        # Get counts of points in each cluster (excluding noise which is -1)
        cluster_sizes = {}
        for label in np.unique(clusters):
            if label >= 0:  # Skip noise points
                cluster_sizes[label] = np.sum(clusters == label)
        
        # Skip if no clusters formed
        if not cluster_sizes:
            continue
            
        # Check if any cluster has more points than the threshold
        has_dense_cluster = any(size >= density_threshold for size in cluster_sizes.values())
        
        if has_dense_cluster:
            # Debug: Print info about large clusters (first 5 ships only)
            if len(anomalous_mmsis) < 5:
                for label, size in cluster_sizes.items():
                    if size >= density_threshold:
                        print(f"MMSI {mmsi}: Found anomalous cluster with {size} points")
            
            anomalous_mmsis.add(mmsi)
    
    # Report found anomalies
    if anomalous_mmsis:
        print(f"\nDetected {len(anomalous_mmsis)} ships with anomalous behavior")
        
        # Remove anomalous ships from the dataset
        print(f"Removing anomalous ships from dataset...")
        cleaned_df = df[~df['MMSI'].isin(anomalous_mmsis)]
        
        # Save cleaned data
        print(f"Saving cleaned data to {output_path}...")
        cleaned_df.to_csv(output_path, single_file=True, index=False)
        
        # Report statistics
        percent_removed = (len(anomalous_mmsis) / len(unique_mmsis)) * 100
        print(f"\nCleaning summary:")
        print(f"- Total vessels analyzed: {len(unique_mmsis)}")
        print(f"- Anomalous vessels removed: {len(anomalous_mmsis)} ({percent_removed:.2f}%)")
        print(f"- Cleaned data saved to: {output_path}")
        
        return cleaned_df
    else:
        print("No anomalous ships detected with current parameters")
        print(f"Saving original data to {output_path}...")
        df.to_csv(output_path, single_file=True, index=False)
        return df

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clusterDetection.py <input_file> [output_file] [density_threshold] [radius_km]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    density_threshold = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    radius_km = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    
    print(f"Running with parameters:")
    print(f"- Density threshold: {density_threshold} points")
    print(f"- Cluster radius: {radius_km} km")
    
    detect_and_remove_anomalies(
        input_file, 
        output_file, 
        density_threshold=density_threshold,
        radius_km=radius_km
    )