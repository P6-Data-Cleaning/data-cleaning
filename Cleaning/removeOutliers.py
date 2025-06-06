import dask.dataframe as dd
import numpy as np
import pandas as pd
import time
def vectorized_haversine(lat1, lon1, lat2, lon2):
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c  # Earth radius in km
    return km

def computing(df, meta, max_iter=5):
    # Ensure the DataFrame is sorted by timestamp so that shifting works correctly
    df = df.sort_values('# Timestamp')
    
    for _ in range(max_iter):
        # Create temporary columns with a prefix to avoid clashing with existing columns
        df['_prev_lat'] = df['Latitude'].shift(1)
        df['_prev_lon'] = df['Longitude'].shift(1)
    
        # Calculate distance using the vectorized haversine function
        df['_dis'] = vectorized_haversine(
            df['_prev_lat'].values, df['_prev_lon'].values,
            df['Latitude'].values, df['Longitude'].values)
    
        # Compute time difference in seconds
        df['_time'] = (df['# Timestamp'] - df['# Timestamp'].shift(1)).dt.total_seconds()
    
        # Compute speed (km/s) and then compute knob
        df['_kms'] = df['_dis'] / df['_time']
        df['_knob'] = df['_kms'] * 1943.84

        # Determine rows to keep (rows with NaN _knob or _knob <= 100)
        mask = (df['_knob'].isna()) | (df['_knob'] <= 100)
        
        # If no rows are removed, break out of the loop.
        if mask.all():
            # Drop temporary columns and break
            df = df.drop(columns=['_prev_lat', '_prev_lon', '_dis', '_time', '_kms', '_knob'])
            break
        
        # Otherwise, filter the DataFrame and repeat.
        df = df[mask].copy()
    
    df = df.astype(meta)  # Ensure data types match
    df = df.reindex(columns=list(meta.keys()))  # Explicitly set column order   
    return df

def process_by_mmsi(df_group, meta):
    """Process a single MMSI group"""

    df_group = computing(df_group, meta)  # Process the data
    
    # Explicitly set correct column order before returning
    return df_group.reindex(columns=list(meta.keys()))

def measure_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} execution time: {time.time() - start_time:.2f} seconds")
        return result
    return wrapper

@measure_performance
def remove_outliers(df, meta):
    print(f"remove outliers start: {len(df)} rows")
    """
    Remove outliers from the given Dask DataFrame.
    Processes each MMSI separately to prevent comparing points from different vessels.
    """
    
    # Process each MMSI group separately using apply
    # Note: This approach splits the data by MMSI first, then applies the outlier removal
    result = df.groupby('MMSI').apply(
        lambda group: process_by_mmsi(group, meta), 
        meta=meta
    )
    print(f"remove outliers end: {len(result)} rows")
    return result