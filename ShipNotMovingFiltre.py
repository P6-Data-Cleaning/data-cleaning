from removeOutliers import vectorized_haversine
import dask.dataframe as dd
import pandas as pd
import time

def compute_distance(df):
    df = df.sort_values(by=['MMSI', '# Timestamp'])

    df['_prev_lat'] = df['Latitude'].shift(1)
    df['_prev_lon'] = df['Longitude'].shift(1)

    df['_distance'] = vectorized_haversine(df['_prev_lat'], df['_prev_lon'], df['Latitude'], df['Longitude'])

    # Return DataFrame instead of Series for better Dask compatibility
    return pd.DataFrame({
        'MMSI': [df['MMSI'].iloc[0]],  
        'total_distance': [df['_distance'].sum()],
        'is_passenger': [bool((df['Ship type'] == 'Passenger').any())]
    })



def measure_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} execution time: {time.time() - start_time:.2f} seconds")
        return result
    return wrapper

@measure_performance
def filter_moving_ships(cleaned_data, default_threshold=100, passenger_threshold=5):
    # Define metadata for DataFrame result
    meta = pd.DataFrame({
        'MMSI': [0],
        'total_distance': [0.0],
        'is_passenger': [False]
    })
    
    cleaned_data = cleaned_data[(cleaned_data["SOG"] > 0) & 
                                (cleaned_data["Longitude"] != 0) & 
                                (cleaned_data["Latitude"] != 0)]
    
    cleaned_data = cleaned_data.sort_values(by=["MMSI", "# Timestamp"])

    # Group by MMSI and compute distance + check if it's a passenger ship
    ship_stats = cleaned_data.groupby("MMSI").apply(
        compute_distance,
        meta=meta
    ).reset_index(drop=True)  # Drop=True to avoid multi-level index issues
    
    # Create threshold column with a simpler approach
    def add_threshold(df):
        df = df.copy()
        df['threshold'] = df['is_passenger'].map(lambda x: passenger_threshold if x else default_threshold)
        return df
    
    ship_stats = ship_stats.map_partitions(add_threshold)
 
    # Filter to valid ships
    valid_ships_df = ship_stats[ship_stats["total_distance"] > ship_stats["threshold"]][["MMSI"]]

    # Merge back to keep only valid ships
    filtered_data = dd.merge(cleaned_data, valid_ships_df, on="MMSI", how="inner")
    

    return filtered_data