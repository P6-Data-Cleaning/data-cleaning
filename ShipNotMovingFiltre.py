from removeOutliers import vectorized_haversine
import dask.dataframe as dd
import pandas as pd

def compute_distance(df):
    df = df.sort_values(by=['MMSI', '# Timestamp'])

    # Compute the total distance traveled using Haversine formula
    df['_prev_lat'] = df['Latitude'].shift(1)
    df['_prev_lon'] = df['Longitude'].shift(1)
    
    df['_distance'] = vectorized_haversine(df['_prev_lat'], df['_prev_lon'], df['Latitude'], df['Longitude'])

    return df['_distance'].sum()

def filter_moving_ships(cleaned_data, min_distance_threshold=5): # Threshold is 2 km

    cleaned_data = cleaned_data[(cleaned_data["SOG"] > 0) & 
                                (cleaned_data["Longitude"] != 0) & 
                                (cleaned_data["Latitude"] != 0)]
    
    cleaned_data = cleaned_data.sort_values(by=["MMSI", "# Timestamp"])

    ship_distances = cleaned_data.groupby("MMSI").apply(compute_distance, meta=("distance", "f8"))

    ship_distances_df = ship_distances.to_frame("distance").reset_index()
    
    # Get the correct column name that contains MMSI
    mmsi_column = ship_distances_df.columns[0]
    
    # Filter ships that moved more than the threshold
    valid_ships_df = ship_distances_df[ship_distances_df["distance"] > min_distance_threshold][[mmsi_column]]
    
    # Rename to ensure it's called "MMSI" for the merge
    valid_ships_df = valid_ships_df.rename(columns={mmsi_column: "MMSI"})

    filtered_data = dd.merge(cleaned_data, valid_ships_df, on="MMSI", how="inner")
    
    return filtered_data