import dask.dataframe as dd
import time
import os
from cleaning import cleaning
from removeOutliers import vectorized_haversine

def setup_dask():
    # Setup Dask cluster (adjust for your hardware)
    from distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=8, threads_per_worker=12, memory_limit="300GB")
    Client(cluster)

def compute_distance(df):
    #Compute total distance traveled by a ship in a day.
    df = df.sort_values(by="# Timestamp")

    # Compute the total distance traveled using Haversine formula
    df['_prev_lat'] = df['Latitude'].shift(1)
    df['_prev_lon'] = df['Longitude'].shift(1)
    
    df['_distance'] = vectorized_haversine(df['_prev_lat'], df['_prev_lon'], df['Latitude'], df['Longitude'])

    return df['_distance'].sum()  # Total distance for this ship

def filter_moving_ships(cleaned_data, min_distance_threshold=0.5):
    
    # Initial count of unique MMSI values
    initial_mmsi_count = cleaned_data["MMSI"].nunique()
    print(f"Initial number of ships: {initial_mmsi_count}")

    # Filters out ships that moved less than `min_distance_threshold` km in a day.
    cleaned_data = cleaned_data[(cleaned_data["SOG"] > 0) & 
                                (cleaned_data["Longitude"] != 0) & 
                                (cleaned_data["Latitude"] != 0)]
    
    # Count of unique MMSI values after filtering by SOG, Longitude, and Latitude
    after_sog_lon_lat_mmsi_count = cleaned_data["MMSI"].nunique()
    print(f"Number of ships after filtering by SOG, Longitude, and Latitude: {after_sog_lon_lat_mmsi_count}")

    ship_distances = cleaned_data.groupby("MMSI").apply(compute_distance, meta=("distance", "f8"))
    
    valid_ships = ship_distances[ship_distances > min_distance_threshold].compute()

    # Count of unique MMSI values after filtering by distance
    after_distance_mmsi_count = valid_ships.index.nunique()
    print(f"Number of ships after filtering by distance: {after_distance_mmsi_count}")

    filtered_data = cleaned_data[cleaned_data["MMSI"].isin(valid_ships.index)]

    # Create the directory if it doesn't exist
    output_dir = 'test_move_ship'
    os.makedirs(output_dir, exist_ok=True)

    # Save the filtered data to a CSV file
    output_file = os.path.join(output_dir, 'filtered_ships.csv')
    filtered_data.to_csv(output_file, index=False, single_file=True)

    return filtered_data


if __name__ == '__main__':
    # Execute the code directly
    start_time = time.time()
    setup_dask()
    print(f"Setup execution time: {time.time() - start_time} seconds")

    start_time = time.time()
    cleaned = cleaning('aisdk-2025-02-14.csv')
    print(f"Cleaned execution time: {time.time() - start_time} seconds")

    start_time = time.time()
    filter_moving_ships(cleaned)
    print(f"Moving ships execution time: {time.time() - start_time} seconds")