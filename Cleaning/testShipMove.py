import dask.dataframe as dd
import time
from cleaning import cleaning
from testPlot import plot
from removeOutliers import vectorized_haversine

def setup_dask():
    # Setup Dask cluster (adjust for your hardware)
    from distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=8, threads_per_worker=16, memory_limit="300GB")
    Client(cluster)

def compute_distance(df):
    #Compute total distance traveled by a ship in a day.
    df = df.sort_values(by=['MMSI', '# Timestamp'])

    # Compute the total distance traveled using Haversine formula
    df['_prev_lat'] = df['Latitude'].shift(1)
    df['_prev_lon'] = df['Longitude'].shift(1)
    
    df['_distance'] = vectorized_haversine(df['_prev_lat'], df['_prev_lon'], df['Latitude'], df['Longitude'])

    return df['_distance'].sum()  # Total distance for this ship

def filter_moving_ships(cleaned_data, min_distance_threshold=0.5):

    # Filters out ships that moved less than `min_distance_threshold` km in a day.
    cleaned_data = cleaned_data[(cleaned_data["SOG"] > 0) & 
                                (cleaned_data["Longitude"] != 0) & 
                                (cleaned_data["Latitude"] != 0)]
    
    print(f"SOG filtre execution time: {time.time() - start_time} seconds")
    
    # Sort data by MMSI and Timestamp
    cleaned_data = cleaned_data.sort_values(by=["MMSI", "# Timestamp"])

    ship_distances = cleaned_data.groupby("MMSI").apply(compute_distance, meta=("distance", "f8"))
    
    valid_ships = ship_distances[ship_distances > min_distance_threshold].compute()
    
    
    filtered_data = cleaned_data[cleaned_data["MMSI"].isin(valid_ships.index)]
    
    return filtered_data

def categorize_ships_by_distance(cleaned_data):

    ship_distances = cleaned_data.groupby("MMSI").apply(compute_distance, meta=("distance", "f8")).compute()

    distance_ranges = [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 5), (5, 10), (10, 15), (15, 25), (25, 50), (50, 100), (100, float('inf'))]
    range_labels = ["0 - 0.5", "0.5 - 1", "1 - 1.5", "1.5 - 2", "2 - 5", "5 - 10", "10 - 15", "15 - 25", "25 - 50", "50 - 100", "> 100"]

    # Categorize ships into distance ranges
    categorized_counts = {label: 0 for label in range_labels}
    for distance in ship_distances:
        for (low, high), label in zip(distance_ranges, range_labels):
            if low <= distance < high:
                categorized_counts[label] += 1
                break

    for label, count in categorized_counts.items():
        print(f"Number of ships that traveled {label} km: {count}")

if __name__ == '__main__':
    
    DTYPES = {
        '# Timestamp': 'object',
        'MMSI': 'int64',
        'Latitude': 'float64',
        'Longitude': 'float64',
        'COG': 'float64',
        'SOG': 'float64',
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
        'D': 'float64',
        'Type of mobile': 'object'
    }

    
    # Execute the code directly
    start_time = time.time()
    setup_dask()
    print(f"Setup execution time: {time.time() - start_time} seconds")

    start_time = time.time()
    cleaned = cleaning('./Data/aisdk-2025-02-14.csv', DTYPES)
    print(f"Cleaned execution time: {time.time() - start_time} seconds")

    start_time = time.time()
    filtered_data = filter_moving_ships(cleaned)
    print(f"Moving ships execution time: {time.time() - start_time} seconds")
    
    start_time = time.time()
    categorize_ships_by_distance(cleaned)
    print(f"Categorize ships execution time: {time.time() - start_time} seconds")
    
    start_time = time.time()
    plot(filtered_data)
    print(f"Plot execution time: {time.time() - start_time} seconds")
    