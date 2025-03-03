import time
from cleaning import cleaning
from ShipNotMovingFiltre import filter_moving_ships as moving_ships
from removeOutliers import remove_outliers
from Plot import plot


def setup_dask():
    # Setup Dask cluster (adjust for your hardware)
    from distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=8, threads_per_worker=15, memory_limit="300GB")
    Client(cluster)

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


def main():
    start_time = time.time()
    start_time1 = start_time

    setup_dask()

    print(f"Setup execution time: {time.time() - start_time} seconds")
    start_time = time.time()

    cleaned = cleaning('Data/aisdk-2025-02-14.csv', DTYPES)

    print(f"Cleaned execution time: {time.time() - start_time} seconds")
    start_time = time.time()

    movingShips = moving_ships(cleaned)

    print(f"Moving ships execution time: {time.time() - start_time} seconds")
    start_time = time.time()

    result = remove_outliers(movingShips)    

    print(f"Remove outliers execution time: {time.time() - start_time} seconds")
    start_time = time.time()

    plot(result)

    print(f"Plot execution time: {time.time() - start_time} seconds")
    start_time = time.time()

    # Compute the final DataFrame and write it to CSV
    result = result.compute()
    result.to_csv('Data/cleaned_data.csv', index=False)

    print(f"Compute and write to CSV execution time: {time.time() - start_time} seconds")
    start_time = time.time()

    print(f"Execution time: {time.time() - start_time1} seconds")

if __name__ == '__main__':
    main()