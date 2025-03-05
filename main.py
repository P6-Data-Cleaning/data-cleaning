import time
from cleaning import cleaning
from ShipNotMovingFiltre import filter_moving_ships
from removeOutliers import remove_outliers
from Plot import plot
from cargoFilter import cargo_filter
from missingTime import missing_time

def setup_dask():
    # Setup Dask cluster (adjust for your hardware)
    from distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=8, threads_per_worker=16, memory_limit="300GB")
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

META = {
        '# Timestamp': 'datetime64[ns]',
        'Type of mobile': 'object',
        'MMSI': 'int64',
        'Latitude': 'float64',
        'Longitude': 'float64',
        'Navigational status': 'object',
        'ROT': 'float64',
        'SOG': 'float64',
        'COG': 'float64',
        'Ship type': 'object',
        'Draught': 'float64',
        'Destination': 'object',
        'ETA': 'object',
    }


def main():
    start_time = time.time()
    start_time1 = start_time

    setup_dask()

    print(f"Setup execution time: {time.time() - start_time} seconds")
    start_time = time.time()

    result = cleaning('Data/aisdk-2025-02-14.csv', DTYPES)

    print(f"Cleaned execution time: {time.time() - start_time} seconds")
    start_time = time.time()

    result = filter_moving_ships(result)

    print(f"Moving ships execution time: {time.time() - start_time} seconds")
    start_time = time.time()

    result = missing_time(result)

    print(f"Missing time execution time: {time.time() - start_time} seconds")
    start_time = time.time()

    result = remove_outliers(result, META)    

    print(f"Remove outliers execution time: {time.time() - start_time} seconds")
    start_time = time.time()

    result = cargo_filter(result)

    print(f"Cargo filter execution time: {time.time() - start_time} seconds")
    start_time = time.time()

    result = result.compute()

    print(f"Compute execution time: {time.time() - start_time} seconds")
    start_time = time.time()

    plot(result)

    print(f"Plot execution time: {time.time() - start_time} seconds")
    start_time = time.time()

    result.to_csv('Data/cleaned_data.csv', index=False)

    print(f"Write to CSV execution time: {time.time() - start_time} seconds")
    start_time = time.time()

    print(f"Execution time: {time.time() - start_time1} seconds")

if __name__ == '__main__':
    main()