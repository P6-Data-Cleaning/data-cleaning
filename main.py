import time
from cleaning import cleaning
from ShipNotMovingFiltre import filter_moving_ships
from removeOutliers import remove_outliers
from Plot import plot
from cargoFilter import cargo_filter
from missingTime import missing_time
import dask.dataframe as dd
from dask.distributed import performance_report

def setup_dask():
    # Setup Dask cluster (adjust for your hardware)
    from distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=32, threads_per_worker=4, memory_limit="24GB", dashboard_address=":8787")
    client = Client(cluster)
    print(f"Dashboard available at: {client.dashboard_link}")
    return client

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

    result = dd.read_csv('./Data/aisdk-2025-02-14.csv', dtype=DTYPES)

    result = cleaning(result)

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


def newMain():
    start_time = time.time()
    start_time1 = start_time

    client = setup_dask()
    print(f"Setup execution time: {time.time() - start_time} seconds")
    start_time = time.time()

    # Get list of CSV files
    import os
    import glob
    
    # Make sure the output directory exists
    os.makedirs('outputs', exist_ok=True)
    
    csv_files = glob.glob('downloads/*.csv')
    print(f"Found {len(csv_files)} CSV files to process")

    # Create a list to store all dataframes
    dataframes = []

    # Process each file in parallel
    for file_path in csv_files:
        print(f"Processing {file_path}")
        # Read the file - ensure index is properly handled
        df = dd.read_csv(file_path, dtype=DTYPES)
        # Reset index to avoid any index-related issues
        df = df.reset_index(drop=True)
        
        # Apply processing pipeline (return Dask dataframes without computing)
        try:
            df = cleaning(df)
            df = filter_moving_ships(df)
            df = missing_time(df)
            df = remove_outliers(df, META)
            df = cargo_filter(df)
            
            # Add to list
            dataframes.append(df)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    if not dataframes:
        print("No dataframes to process after filtering!")
        client.close()
        return
    
    print(f"Pipeline setup time: {time.time() - start_time} seconds")
    start_time = time.time()

    # Compute all dataframes in parallel
    print("Computing all dataframes in parallel...")
    try:
        # Compute each dataframe individually to better isolate errors
        computed_dfs = []
        for i, df in enumerate(dataframes):
            print(f"Computing dataframe {i+1}/{len(dataframes)}")
            # Reset index again to ensure clean computation
            df = df.reset_index(drop=True)
            
            with performance_report(filename=f"dask-report-{i}.html"):
                result = df.compute()
                computed_dfs.append(result)
        
        print(f"Parallel computation time: {time.time() - start_time} seconds")
        start_time = time.time()

        # Merge all dataframes, being explicit about handling indices
        import pandas as pd
        print("Merging dataframes...")
        final_df = pd.concat(computed_dfs, ignore_index=True)
        
        print(f"Merge time: {time.time() - start_time} seconds")
        start_time = time.time()
        
        # Save to CSV
        print("Saving to CSV...")
        final_df.to_csv('outputs/cleaned_data.csv', index=False)
        
        print(f"Write to CSV execution time: {time.time() - start_time} seconds")
        print(f"Total execution time: {time.time() - start_time1} seconds")
        
    except Exception as e:
        print(f"Error during computation: {str(e)}")
        import traceback
        traceback.print_exc()
        client.close()
        raise
    finally:
        # Always close the client
        client.close()


if __name__ == '__main__':
    newMain()