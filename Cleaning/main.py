import time
from cleaning import cleaning
from ShipNotMovingFiltre import filter_moving_ships
from removeOutliers import remove_outliers
from cargoFilter import cargo_filter
from missingTime import missing_time
from trajectoryReducer import trajectory_reducer
from polyIntersect import poly_intersect
import dask.dataframe as dd
import logging
import pandas as pd
import os

def setup_dask():
    import os
    os.environ["DASK_DISTRIBUTED__DIAGNOSTICS__NVML"] = "False"
    from distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=20, threads_per_worker=4, memory_limit="200GB")
    return Client(cluster)

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
        'Heading': 'float64',
    }

def main():
    logging.getLogger("distributed.shuffle._scheduler_plugin").setLevel(logging.ERROR)
    logging.getLogger("distributed.sizeof").setLevel(logging.ERROR)

    start_time = time.time()
    start_time1 = start_time

    client = setup_dask()
    print(f"Setup execution time: {time.time() - start_time} seconds")
    start_time = time.time()
    
    # Make sure the output directory exists
    os.makedirs('outputs', exist_ok=True)
    
    csv_files = []
    for root, _, files in os.walk('Data/mar'):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
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
            
            result = df.compute()
            computed_dfs.append(result)
        
        print(f"Parallel computation time: {time.time() - start_time} seconds")
        start_time = time.time()

        # Merge all dataframes, being explicit about handling indices
        print("Merging dataframes...")
        final_df = pd.concat(computed_dfs, ignore_index=True)
        print(f"Merge time: {time.time() - start_time} seconds")
        start_time = time.time()

        start_rows = len(final_df)
        final_df = trajectory_reducer(final_df)
        print(f"Reduced to {len(final_df)} rows from {start_rows}")
        print(f"Trajectory reduction time: {time.time() - start_time} seconds")
        start_time = time.time()
        
        start_rows = len(final_df)
        final_df = poly_intersect(final_df)
        print(f"Reduced to {len(final_df)} rows from {start_rows}")
        print(f"Poly intersection time: {time.time() - start_time} seconds")
        start_time = time.time()

         # Save to CSV
        print("Saving to CSV...")
        final_df.to_csv('outputs/csv/cleaned_data_mar.csv', index=False)
        
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
    main()