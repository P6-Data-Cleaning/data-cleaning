import time
from cleaning import cleaning
from ShipNotMovingFiltre import filter_moving_ships
from removeOutliers import remove_outliers
from cargoFilter import cargo_filter
from missingTime import missing_time
from trajectoryReducer import trajectory_reducer
from clusterDetection import mainDetectBehavior
from polyIntersect import poly_intersect
import dask.dataframe as dd
import logging
import pandas as pd
import os

def setup_dask():
    from distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=32, threads_per_worker=4, memory_limit="400GB")
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

    # Load data
    print("Loading data cleaned_data_dec_without_cluster.csv...")
    df = dd.read_csv('outputs/csv/cleaned_data_dec_without_cluster.csv', dtype=DTYPES, parse_dates=['# Timestamp'], assume_missing=True)
    
    final_df = mainDetectBehavior(df, client=client)
    print(f"Unusual behavior detection time: {time.time() - start_time} seconds")

    start_time = time.time()
    final_df = final_df.compute()
    print(f"Compute time: {time.time() - start_time} seconds")
    start_time = time.time()

    # Save to CSV
    print("Saving to CSV...")
    final_df.to_csv('outputs/csv/cleaned_data_dec_done2.csv', index=False)
    
    print(f"Write to CSV execution time: {time.time() - start_time} seconds")
    print(f"Total execution time: {time.time() - start_time1} seconds")
        
    

if __name__ == '__main__':
    main()