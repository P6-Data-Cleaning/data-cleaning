import pandas as pd
import dask.dataframe as dd
import psycopg2
from sqlalchemy import create_engine
import sys
from datetime import datetime
from main import META

def plot(file_path):
    from distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=32, threads_per_worker=4, memory_limit="300GB")
    client = Client(cluster)

    # Create a copy of META without the Timestamp column
    meta_without_timestamp = META.copy()
    if '# Timestamp' in meta_without_timestamp:
        del meta_without_timestamp['# Timestamp']

    # Load the data using Dask
    df = dd.read_csv(file_path, 
                     dtype=meta_without_timestamp,
                     parse_dates=['# Timestamp'])
    
    # Select needed columns and convert to pandas
    needed_cols = ['MMSI', '# Timestamp', 'Latitude', 'Longitude', 'COG', 'SOG', 'ROT', 
                  'Navigational status', 'Ship type', 'Draught', 'Destination', 'ETA']
    pdf = df[needed_cols].compute()
    
    # Connect to database
    conn_string = "postgresql://postgres:dbs123@localhost:5432/ais_data"
    engine = create_engine(conn_string)
    
    # Insert data in chunks
    chunk_size = 100000
    for i in range(0, len(pdf), chunk_size):
        chunk = pdf.iloc[i:i+chunk_size].copy()
        
        # Rename columns to match database schema
        chunk.columns = ['mmsi', 'timestamp', 'latitude', 'longitude', 'cog', 'sog', 'rot', 
                         'navigational_status', 'ship_type', 'draught', 'destination', 'eta']
        
        # Insert into database
        chunk.to_sql('ship_trajectories', engine, if_exists='append', index=False)
        print(f"Inserted chunk {i//chunk_size + 1}/{(len(pdf)-1)//chunk_size + 1}")
    
    # Update ship metadata
    ship_metadata = pdf.groupby('MMSI').agg({
        'Ship type': lambda x: x.mode()[0] if not x.mode().empty else None,
        '# Timestamp': 'max'
    }).reset_index()
    
    ship_metadata.columns = ['mmsi', 'ship_type', 'last_seen']
    ship_metadata.to_sql('ships', engine, if_exists='replace', index=False)
    
    print(f"Data successfully loaded into database")