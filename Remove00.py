import os
import dask.dataframe as dd

def filter_moving_ships():
    # Setup Dask cluster (adjust for your hardware)
    from distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=8, threads_per_worker=16, memory_limit="300GB")
    Client(cluster)
    
    # File paths
    script_dir = os.getcwd()
    input_file = os.path.join(script_dir, 'moving_ships.csv')
    output_file = os.path.join(script_dir, 'moving_ships_filtered.csv')

    # Define data types for efficiency
    dtypes = {
        '# Timestamp': 'object',
        'MMSI': 'Int64',
        'SOG': 'float64',
        'ETA': 'object',
        'Longitude': 'float64',
        'Latitude': 'float64'
    }
    
    # Load dataset
    df = dd.read_csv(input_file, dtype=dtypes)

    # Remove rows where Longitude and Latitude are 0
    df = df[(df["Longitude"] != 0) & (df["Latitude"] != 0)]

    # Save results
    df.compute().to_csv(output_file, index=False)

    print(f"Remove 0 , 0 form dataset : {output_file}")

if __name__ == '__main__':
    filter_moving_ships()