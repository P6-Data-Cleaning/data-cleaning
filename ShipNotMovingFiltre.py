import os
import dask.dataframe as dd

def filter_moving_ships():
    # Setup Dask cluster (adjust for your hardware)
    from distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=7, threads_per_worker=16, memory_limit="300GB")
    Client(cluster)
    
    # File paths
    script_dir = os.getcwd()
    input_file = os.path.join(script_dir, 'cleaned.csv')
    output_file = os.path.join(script_dir, 'moving_ships.csv')
    removed_ships_file = os.path.join(script_dir, 'removed_ships.csv')

    # Define data types for efficiency
    dtype = {
        '# Timestamp': 'object',
        'MMSI': 'Int64',
        'Cargo type': 'object',
        'ETA': 'object',
        'Name': 'object',
        'SOG': 'float64', 
        'Longitude': 'float64',
        'Latitude': 'float64'
    }
    
    # Load dataset
    df = dd.read_csv(input_file, dtype=dtype)

    # Find MMSI values where at least one row has SOG > 0 and valid Longitude and Latitude
    valid_mmsi = df[(df["SOG"] > 0) & (df["Longitude"] != 0) & (df["Latitude"] != 0)]["MMSI"].unique().compute()

    # Filter the entire dataset to keep only those ships
    moving_ships = df[df["MMSI"].isin(valid_mmsi)]
    
    # Filter out completely stopped ships
    removed_ships = df[~df["MMSI"].isin(valid_mmsi)]

    # Save results
    moving_ships.compute().to_csv(output_file, index=False)
    removed_ships.compute().to_csv(removed_ships_file, index=False)

    print(f"Moving ships (with at least one SOG > 0) saved to: {output_file}")
    print(f"Completely stopped ships saved to: {removed_ships_file}")

if __name__ == '__main__':
    filter_moving_ships()