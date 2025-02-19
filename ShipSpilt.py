import os
import dask
import time
from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
from dask import delayed

def save_mmsi_to_csv(df, mmsi, output_folder):
    mmsi_df = df[df['MMSI'] == mmsi].sort_values(by='# Timestamp')
    mmsi_output_path = os.path.join(output_folder, f"{mmsi}.csv")
    mmsi_df.to_csv(mmsi_output_path, index=False)

def main():
    # Setup Dask cluster (adjust for your hardware)
    from distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=8, threads_per_worker=16, memory_limit="300GB")
    client = Client(cluster)

    start_time = time.time()

    # File paths
    script_dir = os.getcwd()
    file_path = os.path.join(script_dir, 'moving_ships.csv')
    output_folder = os.path.join(script_dir, "Ships_data")
    os.makedirs(output_folder, exist_ok=True)

    # Data types
    dtype = {
        '# Timestamp': 'object',
        'MMSI': 'int64',
        'Cargo type': 'object',
        'ETA': 'object',
        'Name': 'object'
    }

    # Read the CSV file
    df = dd.read_csv(file_path, dtype=dtype)

    # Loop through each unique MMSI and save to separate CSV files in parallel
    unique_mmsi = df['MMSI'].unique().compute()
    tasks = [delayed(save_mmsi_to_csv)(df, mmsi, output_folder) for mmsi in unique_mmsi]
    dask.compute(*tasks)

    # Calculate the number of files created
    num_files = len(os.listdir(output_folder))

    print(f"Time taken: {time.time() - start_time} seconds")
    print(f"Number of ship files created: {num_files}")

if __name__ == "__main__":
    main()