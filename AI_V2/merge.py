import pandas as pd
import glob
import os
import multiprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor

def process_file(file):
    """Process a single CSV file and return the dataframe with geometry removed."""
    print(f"Reading file: {os.path.basename(file)}")
    df = pd.read_csv(file)
    # Remove the geometry column if it exists
    if 'geometry' in df.columns:
        df = df.drop(columns=['geometry'])
    return df

def merge_csv_files():
    # Path to directory containing CSV files
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/months")
    
    # Get all CSV files in the directory
    all_files = glob.glob(os.path.join(path, "*.csv"))
    
    # Get the number of available CPUs
    num_cpus = multiprocessing.cpu_count()
    print(f"Using {num_cpus} CPU cores for processing")
    
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        dfs = list(executor.map(process_file, all_files))
    
    # Check if we have any dataframes to merge
    if not dfs:
        print("No CSV files found to merge!")
        return
    
    # Concatenate all dataframes
    print("Combining all dataframes...")
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Sort by MMSI and Timestamp
    print("Sorting by MMSI and Timestamp...")
    combined_df = combined_df.sort_values(by=['MMSI', '# Timestamp'])
    
    # Save the merged file
    output_path = os.path.join(path, "merged_data.csv")
    print(f"Saving merged file...")
    combined_df.to_csv(output_path, index=False)
    print(f"Merged file saved to: {output_path}")
    print(f"Total rows in merged file: {len(combined_df)}")

if __name__ == "__main__":
    # This is required for Windows to work correctly with multiprocessing
    merge_csv_files()