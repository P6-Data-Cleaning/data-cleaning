import dask.dataframe as dd
import glob
import os

def read_multiple_csv_files(directory_path, dtypes):
    # Create a glob pattern for all CSV files in the directory
    glob_pattern = os.path.join(directory_path, "*.csv")
    
    # Read all matching CSV files into a single Dask DataFrame
    df = dd.read_csv(glob_pattern, dtype=dtypes)
    
    return df