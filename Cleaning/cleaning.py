import dask.dataframe as dd
import sys
import time

# Configuration
OUTPUT_CSV = 'cleaned.csv'

def measure_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} execution time: {time.time() - start_time:.2f} seconds")
        return result
    return wrapper

@measure_performance
def cleaning(df):
    print(f"Cleaning start: {len(df)} rows")
    # Define explicit dtypes for all columns in the CSV

    # List the columns you want to drop
    columns_to_remove = ['IMO', 'Callsign', 'Name', 'Cargo type',
                        'Width', 'Length', 'Type of position fixing device', 'Data source type', 'A',
                        'B', 'C', 'D']

    # Drop the redundant columns
    df = df.drop(columns=columns_to_remove)
    
    # Drop duplicates based on the two columns
    df_unique = df.drop_duplicates(subset=['# Timestamp', 'MMSI'])

    print(f"Duplicate rows removed")
    print(f"Cleaning end: {len(df_unique)} rows")
    return df_unique

# This is an entry point so that the script can be run from the command line ex - python cleaning.py asisdk-2025-02-14.csv
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cleaning.py <filename>")
        sys.exit(1)

    fileName = sys.argv[1]
    
    cleaned = cleaning(fileName)

     # Write to CSV
    cleaned.to_csv(OUTPUT_CSV, index=False, single_file=True)
    print(f"Saved to {OUTPUT_CSV}")
