import dask.dataframe as dd

# Configuration
OUTPUT_CSV = 'unique_data.csv'

def remove_duplicate(fileName):
    # Define explicit dtypes for all columns in the CSV
    DTYPES = {
        '# Timestamp': 'object',  # Use 'object' for string or mixed types
        'MMSI': 'int64',          # Use 'int64' for numeric IDs
        'Cargo type': 'object',   # Use 'object' for string or mixed types
        'ETA': 'object',          # Use 'object' for datetime strings
        'Name': 'object',         # Use 'object' for string or mixed types
    }

    # Read the CSV file using Dask with explicit dtypes
    df = dd.read_csv(fileName, dtype=DTYPES)
    
    # Drop duplicates based on the two columns
    df_unique = df.drop_duplicates(subset=['# Timestamp', 'MMSI'])
    
    # Write to CSV
    df_unique.to_csv(OUTPUT_CSV, index=False, single_file=True)

    print(f"Duplicate rows removed and saved to {OUTPUT_CSV}")
    return df_unique

    