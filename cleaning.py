import dask.dataframe as dd
import sys

# Configuration
OUTPUT_CSV = 'cleaned.csv'

def cleaning(fileName):
    # Define explicit dtypes for all columns in the CSV
    DTYPES = {
        '# Timestamp': 'object',  # Use 'object' for string or mixed types
        'MMSI': 'int64',          # Use 'int64' for numeric IDs
        'Cargo type': 'object',   # Use 'object' for string or mixed types
        'ETA': 'object',          # Use 'object' for datetime strings
        'Name': 'object',         # Use 'object' for string or mixed types
    }

    # List the columns you want to drop
    columns_to_remove = ['Heading', 'IMO', 'Callsign', 'Name', 'Cargo type',
                        'Width', 'Length', 'Type of position fixing device', 'Data source type', 'A',
                        'B', 'C', 'D']

    # Read the CSV file using Dask with explicit dtypes
    df = dd.read_csv(fileName, dtype=DTYPES)

    # Drop the redundant columns
    df = df.drop(columns=columns_to_remove)
    
    # Drop duplicates based on the two columns
    df_unique = df.drop_duplicates(subset=['# Timestamp', 'MMSI'])
    
    # Write to CSV
    df_unique.to_csv(OUTPUT_CSV, index=False, single_file=True)

    print(f"Duplicate rows removed and saved to {OUTPUT_CSV}")
    return df_unique

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cleaning.py <filename>")
        sys.exit(1)

    fileName = sys.argv[1]
    cleaning(fileName)
