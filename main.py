import pandas as pd
import os
import time
from duplicate import remove_duplicate
from ShipSpilt import split_ships

def count_rows_columns(file_path):
    df = pd.read_csv(file_path)
    rows, columns = df.shape

    # Remove duplicate rows
    remove_duplicate(df)
    
    return rows, columns

print("Reading the CSV file...")
start_time = time.time()

# Get the directory of the current script
script_dir = os.path.dirname(__file__)
# Construct the full path to the CSV file
file_path = os.path.join(script_dir, 'aisdk-2025-02-14.csv')
output_folder = os.path.join(script_dir, "Ships_data")  # Absolute path
print(f"File path: {file_path}")

# Call the split_ships function
split_ships(file_path, output_folder)