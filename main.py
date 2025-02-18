import pandas as pd
import os
import time
from duplicate import remove_duplicate

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
print(f"File path: {file_path}")

rows, columns = count_rows_columns(file_path)
print(f"Number of rows: {rows}")
print(f"Number of columns: {columns}")
print(f"Execution time: {time.time() - start_time} seconds")