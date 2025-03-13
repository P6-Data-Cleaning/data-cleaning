import pandas as pd
import sys
import time
import os

def trajectory_reducer(df, threshold=2.5):
    start_rows = len(df)
    
    # Group by MMSI and process each vessel's data separately
    result_dfs = []
    
    for mmsi, group in df.groupby('MMSI'):
        # Sort by timestamp if needed
        group = group.sort_values('# Timestamp')
        
        # Initialize with first row
        reduced_group = [group.iloc[0]]
        prevRow = group.iloc[0]
        
        # Process remaining rows for this MMSI
        for _, row in group.iloc[1:].iterrows():
            delta = abs(prevRow.COG - row.COG)
            if delta >= threshold:
                reduced_group.append(row)
                prevRow = row
        
        # Convert list of rows back to DataFrame
        result_df = pd.DataFrame(reduced_group)
        result_dfs.append(result_df)
    
    # Combine all reduced groups
    final_df = pd.concat(result_dfs)
    return final_df

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python trajectory_reducer.py <filename>")
        sys.exit(1)

    fileName = sys.argv[1]
    # Read the CSV file into a DataFrame
    df = pd.read_csv(fileName)
    trajectory_reducer(df)
