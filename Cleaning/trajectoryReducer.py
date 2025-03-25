import pandas as pd
import sys
import time
import os

def trajectory_reducer(df):
    start_rows = len(df)
    result_dfs = []
    
    for mmsi, group in df.groupby('MMSI'):
        group = group.sort_values('# Timestamp')
        
        reduced_group = [group.iloc[0]]
        prevRow = group.iloc[0]
        
        for _, row in group.iloc[1:].iterrows():
            speed = row.SOG

            if speed < 10:
                threshold = 3
            elif speed < 20:
                threshold = 6
            else:
                threshold = 9
            
            delta = abs(prevRow.COG - row.COG)
            if delta >= threshold:
                reduced_group.append(row)
                prevRow = row

        if (len(reduced_group) == 1):
            print(f"Warning: Only one row remaining in the DataFrame after trajectory reducer (removing all): {reduced_group[0]['MMSI']}")
            result_dfs.append(pd.DataFrame(columns=df.columns))
            continue

        # Convert list of rows back to DataFrame
        result_df = pd.DataFrame(reduced_group)
        result_dfs.append(result_df)
    
    # Combine all reduced groups
    final_df = pd.concat(result_dfs)
    end_rows = len(final_df)
    reduction_percent = ((start_rows - end_rows) / start_rows) * 100
    print(f"Reduced dataset from {start_rows} to {end_rows} rows ({reduction_percent:.2f}% reduction)")
    return final_df

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python trajectory_reducer.py <filename>")
        sys.exit(1)

    fileName = sys.argv[1]
    df = pd.read_csv(fileName)
    result_df = trajectory_reducer(df)
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        base_name = os.path.basename(fileName)
        name_parts = os.path.splitext(base_name)
        output_file = f"{name_parts[0]}_trajectory_reduced{name_parts[1]}"
    result_df.to_csv(output_file, index=False)
