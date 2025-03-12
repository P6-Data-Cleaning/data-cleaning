import pandas as pd
import sys
import time

def trajectory_reducer(data_file):
    start_time = time.time()
    
    df = pd.read_csv(data_file)
    start_rows = len(df)

    # count the number of rows

    df.to_csv(f'outputs/csv{mmsi}_reduced.csv', index=False)
    print(f"Data extracted for MMSI {mmsi}")
    print(f"Reduced to {len(df)} rows from {start_rows}")
    print(f"It took: {time.time() - start_time} seconds")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python trajectory_reducer.py <filename>")
        sys.exit(1)

    fileName = sys.argv[1]
    mmsi = int(sys.argv[2])
    trajectory_reducer(fileName)
