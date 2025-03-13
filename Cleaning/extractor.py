import pandas as pd
import sys
import time

def extractor(data_file, mmsi):
    start_time = time.time()
    
    df = pd.read_csv(data_file)
    df = df[df['MMSI'] == mmsi]

    # count the number of rows

    df.to_csv(f'outputs/{mmsi}.csv', index=False)
    print(f"Data extracted for MMSI {mmsi}")
    print(f"Found {len(df)} rows for MMSI {mmsi}")
    print(f"It took: {time.time() - start_time} seconds")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extractor.py <filename> <mmsi>")
        sys.exit(1)

    fileName = sys.argv[1]
    mmsi = int(sys.argv[2])
    extractor(fileName, mmsi)
