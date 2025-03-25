import pandas as pd
import numpy as np
import torch
import os

def spatial_indexing(fileName):
    print(f"Reading file: {fileName}")
    df = pd.read_csv(fileName)
    
    # Check data types and print information
    print("DataFrame info:")
    print(df.info())
    print("DataFrame head:")
    print(df.head())

    # Convert timestamp and ETA to unix timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
    df['timestamp'] = df['timestamp'].astype(np.int64) // 10**9
    df['eta'] = pd.to_datetime(df['eta'], format='%d/%m/%Y %H:%M:%S')
    df['eta'] = df['eta'].astype(np.int64) // 10**9

    # Hot encode the categorical columns
    df = pd.get_dummies(df, columns=['navigational_status', 'ship_type'])

    # Drop the Destination column
    df = df.drop(columns=['destination'])
    
    print("DataFrame head after converting timestamp and ETA:")
    print(df.head())

    print("DataFrame info after converting timestamp and ETA:")
    print(df.info())
    
    # Convert to numpy array with explicit float type
    data = df.to_numpy(dtype=np.float32)
    print(f"NumPy array shape: {data.shape}")
    
    # Check for any NaN values that might cause issues
    if np.isnan(data).any():
        print("Warning: NaN values found in the data. Filling with 0.")
        data = np.nan_to_num(data, nan=0.0)
    
    # Now convert to torch tensor
    positions = torch.tensor(data, dtype=torch.float32)
    
    # Move to CUDA if available
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        positions = positions.to("cuda")
    else:
        print("CUDA not available, using CPU")
    
    return positions

if __name__ == "__main__":
    print("Current directory:", os.getcwd())
    
    file_path = "cleaned_data27_reduced.csv"
    result = spatial_indexing(file_path)
    print(f"Tensor shape: {result.shape}")
    print("Processing completed successfully")
    print(result)