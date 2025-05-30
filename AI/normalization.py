import pandas as pd
import numpy as np
import torch
import os
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

def normalization(fileName, save_scaler=True):
    print(f"Reading file: {fileName}")
    df = pd.read_csv(fileName)

    # Convert timestamp and ETA to unix timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
    df['timestamp'] = df['timestamp'].astype(np.int64) // 10**9
    df['eta'] = pd.to_datetime(df['eta'], format='%d/%m/%Y %H:%M:%S')
    df['eta'] = df['eta'].astype(np.int64) // 10**9

    # Label encode navigational_status and ship_type
    label_encoders = {}
    for column in ['navigational_status', 'ship_type']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Drop the Destination column
    df = df.drop(columns=['destination'])
    
    # Normalize the data
    continuous_cols = ['timestamp', 'latitude', 'longitude', 'cog', 'sog', 'draught', 'eta']
    scaler = StandardScaler()

    df_scaled_cont = pd.DataFrame(scaler.fit_transform(df[continuous_cols]), columns=continuous_cols, index=df.index)
    df[continuous_cols] = df_scaled_cont


    # Print scaling parameters
    print("Scaling parameters:")
    for i, col in enumerate(continuous_cols):
        print(f"{col}: mean={scaler.mean_[i]:.6f}, scale={scaler.scale_[i]:.6f}")
    
    print("DataFrame head after normalizing the data:")
    print(df.head())

    # Save scaler for future inverse transformation
    if save_scaler:
        with open('label_encoders.pkl', 'wb') as f:
            pickle.dump(label_encoders, f)
        print("Label encoders saved to 'label_encoders.pkl'")

        with open('standard_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        print("Scaler saved to 'standard_scaler.pkl'")

    return df, scaler


    """ # Convert to numpy array with explicit float type
    data = df.to_numpy(dtype=np.float32)
    print(f"NumPy array shape: {data.shape}")
    
    # Check for any NaN values that might cause issues
    if np.isnan(data).any():
        print("Warning: NaN values found in the data. Filling with 0.")
        data = np.nan_to_num(data, nan=0.0)
 """
    
# Inverse transform example
def inverse_transform_example(df, scaler, continuous_cols):
    # Get scaled data
    scaled_data = df[continuous_cols].values
    
    # Inverse transform
    original_data = scaler.inverse_transform(scaled_data)
    
    # Create DataFrame with original values
    df_original = pd.DataFrame(
        original_data, 
        columns=continuous_cols,
        index=df.index
    )
    
    print("Inverse transformed data:")
    print(df_original.head())
    
    return df_original

if __name__ == "__main__":
    
    file_path = "cleaned_data27_reduced.csv"
    result, scaler = normalization(file_path)

    # Inverse transform example
    #continuous_cols = ['timestamp', 'latitude', 'longitude', 'cog', 'sog', 'draught', 'eta']
    #inverse_transform_example(result, scaler, continuous_cols)

    print("Processing completed successfully")
    #print(result)