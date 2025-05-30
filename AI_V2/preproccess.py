from math import cos, radians

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Constants
SEQUENCE_LENGTH = 20
MASK_START = 10
MASK_LENGTH = 5
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
EVAL_SIZE = 0.15

def normalize_features(df, columns):
    mean = df[columns].mean()
    std = df[columns].std() + 1e-8  # Avoid division by zero

    normalized_data = (df[columns] - mean) / std

    params = {
        "mean": mean,
        "std": std
    }

    return normalized_data, params

def preprocess_ais(file_path):
    df = pd.read_csv(file_path, parse_dates=["# Timestamp"])
    df.sort_values(by=["MMSI", "# Timestamp"], inplace=True)

    # Add a date column to group by day
    df["Date"] = df["# Timestamp"].dt.date

    df["DeltaTime"] = df.groupby("MMSI")["# Timestamp"].diff().dt.total_seconds()
    df["DeltaTime"] = df["DeltaTime"].fillna(0)

    df["COG"] = df["COG"].fillna(0)

    # Drop rows with invalid Latitude/Longitude
    df = df.dropna(subset=["Latitude", "Longitude"])

    # Normalize relevant numerical columns
    feature_columns = ["Latitude", "Longitude", "SOG", "COG", "DeltaTime"]
    normalized_data, norm_params = normalize_features(df, feature_columns)
    df[feature_columns] = normalized_data

    # Debugging: Check for NaN values
    if df[feature_columns].isna().any().any():
        print("NaN detected in feature columns after normalization!")
        print(df[feature_columns].isna().sum())

    sequences = []
    targets = []

    # Group by MMSI and Date to ensure sequences don't cross days
    for (mmsi, date), group in df.groupby(["MMSI", "Date"]):
        group = group.reset_index(drop=True)
        if len(group) < SEQUENCE_LENGTH:
            continue

        for i in range(len(group) - SEQUENCE_LENGTH + 1):
            window = group.iloc[i:i + SEQUENCE_LENGTH].copy()
            sequence = window[feature_columns].values

            # Add MMSI as a feature (broadcasted to the sequence length)
            mmsi_feature = np.full((SEQUENCE_LENGTH, 1), mmsi)
            sequence = np.hstack((sequence, mmsi_feature))

            # Debugging: Check for NaN in sequences
            if np.isnan(sequence).any():
                print(f"NaN detected in sequence for MMSI {mmsi} on {date} at index {i}")
                continue

            # Mask the middle segment (mask X/Y only)
            masked_sequence = sequence.copy()
            masked_sequence[MASK_START:MASK_START + MASK_LENGTH, :2] = 0  # mask X/Y only

            # Target: the true X/Y values for the masked region
            target = sequence[MASK_START:MASK_START + MASK_LENGTH, :2]

            sequences.append(masked_sequence)
            targets.append(target)

    return np.array(sequences), np.array(targets), norm_params

def split_data(sequences, targets, train_size=TRAIN_SIZE, val_size=VAL_SIZE, eval_size=EVAL_SIZE):
    # Ensure the splits add up to 1
    assert abs(train_size + val_size + eval_size - 1.0) < 1e-6, "Train, val, and eval sizes must sum to 1."

    # Split into train and temp (val + eval)
    train_sequences, temp_sequences, train_targets, temp_targets = train_test_split(
        sequences, targets, test_size=(1 - train_size), random_state=42
    )

    # Calculate the proportion of val and eval in the temp set
    val_ratio = val_size / (val_size + eval_size)

    # Split temp into val and eval
    val_sequences, eval_sequences, val_targets, eval_targets = train_test_split(
        temp_sequences, temp_targets, test_size=(1 - val_ratio), random_state=42
    )

    return train_sequences, val_sequences, eval_sequences, train_targets, val_targets, eval_targets

def remove_mmsi_feature(sequences, targets):
    # Remove the last column (MMSI) from sequences
    sequences = sequences[:, :, :-1]
    return sequences, targets


# PyTorch Dataset for AIS Trajectories
class AISTrajectoryDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def load_dataset_as_tensors(file_path):
    sequences, targets = preprocess_ais(file_path)
    return AISTrajectoryDataset(sequences, targets)

if __name__ == "__main__":
    # Example usage
    file_path = "data/cleaned_data_year.csv"  # Replace with your actual file path
    sequences, targets, norm_params = preprocess_ais(file_path)

    # Split the data
    train_sequences, val_sequences, eval_sequences, train_targets, val_targets, eval_targets = split_data(sequences, targets)

    # Remove MMSI from train and validation sets
    train_sequences, train_targets = remove_mmsi_feature(train_sequences, train_targets)
    val_sequences, val_targets = remove_mmsi_feature(val_sequences, val_targets)

    # Save the splits
    np.savez("data/processed_data_year_splits_v2.npz",
             train_sequences=train_sequences, val_sequences=val_sequences, eval_sequences=eval_sequences,
             train_targets=train_targets, val_targets=val_targets, eval_targets=eval_targets, norm_params=norm_params)

    print(f"Processed data saved with {len(train_sequences)} train, {len(val_sequences)} val, and {len(eval_sequences)} eval sequences.")

