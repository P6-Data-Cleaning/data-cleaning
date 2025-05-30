import pandas as pd
import math

# 1-60 days: training, 61-75 days: validation, 76-90 days: test
TRAINING = 61
VALIDATION = 76
# TEST = 90 which is just end of the dataset

def splitting(df, scaler):
    # Strategy: split the dataset into 3 parts: training, validation and test

    df = df.sort_values(by='timestamp')

    first_day = df['timestamp'].iloc[0]
    last_day = df['timestamp'].iloc[-1]

    # Scale the timestamp column
    first_day = first_day * scaler.scale_[0] + scaler.mean_[0]
    last_day = last_day * scaler.scale_[0] + scaler.mean_[0]

    print(f"First day: {first_day}, Last day: {last_day}")

    # Calculate the number of days in the dataset (they are in unix format)
    num_days = math.ceil((last_day - first_day) / 86400)
    print(f"Number of days: {num_days}")

    # Calculate the split points
    split1 = first_day + TRAINING * 86400
    split2 = first_day + VALIDATION * 86400
    print(f"Split points: {split1}, {split2}")

    # Scale the split points
    train_end_point = (split1 - 1 - scaler.mean_[0]) / scaler.scale_[0]
    valid_start_point = (split1 - scaler.mean_[0]) / scaler.scale_[0]
    valid_end_point = (split2 - 1 - scaler.mean_[0]) / scaler.scale_[0]
    test_start_point = (split2 - scaler.mean_[0]) / scaler.scale_[0]
    print(f"Split points in scaled format: {train_end_point}, {valid_start_point}, {valid_end_point}, {test_start_point}")
    

    # Split the dataset
    training_set = df[df['timestamp'] <= train_end_point]
    validation_set = df[(df['timestamp'] >= valid_start_point) & (df['timestamp'] <= valid_end_point)]
    test_set = df[df['timestamp'] >= test_start_point]

    # Group by MMSI and sort by timestamp
    training_set = training_set.groupby('mmsi').apply(lambda x: x.sort_values('timestamp'))
    validation_set = validation_set.groupby('mmsi').apply(lambda x: x.sort_values('timestamp'))
    test_set = test_set.groupby('mmsi').apply(lambda x: x.sort_values('timestamp'))

    first_day = training_set['timestamp'].iloc[0]
    last_day = training_set['timestamp'].iloc[-1]

    print(f"Training set: {first_day} to {last_day}")

    first_day = validation_set['timestamp'].iloc[0]
    last_day = validation_set['timestamp'].iloc[-1]

    print(f"Validation set: {first_day} to {last_day}")

    first_day = test_set['timestamp'].iloc[0]

    print(f"Test set: {first_day} to end of dataset")

    return training_set, validation_set, test_set