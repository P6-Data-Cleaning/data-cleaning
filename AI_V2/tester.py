import numpy as np

data = np.load("data/processed_data_feb_v4.npz")
sequences, targets = data["sequences"], data["targets"]

print("Sequences stats:")
print(f"Min: {sequences.min()}, Max: {sequences.max()}, Mean: {sequences.mean()}, Std: {sequences.std()}")
print("Targets stats:")
print(f"Min: {targets.min()}, Max: {targets.max()}, Mean: {targets.mean()}, Std: {targets.std()}")

# Check for NaN or inf
print("Sequences NaN:", np.isnan(sequences).any())
print("Targets NaN:", np.isnan(targets).any())
print("Sequences Inf:", np.isinf(sequences).any())
print("Targets Inf:", np.isinf(targets).any())
