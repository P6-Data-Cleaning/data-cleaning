import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv  # Graph Attention layer
import os
import random
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from torch_geometric.data import Data
import numpy as np
from sklearn.neighbors import NearestNeighbors


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

class FutureDecoder(nn.Module):
    def __init__(self, hidden_dim, gap_len):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.reduce = nn.Linear(hidden_dim * 2, hidden_dim)  # NEW: for feeding GRU
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, 3)  # Original: Lat, Long, Time
            nn.Linear(hidden_dim, 2)  # Modified: Only Lat, Long
        )
        self.gap_len = gap_len

    def forward(self, h, batch, gap_start_idx=4):
        # If batch is None (single graph), create a tensor of zeros with appropriate size
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        elif not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch, device=h.device)

        gap_nodes = []
        offset = 0
        if batch.dim() == 0 or len(batch) == 0:
            gap_nodes = [gap_start_idx]
        else:
            for count in batch.bincount():
                gap_nodes.append(offset + gap_start_idx)
                offset += count.item()
                
        h_before_gap = h[torch.tensor(gap_nodes, device=h.device)]  # [batch_size, hidden_dim]
        input_step = h_before_gap.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        h_t = h_before_gap.unsqueeze(0).repeat(2, 1, 1)  # [2, batch_size, hidden_dim]

        preds = []
        for _ in range(self.gap_len):
            rnn_out, h_t = self.rnn(input_step, h_t)        # [batch, 1, hidden_dim*2]
            pred = self.head(rnn_out)                       # [batch, 1, 3]
            preds.append(pred)
            input_step = self.reduce(rnn_out)               # [batch, 1, hidden_dim] → next input

        return torch.cat(preds, dim=1)  # [batch_size, gap_len, 3]


class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.2):
        super().__init__()
        # Save dropout as instance attribute
        self.dropout = dropout
        # First GAT layer: in_channels -> hidden_channels with multiple heads
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        # Second GAT layer: (hidden_channels * heads) -> out_channels, single head
        self.conv2 = GATConv(hidden_channels * heads, out_channels,
                             heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        # Apply dropout and ELU after the first layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Final layer (no activation if feeding into decoder directly)
        x = self.conv2(x, edge_index)
        return x

class ShipTrajectoryModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, embedding_dim, gap_len=3):
        super().__init__()
        # Embeddings for ship type and nav status
        self.ship_type_embedding = nn.Embedding(10, 4)  # adjust num embeddings and dim if needed
        self.nav_status_embedding = nn.Embedding(15, 4)
        # GATEncoder is now used instead of GCN for node representation
        self.encoder = GATEncoder(in_dim + 8, hidden_dim, embedding_dim, heads=4, dropout=0.2)
        self.decoder = FutureDecoder(embedding_dim, gap_len=gap_len)  # Use gap_len=3 or adjust as needed

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch  # node features, edges, batch from PyG Data
        ship_type_idx = data.ship_type_idx
        nav_status_idx = data.nav_status_idx

        if batch is not None:
            ship_emb = self.ship_type_embedding(ship_type_idx)[batch]
            nav_emb = self.nav_status_embedding(nav_status_idx)[batch]
        else:
            ship_emb = self.ship_type_embedding(ship_type_idx).unsqueeze(0).expand(x.size(0), -1)
            nav_emb = self.nav_status_embedding(nav_status_idx).unsqueeze(0).expand(x.size(0), -1)

        x = torch.cat([x, ship_emb, nav_emb], dim=1)
        node_embeddings = self.encoder(x, edge_index)  # calls GATEncoder.forward
        # Now feed node_embeddings into the decoder to predict the trajectory
        output = self.decoder(node_embeddings, batch, gap_start_idx=4)
        return output

def load_data():
    df = pd.read_csv('data/merged_data.csv', parse_dates=['# Timestamp'])
    df['# Timestamp'] = pd.to_datetime(df['# Timestamp'])
    return df

# Create features based on the ship's path
def create_features(df):
    from sklearn.preprocessing import LabelEncoder
    df["unix_ts"] = df["# Timestamp"].astype("int64") // 10**9
    df["delta_t"] = df["unix_ts"].diff().fillna(0)
    df["delta_lat"] = df["Latitude"].diff().fillna(0)
    df["delta_lon"] = df["Longitude"].diff().fillna(0)
    df["hour"] = df["# Timestamp"].dt.hour + df["# Timestamp"].dt.minute / 60.0
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["ROT"] = df["ROT"].fillna(0)  # or interpolate if better

    # New feature: velocity magnitude
    df["velocity_magnitude"] = np.sqrt(df["delta_lat"] ** 2 + df["delta_lon"] ** 2)

    if "step" not in df.columns:
        df["step"] = range(len(df))  # fallback

    # Integer encode categorical columns
    if "Ship type" in df.columns:
        le_ship = LabelEncoder()
        df["ship_type_idx"] = le_ship.fit_transform(df["Ship type"].astype(str))
    if "Navigational status" in df.columns:
        le_nav = LabelEncoder()
        df["nav_status_idx"] = le_nav.fit_transform(df["Navigational status"].astype(str))

    base_features = [
        "Latitude", "Longitude", "SOG", "COG", "delta_t", "step",
        "delta_lat", "delta_lon", "velocity_magnitude", "sin_hour", "cos_hour",
        "ship_type_idx", "nav_status_idx"
    ]
    feature_cols = [f for f in base_features if f in df.columns]
    features = df[feature_cols].astype(float)

    # Move ship_type_idx and nav_status_idx to the end
    if "ship_type_idx" in df.columns and "nav_status_idx" in df.columns:
        reordered_cols = [c for c in features.columns if c not in ["ship_type_idx", "nav_status_idx"]] + ["ship_type_idx", "nav_status_idx"]
        features = features[reordered_cols]

    features_tensor = torch.tensor(features.iloc[:, :-2].values, dtype=torch.float)

    for i in range(features_tensor.shape[1]):  # All features now are continuous
        col = features_tensor[:, i]
        if torch.max(col) - torch.min(col) > 1e-6:
            features_tensor[:, i] = (col - torch.mean(col)) / (torch.std(col) + 1e-8)

    return features_tensor

def process_ship(ship_df, gap_start=5, gap_end=15):
    if len(ship_df) < gap_end:
        return None
    
    df_known = pd.concat([ship_df.iloc[:gap_start], ship_df.iloc[gap_end:]]).reset_index(drop=True)
    df_missing = ship_df.iloc[gap_start:gap_end].reset_index(drop=True)

    df_known["step"] = range(len(df_known))

    x = create_features(df_known)

    edge_index = torch.tensor(
        [[i, i+1] for i in range(len(x) - 1)] + [[i+1, i] for i in range(len(x) - 1)],
        dtype=torch.long
    ).t().contiguous()

    # Normalize target values using only df_known for per-ship normalization
    lat_mean = df_known['Latitude'].mean()
    lat_std = df_known['Latitude'].std()
    lon_mean = df_known['Longitude'].mean()
    lon_std = df_known['Longitude'].std()
    # Comment out time normalization
    # time_deltas = df.groupby("MMSI")["# Timestamp"].diff().dropna().dt.total_seconds()
    # time_mean = time_deltas.mean()
    # time_std = time_deltas.std()
    
    # Modified target: Only position, no time
    target = torch.tensor([
        [
            (row["Latitude"] - lat_mean) / lat_std,
            (row["Longitude"] - lon_mean) / lon_std,
            # (row["# Timestamp"] - df_known.iloc[gap_start - 1]["# Timestamp"]).total_seconds() / time_std
        ]
        for _, row in df_missing.iterrows()
    ], dtype=torch.float)
    
    # Store original target for later denormalization
    original_target = target.clone()

    return Data(
        x=x,
        edge_index=edge_index,
        y=target,
        original_y=original_target,
        norm_params=torch.tensor([lat_mean, lat_std, lon_mean, lon_std]),
        ship_type_idx=torch.tensor(df_known["ship_type_idx"].iloc[0], dtype=torch.long),
        nav_status_idx=torch.tensor(df_known["nav_status_idx"].iloc[0], dtype=torch.long)
    )


df = load_data()

# Add trajectory vector and NearestNeighbors logic
df["trajectory_vector"] = df[["Latitude", "Longitude"]].rolling(window=5, min_periods=1).mean().apply(
    lambda row: f"{row['Latitude']:.5f},{row['Longitude']:.5f}", axis=1
)
trajectory_vectors = df.groupby("MMSI")["trajectory_vector"].apply(lambda x: " ".join(x)).reset_index(name="trajectory_string")
trajectory_vectors["trajectory_array"] = trajectory_vectors["trajectory_string"].apply(
    lambda s: [float(coord) for point in s.split() for coord in point.split(",")]
)

# Filter only arrays with consistent length (e.g., 100 elements = 50 lat/lon pairs)
desired_length = 100  # Adjust if needed
filtered = trajectory_vectors["trajectory_array"].apply(lambda x: len(x) == desired_length)
trajectory_matrix = np.vstack(trajectory_vectors[filtered]["trajectory_array"].values.tolist())
nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(trajectory_matrix)

grouped = df.groupby("MMSI")
grouped_list = list(grouped)
train_groups, val_groups = train_test_split(grouped_list, test_size=0.2, random_state=42)

train_data_list = []
val_data_list = []

def get_nearest_trajectories(target_vector, k=5):
    target_vector = np.array(target_vector).reshape(1, -1)
    distances, indices = nbrs.kneighbors(target_vector, n_neighbors=k)
    return trajectory_vectors.iloc[indices[0]]

# Augment training data with similar ships
for mmsi, ship_df in train_groups:
    data = process_ship(ship_df)
    if data is not None:
        train_data_list.append(data)

        # Add similar ships to batch (optional: adjust k)
        if 'trajectory_vector' in ship_df.columns:
            rolling_lat = ship_df["Latitude"].rolling(window=5, min_periods=1).mean().values
            rolling_lon = ship_df["Longitude"].rolling(window=5, min_periods=1).mean().values
            trajectory_points = [coord for pair in zip(rolling_lat, rolling_lon) for coord in pair]

            # Pad or truncate to match desired length
            desired_length = 100
            if len(trajectory_points) < desired_length:
                trajectory_points += [0.0] * (desired_length - len(trajectory_points))  # pad
            else:
                trajectory_points = trajectory_points[:desired_length]  # truncate

            vector_array = trajectory_points
            similar_ships = get_nearest_trajectories(vector_array, k=3)

            for _, sim_row in similar_ships.iterrows():
                sim_mmsi = sim_row["MMSI"]
                sim_df = df[df["MMSI"] == sim_mmsi]
                sim_data = process_ship(sim_df)
                if sim_data is not None:
                    train_data_list.append(sim_data)

for mmsi, ship_df in val_groups:
    data = process_ship(ship_df)
    if data is not None:
        val_data_list.append(data)

train_loader = DataLoader(train_data_list, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data_list, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ShipTrajectoryModel(
    in_dim=train_data_list[0].x.shape[1],
    hidden_dim=128,
    embedding_dim=128,
    gap_len=10
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

model_save_path = 'best_gat_model_all_v2.pt'
start_epoch = 0
best_loss = float('inf')

# Optional: Try to load existing checkpoint
if os.path.exists(model_save_path):
    print(f"Loading checkpoint from {model_save_path}")
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['loss']

patience_counter = 0
patience_limit = 100

for epoch in range(start_epoch, 5000):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        target = batch.y.to(device)

        output = output.view(-1, 10, 2)
        target = target.view(-1, 10, 2)

        loss = F.smooth_l1_loss(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            output = model(batch)
            target = batch.y.to(device)

            output = output.view(-1, 10, 2)
            target = target.view(-1, 10, 2)

            val_loss += F.smooth_l1_loss(output, target).item()

    scheduler.step(val_loss)

    print(f"Epoch {epoch} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': best_loss,
        }, model_save_path)
        print(f"Model saved to {model_save_path}")
    else:
        patience_counter += 1

    if patience_counter >= patience_limit:
        print(f"Early stopping at epoch {epoch}")
        break

################################################################################
# Evaluation of 50 random ships after training
################################################################################

# Define nearest trajectory search function before evaluating random ships
def get_nearest_trajectories(target_vector, k=5):
    target_vector = np.array(target_vector).reshape(1, -1)
    distances, indices = nbrs.kneighbors(target_vector, n_neighbors=k)
    return trajectory_vectors.iloc[indices[0]]

with torch.no_grad():
    # grouped must be defined in the main script, as in newMain.py
    ship_groups = list(grouped)
    random_ships = random.sample(ship_groups, min(50, len(ship_groups)))
    
    print("\nPredictions for 50 random ships:")
    print("-" * 50)
    
    total_error = 0.0
    total_mae_pos = 0.0
    total_haversine = 0.0
    num_valid_ships = 0
    
    # Use device of model
    device = next(model.parameters()).device if 'model' in locals() else torch.device('cpu')
    
    for i, (mmsi, ship_df) in enumerate(random_ships):
        ship_data = process_ship(ship_df)
        
        if ship_data is not None:
            model.eval()
            ship_data = ship_data.to(device)
            
            # Pass the entire Data object to the model
            pred_seq = model(ship_data)
            
            if pred_seq.dim() == 2:
                pred_seq = pred_seq.view(1, -1, 2)
                
            original_target = ship_data.original_y

            norm_params = ship_data.norm_params
            lat_mean, lat_std = norm_params[0].item(), norm_params[1].item()
            lon_mean, lon_std = norm_params[2].item(), norm_params[3].item()

            denorm_preds = torch.zeros_like(pred_seq[0])
            denorm_preds[:, 0] = pred_seq[0, :, 0] * lat_std + lat_mean
            denorm_preds[:, 1] = pred_seq[0, :, 1] * lon_std + lon_mean

            denorm_targets = torch.zeros_like(original_target)
            if original_target.dim() == 2 and original_target.shape[0] > 0:
                denorm_targets[:, 0] = original_target[:, 0] * lat_std + lat_mean
                denorm_targets[:, 1] = original_target[:, 1] * lon_std + lon_mean

            print(f"Ship {i+1} (MMSI: {mmsi}):")
            for j in range(denorm_preds.shape[0]):
                print(f"  Step {j+1} — Predicted Lat: {denorm_preds[j, 0]:.4f}, Lon: {denorm_preds[j, 1]:.4f}")

            if original_target.dim() == 2 and original_target.shape[0] > 0:
                print(f"  True values:")
                for j in range(original_target.shape[0]):
                    print(f"  Step {j+1} — True Lat: {denorm_targets[j, 0]:.4f}, Lon: {denorm_targets[j, 1]:.4f}")
                
                error = torch.norm(denorm_preds - denorm_targets, dim=1).mean().item()
                print(f"  Average Error: {error:.4f}")
                total_error += error

                mae_pos = F.l1_loss(denorm_preds, denorm_targets).item()
                print(f"  MAE Position: {mae_pos:.4f}")
                total_mae_pos += mae_pos

                haversine_errors = [
                    haversine(t[0].item(), t[1].item(), p[0].item(), p[1].item())
                    for t, p in zip(denorm_targets, denorm_preds)
                ]
                mean_haversine_error = sum(haversine_errors) / len(haversine_errors)
                print(f"  Mean Haversine Distance: {mean_haversine_error:.2f} km")
                total_haversine += mean_haversine_error
                
                num_valid_ships += 1

            print("-" * 50)

            """ os.makedirs("plots", exist_ok=True)
            plt.figure(figsize=(8, 4))
            plt.plot([row[0].item() for row in denorm_targets], label='True Lat', marker='o')
            plt.plot([row[0].item() for row in denorm_preds], label='Pred Lat', marker='x')
            plt.legend()
            plt.title(f'Ship {mmsi} - Latitude prediction')
            plt.xlabel("Timestep")
            plt.ylabel("Latitude")
            plt.grid(True)
            plt.savefig(f"plots/ship_{mmsi}_lat_prediction.png", bbox_inches='tight')
            plt.close() """
        else:
            print(f"Ship {i+1} (MMSI: {mmsi}): Insufficient data")
            print("-" * 50)
    
    if num_valid_ships > 0:
        print("\n" + "=" * 50)
        print(f"AVERAGE METRICS ACROSS {num_valid_ships} SHIPS:")
        print(f"Average Error: {total_error / num_valid_ships:.4f}")
        print(f"Average MAE Position: {total_mae_pos / num_valid_ships:.4f}")
        print(f"Average Haversine Distance: {total_haversine / num_valid_ships:.2f} km")
        print("=" * 50)
    else:
        print("\nNo valid ships found for evaluation.")