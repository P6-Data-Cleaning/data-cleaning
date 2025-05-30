# Import necessary libraries
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
import os
from sklearn.model_selection import train_test_split
from math import radians, cos, sin, asin, sqrt
import numpy as np
import time

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
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch, device=h.device)

        gap_nodes = []
        offset = 0
        if batch.dim() == 0:
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

class ShipGNNWithDecoder(nn.Module):
    def __init__(self, in_feats, hidden_dim=128, gap_len=3):
        super().__init__()
        self.gnn = ShipGNN(in_feats + 8, hidden_dim)
        self.decoder = FutureDecoder(hidden_dim, gap_len)
        self.ship_type_embedding = nn.Embedding(10, 4)  # Adjust num embeddings based on actual ship types
        self.nav_status_embedding = nn.Embedding(10, 4)  # Adjust accordingly

    def forward(self, x, edge_index, batch, gap_start_idx=4):
        # Embedding categorical features before GNN
        ship_type_idx = x[:, -2].long()  # Assuming last two features are categorical indexes
        nav_status_idx = x[:, -1].long()

        ship_embed = self.ship_type_embedding(ship_type_idx)
        nav_embed = self.nav_status_embedding(nav_status_idx)

        x = x[:, :-2]  # Remove integer encoded features
        x = torch.cat([x, ship_embed, nav_embed], dim=1)

        h, _ = self.gnn(x, edge_index)  # Get hidden representation
        pred_seq = self.decoder(h, batch, gap_start_idx=gap_start_idx)
        return pred_seq

# Define the GNN model
class ShipGNN(torch.nn.Module):
    def __init__(self, in_feats, hidden_dim=128):
        super().__init__()
        # Main graph layers
        self.conv1 = GCNConv(in_feats, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Separate prediction heads
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Lat, Long
        )
        
        # Comment out time head
        # self.time_head = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim//2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim//2, 1)  # Time delta
        # )
        
        # Add proper weight initialization
        self._init_weights()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index):
        # Graph convolution layers
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        
        # Return both the hidden representation and the predictions
        # Hidden representation for the decoder
        hidden = x
        
        # Separate predictions
        pos = self.position_head(x)
        # time = self.time_head(x)
        
        # Return just position predictions
        # pred = torch.cat([pos, time], dim=1)
        pred = pos
        
        return hidden, pred


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

    features_tensor = torch.tensor(features.values, dtype=torch.float)

    for i in range(features_tensor.shape[1] - 2):  # Skip last two categorical indexes
        col = features_tensor[:, i]
        if torch.max(col) - torch.min(col) > 1e-6:
            features_tensor[:, i] = (col - torch.mean(col)) / (torch.std(col) + 1e-8)

    return features_tensor

def process_ship(ship_df, gap_start=5, gap_end=8):
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

    # Normalize target values
    lat_mean = df['Latitude'].mean()
    lat_std = df['Latitude'].std()
    lon_mean = df['Longitude'].mean()
    lon_std = df['Longitude'].std()
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

    return Data(x=x, edge_index=edge_index, y=target, 
                original_y=original_target, 
                norm_params=torch.tensor([lat_mean, lat_std, lon_mean, lon_std]))
                # norm_params=torch.tensor([lat_mean, lat_std, lon_mean, lon_std, time_mean, time_std]))

df = load_data()
grouped = df.groupby("MMSI")

grouped_list = list(grouped)
train_groups, val_groups = train_test_split(grouped_list, test_size=0.2, random_state=42)

train_data_list = []
val_data_list = []

for mmsi, ship_df in train_groups:
    data = process_ship(ship_df)
    if data is not None:
        train_data_list.append(data)

for mmsi, ship_df in val_groups:
    data = process_ship(ship_df)
    if data is not None:
        val_data_list.append(data)
    
print(f"Prepared {len(train_data_list)} ship graphs for training.")
print(f"Prepared {len(val_data_list)} ship graphs for validation.")

# Print feature statistics to check for extreme values
all_features = torch.cat([data.x for data in train_data_list])
print("Feature stats:")
for i in range(all_features.shape[1]):
    col = all_features[:, i]
    print(f"Feature {i}: min={col.min().item():.2f}, max={col.max().item():.2f}, mean={col.mean().item():.2f}, std={col.std().item():.2f}")

train_loader = DataLoader(train_data_list, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data_list, batch_size=32, shuffle=False)

model = ShipGNNWithDecoder(in_feats=train_data_list[0].x.shape[1] - 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

# Load existing model
model_save_path = 'best_ship_gnn_model_all_v2.pt'
start_epoch = 0

# Check if the model file exists
if os.path.exists(model_save_path):
    print(f"Loading pre-trained model from {model_save_path}")
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['loss']
    print(f"Resuming from epoch {start_epoch}, best loss: {best_loss:.4f}")
else:
    print("No saved model found. Starting from scratch.")
    best_loss = float('inf')

patience_counter = 0
patience_limit = 100

for epoch in range(start_epoch, 5000):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch.x, batch.edge_index, batch.batch)
            target = batch.y
            out = out.view(-1, 3, 2)  # Changed from 3 to 2 for removing time
            target = target.view(-1, 3, 2)  # Changed from 3 to 2 for removing time
            pos_loss = F.smooth_l1_loss(out, target)
            # Removed time_loss
            val_loss += pos_loss.item()
    print(f"[Validation Loss] Epoch {epoch} | Loss: {val_loss:.4f}")
    model.train()

    total_loss = 0.0

    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        target = batch.y
        
        # Check and reshape tensors if needed
        if out.dim() == 2:
            # If output is [batch_size * gap_len, 2], reshape to [batch_size, gap_len, 2]
            out = out.view(-1, 3, 2)  # Changed from 3 to 2 for removing time
            
        if target.dim() == 2:
            # If target is [batch_size * gap_len, 2], reshape to [batch_size, gap_len, 2]
            target = target.view(-1, 3, 2)  # Changed from 3 to 2 for removing time

        # Loss for position only
        loss = F.smooth_l1_loss(out, target)
        # Removed time loss

        loss.backward()
        
        # Add gradient clipping to prevent explosions
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item()

    # Early stopping if loss is NaN
    if torch.isnan(torch.tensor(total_loss)):
        print(f"Training stopped at epoch {epoch} due to NaN loss")
        break
    
    # Update learning rate
    scheduler.step(total_loss)
    
    print(f"Epoch {epoch} | Loss: {total_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Early stopping based on loss improvement
    if total_loss < best_loss:
        best_loss = total_loss
        patience_counter = 0
        
        # Save the model when we find a better one
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

with torch.no_grad():
    # Convert grouped to a list and randomly sample 10 ships
    ship_groups = list(grouped)
    import random
    random_ships = random.sample(ship_groups, min(50, len(ship_groups)))
    
    print("\nPredictions for 50 random ships:")
    print("-" * 50)
    
    # Initialize metrics accumulators
    total_error = 0.0
    total_mae_pos = 0.0
    # Removed total_mae_time
    total_haversine = 0.0
    num_valid_ships = 0
    
    for i, (mmsi, ship_df) in enumerate(random_ships):
        ship_data = process_ship(ship_df)
        
        if ship_data is not None:
            model.eval()
            with torch.no_grad():
                device = next(model.parameters()).device
                ship_data.x = ship_data.x.to(device)
                ship_data.edge_index = ship_data.edge_index.to(device)
                batch = torch.zeros(len(ship_data.x), dtype=torch.long, device=device)
                
                pred_seq = model(ship_data.x, ship_data.edge_index, batch=batch)
                
                # Ensure pred_seq has the right dimensions [1, gap_len, 3]
                if pred_seq.dim() == 2:
                    pred_seq = pred_seq.view(1, -1, 3)
                
                # Get original target for comparison
                original_target = ship_data.original_y

                # Normalize parameters
                norm_params = ship_data.norm_params
                lat_mean, lat_std = norm_params[0].item(), norm_params[1].item()
                lon_mean, lon_std = norm_params[2].item(), norm_params[3].item()
                # time_mean, time_std = norm_params[4].item(), norm_params[5].item()

                # Denormalize predictions
                denorm_preds = torch.zeros_like(pred_seq[0])
                denorm_preds[:, 0] = pred_seq[0, :, 0] * lat_std + lat_mean
                denorm_preds[:, 1] = pred_seq[0, :, 1] * lon_std + lon_mean
                # Removed time denormalization

                # Denormalize original targets
                denorm_targets = torch.zeros_like(original_target)
                if original_target.dim() == 2 and original_target.shape[0] > 0:
                    denorm_targets[:, 0] = original_target[:, 0] * lat_std + lat_mean
                    denorm_targets[:, 1] = original_target[:, 1] * lon_std + lon_mean
                    # Removed time denormalization

                print(f"Ship {i+1} (MMSI: {mmsi}):")
                for j in range(denorm_preds.shape[0]):
                    print(f"  Step {j+1} — Predicted Lat: {denorm_preds[j, 0]:.4f}, Lon: {denorm_preds[j, 1]:.4f}")

                if original_target.dim() == 2 and original_target.shape[0] > 0:
                    print(f"  True values:")
                    for j in range(original_target.shape[0]):
                        print(f"  Step {j+1} — True Lat: {denorm_targets[j, 0]:.4f}, Lon: {denorm_targets[j, 1]:.4f}")
                    
                    # Calculate error using denormalized values
                    error = torch.norm(denorm_preds - denorm_targets, dim=1).mean().item()
                    print(f"  Average Error: {error:.4f}")
                    total_error += error

                    # MAE for lat/lon
                    mae_pos = F.l1_loss(denorm_preds, denorm_targets).item()
                    print(f"  MAE Position: {mae_pos:.4f}")
                    total_mae_pos += mae_pos
                    # Removed MAE time

                    # Haversine distance errors
                    haversine_errors = [
                        haversine(t[0].item(), t[1].item(), p[0].item(), p[1].item())
                        for t, p in zip(denorm_targets, denorm_preds)
                    ]
                    mean_haversine_error = sum(haversine_errors) / len(haversine_errors)
                    print(f"  Mean Haversine Distance: {mean_haversine_error:.2f} km")
                    total_haversine += mean_haversine_error
                    
                    # Count valid ships
                    num_valid_ships += 1

                print("-" * 50)

                os.makedirs("plots", exist_ok=True)

                # Compare predictions and actual using denormalized values
                plt.figure(figsize=(8, 4))
                plt.plot([row[0].item() for row in denorm_targets], label='True Lat', marker='o')
                plt.plot([row[0].item() for row in denorm_preds], label='Pred Lat', marker='x')
                plt.legend()
                plt.title(f'Ship {mmsi} - Latitude prediction')
                plt.xlabel("Timestep")
                plt.ylabel("Latitude")
                plt.grid(True)
                plt.savefig(f"plots/ship_{mmsi}_lat_prediction.png", bbox_inches='tight')
                plt.close()
        else:
            print(f"Ship {i+1} (MMSI: {mmsi}): Insufficient data")
            print("-" * 50)
    
    # Calculate and display average metrics across all valid ships
    if num_valid_ships > 0:
        print("\n" + "=" * 50)
        print(f"AVERAGE METRICS ACROSS {num_valid_ships} SHIPS:")
        print(f"Average Error: {total_error / num_valid_ships:.4f}")
        print(f"Average MAE Position: {total_mae_pos / num_valid_ships:.4f}")
        print(f"Average Haversine Distance: {total_haversine / num_valid_ships:.2f} km")
        print("=" * 50)
    else:
        print("\nNo valid ships found for evaluation.")