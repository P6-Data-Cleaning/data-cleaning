# Import necessary libraries
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

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
        
        self.time_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)  # Time delta
        )
        
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
        
        # Separate predictions
        pos = self.position_head(x)
        time = self.time_head(x)
        
        # Combine predictions
        return torch.cat([pos, time], dim=1)


def load_data():
    df = pd.read_csv('cleaned_data.csv', parse_dates=['# Timestamp'])
    df['# Timestamp'] = pd.to_datetime(df['# Timestamp'])
    return df

# Create features based on the ship's path
def create_features(df):
    df["unix_ts"] = df["# Timestamp"].astype("int64") // 10**9
    df["delta_t"] = df["unix_ts"].diff().fillna(0)

    features = df[["Latitude", "Longitude", "SOG", "COG", "delta_t"]]
    
    # Normalize features to prevent numerical instability
    features_tensor = torch.tensor(features.values, dtype=torch.float)
    
    # Skip normalization if all values are the same (would cause division by zero)
    for i in range(features_tensor.shape[1]):
        col = features_tensor[:, i]
        if torch.max(col) - torch.min(col) > 1e-6:  # Only normalize if there's variation
            features_tensor[:, i] = (col - torch.mean(col)) / (torch.std(col) + 1e-8)
    
    return features_tensor

def process_ship(ship_df, gap_start=5, gap_end=8):
    if len(ship_df) < gap_end:
        return None
    
    df_known = pd.concat([ship_df.iloc[:gap_start], ship_df.iloc[gap_end:]]).reset_index(drop=True)
    df_missing = ship_df.iloc[gap_start:gap_end].reset_index(drop=True)

    x = create_features(df_known)

    edge_index = torch.tensor(
        [[i, i+1] for i in range(len(x) - 1)] + [[i+1, i] for i in range(len(x) - 1)],
        dtype=torch.long
    ).t().contiguous()

    # Label: First missing point
    target = torch.tensor([
        df_missing.iloc[0]["Latitude"],
        df_missing.iloc[0]["Longitude"],
        (df_missing.iloc[0]["# Timestamp"] - df_known.iloc[gap_start - 1]["# Timestamp"]).total_seconds()
    ], dtype=torch.float)
    
    # Store original target for later denormalization
    original_target = target.clone()
    
    # Normalize target values
    lat_mean, lat_std = 55.0, 5.0  # Approximate values for the region
    lon_mean, lon_std = 10.0, 3.0  # Approximate values for the region
    time_mean, time_std = 60.0, 60.0  # Assuming most gaps are around 1 minute
    
    target[0] = (target[0] - lat_mean) / lat_std
    target[1] = (target[1] - lon_mean) / lon_std
    target[2] = (target[2] - time_mean) / time_std

    return Data(x=x, edge_index=edge_index, y=target, 
                original_y=original_target, 
                norm_params=torch.tensor([lat_mean, lat_std, lon_mean, lon_std, time_mean, time_std]))

df = load_data()
grouped = df.groupby("MMSI")

data_list = []
for mmsi, ship_df in grouped:
    data = process_ship(ship_df)
    if data is not None:
        data_list.append(data)
    
print(f"Prepared {len(data_list)} ship graphs for training.")

# Print feature statistics to check for extreme values
all_features = torch.cat([data.x for data in data_list])
print("Feature stats:")
for i in range(all_features.shape[1]):
    col = all_features[:, i]
    print(f"Feature {i}: min={col.min().item():.2f}, max={col.max().item():.2f}, mean={col.mean().item():.2f}, std={col.std().item():.2f}")

loader = DataLoader(data_list, batch_size=4, shuffle=True)

model = ShipGNN(in_feats=data_list[0].x.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

best_loss = float('inf')
patience_counter = 0
patience_limit = 30

for epoch in range(500):
    model.train()
    total_loss = 0.0

    for batch in loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)

        # Get predictions at the node just before the gap
        mask = []
        offset = 0
        for x in batch.x.split(batch.batch.bincount().tolist()):
            mask.append(offset + 4)  # 4 == gap_start - 1
            offset += x.size(0)
        pred = out[torch.tensor(mask, dtype=torch.long)]
        
        # Reshape target
        target = batch.y.view(-1, 3)
        
        # Weighted loss to balance position and time prediction
        pos_loss = F.mse_loss(pred[:, :2], target[:, :2])
        time_loss = F.mse_loss(pred[:, 2], target[:, 2])
        loss = pos_loss + 0.5 * time_loss  # Less weight on time prediction
        
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
    else:
        patience_counter += 1
        
    if patience_counter >= patience_limit:
        print(f"Early stopping at epoch {epoch}")
        break

with torch.no_grad():
    # Convert grouped to a list and randomly sample 10 ships
    ship_groups = list(grouped)
    import random
    random_ships = random.sample(ship_groups, min(10, len(ship_groups)))
    
    print("\nPredictions for 10 random ships:")
    print("-" * 50)
    
    for i, (mmsi, ship_df) in enumerate(random_ships):
        ship_data = process_ship(ship_df)
        
        if ship_data is not None:
            model.eval()
            out = model(ship_data.x, ship_data.edge_index)
            pred = out[4]  # node just before the gap
            
            # Get normalization parameters
            norm_params = ship_data.norm_params
            lat_mean, lat_std = norm_params[0].item(), norm_params[1].item()
            lon_mean, lon_std = norm_params[2].item(), norm_params[3].item()
            time_mean, time_std = norm_params[4].item(), norm_params[5].item()
            
            # Denormalize prediction
            denorm_pred = torch.zeros(3)
            denorm_pred[0] = pred[0] * lat_std + lat_mean
            denorm_pred[1] = pred[1] * lon_std + lon_mean
            denorm_pred[2] = pred[2] * time_std + time_mean
            
            print(f"Ship {i+1} (MMSI: {mmsi}):")
            print(f"  Predicted: {denorm_pred.tolist()}")
            print(f"  True:      {ship_data.original_y.tolist()}")
            print(f"  Error:     {torch.norm(denorm_pred - ship_data.original_y).item():.4f}")
            print("-" * 50)
        else:
            print(f"Ship {i+1} (MMSI: {mmsi}): Insufficient data")
            print("-" * 50)