import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from torch_geometric.data import Data
import numpy as np
from sklearn.preprocessing import LabelEncoder


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to input
        # x shape: [batch_size, seq_len, d_model]
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class GraphToSequenceConverter(nn.Module):
    def __init__(self, node_features, seq_features, max_seq_len=500):  # Add max_seq_len
        super().__init__()
        self.projection = nn.Linear(node_features, seq_features)
        self.max_seq_len = max_seq_len
        
    def forward(self, x, edge_index, batch=None):
        # Project node features
        x = self.projection(x)
        
        # Convert graph nodes to sequences based on edges
        if batch is None:
            # For a single graph, sort nodes by their order in the trajectory
            nodes = torch.unique(edge_index[0], sorted=True)
            # Limit sequence length to prevent OOM
            if len(nodes) > self.max_seq_len:
                nodes = nodes[:self.max_seq_len]
            return x[nodes].unsqueeze(0)  # [1, seq_len, features]
        else:
            # For batched graphs, create sequences for each graph
            sequences = []
            max_len_observed = 0
            for i in torch.unique(batch):
                # Get nodes for this graph
                mask = batch == i
                graph_nodes = torch.arange(x.size(0))[mask]
                max_len_observed = max(max_len_observed, len(graph_nodes))
                
                # Limit sequence length
                if len(graph_nodes) > self.max_seq_len:
                    graph_nodes = graph_nodes[:self.max_seq_len]
                    
                # Sort nodes by their position in the trajectory
                sequences.append(x[graph_nodes])
            
            print(f"Max sequence length in batch: {max_len_observed}")
            
            # Pad sequences to the same length
            padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
            return padded_sequences  # [batch_size, max_seq_len, features]

class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, gap_len, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gap_len = gap_len
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Create transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Output projection to predict coordinates
        self.output_proj = nn.Linear(hidden_dim, 2)  # lat, lon
        
        # Query embedding to initialize the sequence
        self.query_embed = nn.Parameter(torch.randn(1, gap_len, hidden_dim))
        
    def forward(self, h, batch, gap_start_idx=4):
        # If batch is None (single graph), create a tensor of zeros with appropriate size
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        elif not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch, device=h.device)
            
        # Extract nodes before the gap - SAFELY
        try:
            # Get valid indices for gap nodes, ensuring they are in bounds
            node_count = h.size(0)
            
            # Create safe gap nodes
            gap_nodes = []
            
            # If batch has multiple graphs
            if batch.dim() > 0 and len(batch) > 0:
                # Get counts of nodes per graph
                graph_sizes = []
                for i in torch.unique(batch):
                    size = (batch == i).sum().item()
                    graph_sizes.append(size)
                
                # For each graph, find a safe index for the gap node
                offset = 0
                for size in graph_sizes:
                    # Make sure gap_start_idx is within bounds for this graph
                    safe_idx = min(gap_start_idx, max(0, size - 1))
                    gap_node_idx = offset + safe_idx
                    
                    # Double-check if the index is valid
                    if 0 <= gap_node_idx < node_count:
                        gap_nodes.append(gap_node_idx)
                    else:
                        # Fallback to a safe index in this graph
                        safe_fallback = offset if offset < node_count else 0
                        gap_nodes.append(safe_fallback)
                    
                    offset += size
            else:
                # For a single graph, just use the first node as a fallback
                gap_nodes = [0]
        
        except Exception as e:
            print(f"Error in gap node calculation: {e}")
            # Fallback to using the first node
            gap_nodes = [0]
            
        # Make sure we have at least one valid index
        if not gap_nodes:
            gap_nodes = [0]
            
        # Convert to tensor and ensure all indices are valid
        gap_nodes_tensor = torch.tensor(gap_nodes, device=h.device)
        gap_nodes_tensor = torch.clamp(gap_nodes_tensor, 0, h.size(0) - 1)
        
        # Get embeddings safely
        memory = h[gap_nodes_tensor].unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Create batch size copies of the query embedding
        batch_size = memory.size(0)
        tgt = self.query_embed.expand(batch_size, -1, -1)  # [batch_size, gap_len, hidden_dim]
        
        # Apply positional encoding
        tgt = self.pos_encoder(tgt)
        
        # Create attention mask safely
        try:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.gap_len).to(h.device)
        except Exception as e:
            print(f"Error generating mask: {e}")
            # Create a simple causal mask manually as fallback
            tgt_mask = torch.triu(torch.ones(self.gap_len, self.gap_len) * float('-inf'), diagonal=1).to(h.device)
        
        # Repeat memory for transformer decoder
        memory_expanded = memory.repeat(1, self.gap_len, 1)
        
        # Apply transformer decoder with error handling
        try:
            output = self.transformer_decoder(
                tgt,
                memory_expanded,
                tgt_mask=tgt_mask
            )
        except Exception as e:
            print(f"Error in transformer decoder: {e}")
            # Fallback: just use the query embeddings directly
            output = tgt
            
        # Project to coordinates
        predictions = self.output_proj(output)  # [batch_size, gap_len, 2]
        
        return predictions


class TransformerEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        # Project input features to hidden dimension
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_channels)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=nhead,
            dim_feedforward=hidden_channels*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Graph to sequence converter
        self.graph_to_seq = GraphToSequenceConverter(in_channels, hidden_channels)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        # Convert graph to sequence representation
        seq = self.graph_to_seq(x, edge_index)
        
        # Apply positional encoding
        seq = self.pos_encoder(seq)
        
        # Apply transformer encoder
        output = self.transformer_encoder(seq)
        
        # Project to output dimension
        output = self.output_proj(output)
        
        # Convert back to original node ordering
        # For simplicity, we'll just return the sequence output
        # In a real application, you might want to map this back to node representation
        return output.squeeze(0)  # [seq_len, out_channels]


class ShipTrajectoryTransformerModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, embedding_dim, gap_len=3, nhead=8, num_encoder_layers=4, num_decoder_layers=4, dropout=0.1):
        super().__init__()
        # Enhanced embeddings with larger dimension
        self.ship_type_embedding = nn.Embedding(10, 8)  # Increased from 4 to 8
        self.nav_status_embedding = nn.Embedding(15, 8)  # Increased from 4 to 8
        
        # Additional feature preprocessing
        self.feature_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Transformer encoder with more layers
        self.encoder = TransformerEncoder(
            hidden_dim + 16,  # 64 + 16 = 80
            hidden_dim,
            embedding_dim,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dropout=dropout
        )
        
        # Enhanced decoder
        self.decoder = TransformerDecoder(
            embedding_dim,
            gap_len=gap_len,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dropout=dropout
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        ship_type_idx = data.ship_type_idx
        nav_status_idx = data.nav_status_idx

        # Project features first
        x = self.feature_proj(x)

        if batch is not None:
            ship_emb = self.ship_type_embedding(ship_type_idx)[batch]
            nav_emb = self.nav_status_embedding(nav_status_idx)[batch]
        else:
            ship_emb = self.ship_type_embedding(ship_type_idx).unsqueeze(0).expand(x.size(0), -1)
            nav_emb = self.nav_status_embedding(nav_status_idx).unsqueeze(0).expand(x.size(0), -1)

        # Concatenate projected features with embeddings
        x = torch.cat([x, ship_emb, nav_emb], dim=1)
        
        # Encode using transformer encoder
        node_embeddings = self.encoder(x, edge_index)
        
        # Decode using transformer decoder to predict trajectory
        output = self.decoder(node_embeddings, batch, gap_start_idx=4)
        
        return output


def load_data():
    try:
        file_path = 'data/cleaned_data_feb_done.csv'
        if not os.path.exists(file_path):
            print(f"ERROR: Data file not found at {file_path}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Available files in data directory:")
            if os.path.exists('data'):
                print(os.listdir('data'))
            else:
                print("data directory not found!")
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        df = pd.read_csv(file_path, parse_dates=['# Timestamp'])
        df['# Timestamp'] = pd.to_datetime(df['# Timestamp'])
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


# Create features based on the ship's path
def create_features(df):
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

    df["acceleration"] = df["velocity_magnitude"].diff().fillna(0)
    df["bearing_change"] = df["COG"].diff().fillna(0)

    # Cyclical encoding for day of week
    df["day_of_week"] = df["# Timestamp"].dt.dayofweek
    df["sin_day"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_day"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

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
        "acceleration", "bearing_change", "sin_day", "cos_day",
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

def get_gap_length(epoch):
        if epoch < 50:
            return 3
        elif epoch < 100:
            return 5
        else:
            return 8

def process_ship(ship_df, epoch=None, gap_start=5, augment=True):
    
    def augment_trajectory(df):
        # Random time warping
        if random.random() > 0.5:
            df["# Timestamp"] += pd.to_timedelta(np.random.normal(0, 3600), 's')
        
        # Random noise
        if random.random() > 0.7:
            df[["Latitude", "Longitude"]] += np.random.normal(0, 0.001, size=(len(df), 2))
        
        return df
    
    if augment and random.random() > 0.5:  # 50% chance to augment
        ship_df = augment_trajectory(ship_df.copy())

    if epoch is not None:
        gap_len = get_gap_length(epoch)
        gap_end = gap_start + gap_len
    else:
        gap_end = gap_start + 3  # Default gap length
    
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
    lat_std = max(df_known['Latitude'].std(), 1e-6)
    lon_mean = df_known['Longitude'].mean()
    lon_std = max(df_known['Longitude'].std(), 1e-6)
    
    # Modified target: Only position, no time
    target = torch.tensor([
        [
            (row["Latitude"] - lat_mean) / lat_std,
            (row["Longitude"] - lon_mean) / lon_std,
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

# Replace smooth_l1_loss with:
def haversine_loss(pred, target, norm_params):
    # Convert normalized predictions to actual coordinates
    pred_lat = pred[..., 0] * norm_params[1] + norm_params[0]
    pred_lon = pred[..., 1] * norm_params[3] + norm_params[2]
    target_lat = target[..., 0] * norm_params[1] + norm_params[0]
    target_lon = target[..., 1] * norm_params[3] + norm_params[2]
    
    # Calculate haversine distance in km
    R = 6371.0
    pred_lat, pred_lon, target_lat, target_lon = map(torch.deg2rad, 
                                                    [pred_lat, pred_lon, target_lat, target_lon])
    dlat = target_lat - pred_lat
    dlon = target_lon - pred_lon
    
    a = torch.sin(dlat/2)**2 + torch.cos(pred_lat) * torch.cos(target_lat) * torch.sin(dlon/2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
    return R * c.mean()


def main():
    # Load data
    df = load_data()
    grouped = df.groupby("MMSI")
    grouped_list = list(grouped)
    train_groups, val_groups = train_test_split(grouped_list, test_size=0.2, random_state=42)

    start_epoch = 0

    val_data_list = []
    for mmsi, ship_df in val_groups:
        data = process_ship(ship_df)
        if data is not None:
            val_data_list.append(data)

    val_loader = DataLoader(val_data_list, batch_size=16, shuffle=False)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Prepare a sample train_data_list for model initialization
    sample_data = None
    for mmsi, ship_df in train_groups:
        sample_data = process_ship(ship_df, epoch=start_epoch)
        if sample_data is not None:
            break
    if sample_data is None:
        raise RuntimeError("No valid training data found for model initialization.")

    # Create transformer model
    model = ShipTrajectoryTransformerModel(
        in_dim=sample_data.x.shape[1],
        hidden_dim=64,
        embedding_dim=64,
        gap_len=get_gap_length(start_epoch),
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=4,
        dropout=0.1
    ).to(device)

    # old parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5, betas=(0.9, 0.98))  # Tuned betas
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=50,  # Restart every 50 epochs
        T_mult=1,
        eta_min=1e-6
    ) 
    
    def warmup_lr(epoch, warmup_epochs=10, base_lr=1e-4):
        return base_lr * min(1.0, (epoch + 1) / warmup_epochs)
   

    # Setup training
    """ optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-5, betas=(0.9, 0.98))  # Tuned betas
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=100,  # Restart every 100 epochs
        T_mult=1,
        eta_min=1e-5
    )

    def warmup_lr(epoch, warmup_epochs=10, base_lr=5e-3):
        return base_lr * min(1.0, (epoch + 1) / warmup_epochs) """

    # Add gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

    model_save_path = 'newTransformerModelAdamW.pt'
    best_loss = float('inf')

    # Try to load existing checkpoint
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

    # Training loop
    for epoch in range(start_epoch, 1000):
        gap_len = get_gap_length(epoch)
        # Process training data each epoch (for curriculum learning)
        train_data_list = []
        for mmsi, ship_df in train_groups:
            data = process_ship(ship_df, epoch=epoch)
            if data is not None and data.y.shape[0] == gap_len:
                train_data_list.append(data)
        train_loader = DataLoader(train_data_list, batch_size=16, shuffle=True, drop_last=True)

        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            target = batch.y.to(device)

            output = output.view(-1, gap_len, 2)
            target = target.view(-1, gap_len, 2)

            loss = 0.7 * haversine_loss(output, target, batch.norm_params) + 0.3 * F.smooth_l1_loss(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0.0
        val_gap_len = gap_len  # Use the same gap_len for validation
        val_data_list = []
        for mmsi, ship_df in val_groups:
            data = process_ship(ship_df, epoch=epoch)
            if data is not None and data.y.shape[0] == val_gap_len:
                val_data_list.append(data)
        val_loader = DataLoader(val_data_list, batch_size=16, shuffle=False, drop_last=True)

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)
                target = batch.y.to(device)

                output = output.view(-1, val_gap_len, 2)
                target = target.view(-1, val_gap_len, 2)

                val_loss += 0.7 * haversine_loss(output, target, batch.norm_params) + 0.3 * F.smooth_l1_loss(output, target)

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

    # Evaluation
    val_grouped = dict(val_groups)
    evaluate_model(model, val_grouped.items(), device)


def evaluate_model(model, grouped, device):
    with torch.no_grad():
        ship_groups = list(grouped)
        random_ships = random.sample(ship_groups, min(50, len(ship_groups)))
        
        print("\nPredictions for 50 random ships:")
        print("-" * 50)
        
        total_error = 0.0
        total_mae_pos = 0.0
        total_haversine = 0.0
        num_valid_ships = 0

        metrics = {
            'errors': [],
            'haversine_distances': [],
            'lat_errors': [],
            'lon_errors': []
        }
        
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
                    # Calculate individual errors
                    lat_errors = (denorm_preds[:,0] - denorm_targets[:,0]).abs()
                    lon_errors = (denorm_preds[:,1] - denorm_targets[:,1]).abs()
                    
                    metrics['errors'].append(error)
                    metrics['haversine_distances'].append(mean_haversine_error)
                    metrics['lat_errors'].extend(lat_errors.tolist())
                    metrics['lon_errors'].extend(lon_errors.tolist())

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
            print("\nDetailed Metrics:")
            print(f"Median Error: {np.median(metrics['errors']):.4f}")
            print(f"90th Percentile Error: {np.percentile(metrics['errors'], 90):.4f}")
            print(f"Average Latitude Error: {np.mean(metrics['lat_errors']):.4f}")
            print(f"Average Longitude Error: {np.mean(metrics['lon_errors']):.4f}")
            print("=" * 50)
        else:
            print("\nNo valid ships found for evaluation.")


if __name__ == "__main__":
    main()