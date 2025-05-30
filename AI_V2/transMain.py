import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from math import radians, cos, sin, asin, sqrt
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from linformer import Linformer

# Utility: haversine distance
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

# Positional encoding
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.01)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Graph to sequence
class GraphToSequenceConverter(nn.Module):
    def __init__(self, node_feat, seq_feat, max_seq=500):
        super().__init__()
        self.proj = nn.Linear(node_feat, seq_feat)
        self.max_seq = max_seq
    def forward(self, x, edge_index, batch=None):
        x = self.proj(x)
        if batch is None:
            nodes = torch.unique(edge_index[0], sorted=True)[:self.max_seq]
            return x[nodes].unsqueeze(0)
        seqs = []
        for i in torch.unique(batch):
            mask = batch==i
            idxs = torch.nonzero(mask, as_tuple=False).squeeze(1)[:self.max_seq]
            seqs.append(x[idxs])
        return nn.utils.rnn.pad_sequence(seqs, batch_first=True)

# Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, in_ch, hidden_ch, nhead=8, layers=4, dropout=0.1, max_seq=500):
        super().__init__()
        self.pe = LearnablePositionalEncoding(hidden_ch, max_len=max_seq)
        self.g2s = GraphToSequenceConverter(in_ch, hidden_ch, max_seq)
        self.lin = Linformer(
            dim=hidden_ch, seq_len=max_seq, depth=layers,
            heads=nhead, k=256, one_kv_head=True, share_kv=True, dropout=dropout
        )
    def forward(self, x, edge_index, batch=None):
        seq = self.g2s(x, edge_index, batch)
        seq = self.pe(seq)
        return self.lin(seq)

# Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, gap_len, nhead=8, layers=4, dropout=0.1):
        super().__init__()
        self.gap = gap_len
        self.pe = LearnablePositionalEncoding(hidden_dim, max_len=gap_len)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*4,
            dropout=dropout, batch_first=True
        )
        self.dec = nn.TransformerDecoder(layer, num_layers=layers)
        self.query = nn.Parameter(torch.randn(1, gap_len, hidden_dim)*0.01)
        self.out = nn.Linear(hidden_dim, 2)


    def forward(self, h, batch, gap_idx=4):
        batch_size = h.size(0)
        tgt = self.query.expand(batch_size, -1, -1) # Shape: [batch_size, gap_len, hidden_dim]
        tgt = self.pe(tgt) # Add positional encoding to the target queries
        
        memory = h
      
        mask = nn.Transformer.generate_square_subsequent_mask(self.gap, device=h.device)
        out = self.dec(tgt, memory, tgt_mask=mask)
        return self.out(out)

# Full model
class ShipTrajectoryTransformerModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, ship_emb=4, nav_emb=4, gap_len=3,
                 nhead=8, enc_layers=2, dec_layers=2, dropout=0.1, max_seq=500):
        super().__init__()
        self.se = nn.Embedding(20, ship_emb)
        self.ne = nn.Embedding(20, nav_emb)
        self.enc = TransformerEncoder(in_dim+ship_emb+nav_emb, hidden_dim,
                                      nhead, enc_layers, dropout, max_seq)
        self.dec = TransformerDecoder(hidden_dim, gap_len, nhead, dec_layers, dropout)
    def forward(self, data):
        x, ei, b = data.x, data.edge_index, data.batch
        se = self.se(data.ship_type_idx)[b]
        ne = self.ne(data.nav_status_idx)[b]
        inp = torch.cat([x, se, ne], dim=1)
        emb = self.enc(inp, ei, b)
        return self.dec(emb, b)

# Data utils

def patchify(seq, patch_size):
    B,L,F = seq.shape
    seq = seq[:,:(L//patch_size)*patch_size].reshape(B,-1,patch_size,F)
    return seq.mean(2)


def load_data(path='data/cleaned_data_feb_done.csv'):
    df = pd.read_csv(path, parse_dates=['# Timestamp'])
    df['# Timestamp'] = pd.to_datetime(df['# Timestamp'])
    return df


def create_features(df):
    df = df.copy()
    df['ts'] = df['# Timestamp'].astype(int)//10**9
    df['dt'] = df['ts'].diff().fillna(0)
    df['dlat'] = df['Latitude'].diff().fillna(0)
    df['dlon'] = df['Longitude'].diff().fillna(0)
    hr = df['# Timestamp'].dt.hour+df['# Timestamp'].dt.minute/60
    df['sh'],df['ch'] = np.sin(2*np.pi*hr/24), np.cos(2*np.pi*hr/24)
    df['rot'] = df['ROT'].fillna(0)
    df['vel'] = np.sqrt(df['dlat']**2+df['dlon']**2)
    df['step']=np.arange(len(df))
    for c in ['Ship type','Navigational status']:
        if c in df: df[c+'_idx']=pd.factorize(df[c])[0]
    feats=['Latitude','Longitude','SOG','COG','dt','step','dlat','dlon','vel','sh','ch','Ship type_idx','Navigational status_idx']
    feats=[c for c in feats if c in df]
    arr=torch.tensor(df[feats].values,dtype=torch.float)
    arr=(arr-arr.mean(0))/(arr.std(0)+1e-8)
    return arr, df


def process_ship(df, fixed_gap_start=5, fixed_gap_len=3, patch_size=1, randomize_gap_start=False, min_gap_start=3, max_gap_start_offset=10):
    if df.empty or '# Timestamp' not in df.columns or 'MMSI' not in df.columns:
        print("Skipping empty or invalid DataFrame")
        return None


        # --- Gap Logic ---
    gap_len = fixed_gap_len # Keep gap_len fixed for now

    if randomize_gap_start:
        # Calculate sequence length *after* potential patching
        effective_len = len(df) // patch_size 
        
        # Ensure valid range for random start
        actual_min_start = min_gap_start
        actual_max_start = max(actual_min_start, effective_len - gap_len - max_gap_start_offset)

        if actual_min_start >= actual_max_start:
             # Fallback if sequence too short for randomization range
             gap_start_unpatched = fixed_gap_start * patch_size 
        else:
             # Randomly choose start index (in terms of *patched* sequence)
             gap_start_patched = random.randint(actual_min_start, actual_max_start)
             # Convert back to original DataFrame index (approximate start)
             gap_start_unpatched = gap_start_patched * patch_size 
    else:
        # Use fixed start index (in terms of *patched* sequence)
        gap_start_patched = fixed_gap_start
        gap_start_unpatched = gap_start_patched * patch_size

    # Ensure indices are within original DataFrame bounds
    gap_start_unpatched = max(0, min(gap_start_unpatched, len(df) - gap_len))
    gap_end_unpatched = gap_start_unpatched + gap_len
    # --- End Gap Logic ---


    if len(df) < gap_end_unpatched: # Check if sequence is long enough for the chosen gap
        # print(f"Skipping ship {df['MMSI'].iloc[0]}: Sequence too short for gap.")
        return None

    mmsi = df['MMSI'].iloc[0] if not df.empty else 'Unknown'
    
    # Split using unpatched indices
    known = pd.concat([df.iloc[:gap_start_unpatched], df.iloc[gap_end_unpatched:]]).reset_index(drop=True)
    miss = df.iloc[gap_start_unpatched:gap_end_unpatched]

    if known.empty or miss.empty: # Check if split resulted in empty parts
        # print(f"Skipping ship {mmsi} due to empty 'known' or 'miss' data after split.")
        return None
    
    x, known_processed = create_features(known)

    if known_processed.empty:
        # print(f"Skipping ship {mmsi} due to empty 'known_processed' data after feature creation.")
        return None

    # Apply patching *after* splitting if patch_size > 1 (currently 1)
    if patch_size > 1:
        x = x.unsqueeze(0); x = patchify(x, patch_size).squeeze(0)

    # --- Edge creation needs length of potentially patched 'x' ---
    num_nodes = len(x)
    if num_nodes < 2: # Need at least 2 nodes for edges
         # print(f"Skipping ship {mmsi}: Not enough nodes after processing/patching.")
         return None
    edges = torch.tensor([[i, i + 1] for i in range(num_nodes - 1)] + 
                         [[i + 1, i] for i in range(num_nodes - 1)], dtype=torch.long).t()
    # --- End Edge creation ---
    
    lm = known['Latitude'].mean() if not known['Latitude'].empty else 0
    ls = known['Latitude'].std() if not known['Latitude'].empty else 1
    ls = ls if ls > 1e-8 else 1 
    lnm = known['Longitude'].mean() if not known['Longitude'].empty else 0
    lns = known['Longitude'].std() if not known['Longitude'].empty else 1
    lns = lns if lns > 1e-8 else 1 

    # Ensure 'miss' has the expected length
    if len(miss) != gap_len:
         # print(f"Skipping ship {mmsi}: 'miss' length ({len(miss)}) != expected gap_len ({gap_len}).")
         return None
         
    y = torch.tensor([[(r.Latitude - lm) / ls, (r.Longitude - lnm) / lns] for r in miss.itertuples()], dtype=torch.float)

    ship_type_val = 0
    nav_status_val = 0

    if 'Ship type_idx' in known_processed.columns and not known_processed.empty:
        ship_type_val = known_processed['Ship type_idx'].iloc[0] 
    if 'Navigational status_idx' in known_processed.columns and not known_processed.empty:
        nav_status_val = known_processed['Navigational status_idx'].iloc[0]

    data = Data(x=x, edge_index=edges, y=y, original_y=y.clone(),
              norm_params=torch.tensor([lm, ls, lnm, lns]),
              ship_type_idx=torch.tensor(ship_type_val, dtype=torch.long),
              nav_status_idx=torch.tensor(nav_status_val, dtype=torch.long),
              mmsi=mmsi)
    return data

# Training & eval

def train(model,loader,opt,device):
    model.train(); total=0
    for batch in loader:
        batch=batch.to(device)
        pred=model(batch).view(-1,2)
        loss=F.smooth_l1_loss(pred,batch.y)
        opt.zero_grad();loss.backward();opt.step()
        total+=loss.item()
    return total

def evaluate(model,loader,device):
    model.eval(); tot=0
    with torch.no_grad():
        for batch in loader:
            batch=batch.to(device)
            pred=model(batch).view(-1,2)
            tot+=F.smooth_l1_loss(pred,batch.y).item()
    return tot


def evaluate_detailed(model, val_data_list, device, num_samples=50):
    """Evaluates the model on a sample of validation data with detailed output."""
    model.eval()
    print("\n" + "="*50)
    print(f"Detailed Evaluation on {min(num_samples, len(val_data_list))} Random Validation Ships")
    print("="*50 + "\n")

    if not val_data_list:
        print("Validation data list is empty.")
        return

    # Ensure num_samples is not larger than the available data
    num_samples = min(num_samples, len(val_data_list))
    
    # Get random indices
    sample_indices = random.sample(range(len(val_data_list)), num_samples)

    total_mae_norm = 0
    total_mae_pos = 0
    total_mean_haversine = 0
    count = 0

    with torch.no_grad():
        for i in sample_indices:
            data = val_data_list[i]
            if data is None: continue # Skip if data processing failed for this ship

            data = data.to(device)
            
            # Need to add a batch dimension for the model
            batch_data = Batch.from_data_list([data]) 

            try:
                pred_norm = model(batch_data) # Shape: [1, gap_len, 2]
                pred_norm = pred_norm.squeeze(0) # Shape: [gap_len, 2]
            except Exception as e:
                print(f"Error during prediction for ship index {i} (MMSI: {getattr(data, 'mmsi', 'Unknown')}): {e}")
                continue

            true_norm = data.y # Shape: [gap_len, 2]

            # Ensure shapes match before loss calculation
            if pred_norm.shape != true_norm.shape:
                print(f"Shape mismatch for ship index {i} (MMSI: {getattr(data, 'mmsi', 'Unknown')}): Pred {pred_norm.shape}, True {true_norm.shape}")
                continue
                
            # --- Denormalization ---
            lm, ls, lnm, lns = data.norm_params.cpu().numpy()

            pred_lat = pred_norm[:, 0].cpu() * ls + lm
            pred_lon = pred_norm[:, 1].cpu() * lns + lnm

            # Use original_y for true denormalized values
            true_lat = data.original_y[:, 0].cpu() * ls + lm
            true_lon = data.original_y[:, 1].cpu() * lns + lnm
            # --- End Denormalization ---

            # --- Calculate Metrics ---
            mae_norm = F.l1_loss(pred_norm, true_norm).item()
            mae_pos = (torch.abs(pred_lat - true_lat) + torch.abs(pred_lon - true_lon)).mean().item() / 2
            
            distances = []
            for j in range(len(true_lat)):
                 # Ensure tensor elements are converted to floats for haversine
                 dist = haversine(true_lat[j].item(), true_lon[j].item(), 
                                  pred_lat[j].item(), pred_lon[j].item())
                 distances.append(dist)
            
            mean_dist = np.mean(distances) if distances else 0
            # --- End Calculate Metrics ---

            # --- Update Totals ---
            total_mae_norm += mae_norm
            total_mae_pos += mae_pos
            total_mean_haversine += mean_dist
            count += 1
            # --- End Update Totals ---

            # --- Print Output ---
            mmsi = getattr(data, 'mmsi', 'Unknown') # Safely get MMSI
            print(f"Ship {count} (MMSI: {mmsi}):")
            for step in range(len(pred_lat)):
                print(f"  Step {step+1} — Predicted Lat: {pred_lat[step]:.4f}, Lon: {pred_lon[step]:.4f}")
            
            print("  True values:")
            for step in range(len(true_lat)):
                 print(f"  Step {step+1} — True Lat: {true_lat[step]:.4f}, Lon: {true_lon[step]:.4f}")

            print(f"  Average Error (Norm MAE): {mae_norm:.4f}")
            print(f"  MAE Position (Denorm): {mae_pos:.4f}")
            print(f"  Mean Haversine Distance: {mean_dist:.2f} km")
            print("-" * 50)
            # --- End Print Output ---

    # --- Print Overall Averages ---
    if count > 0:
        print("\n" + "="*50)
        print("Overall Average Metrics for Sample:")
        print(f"  Average Normalized MAE: {total_mae_norm / count:.4f}")
        print(f"  Average Positional MAE (Denormalized): {total_mae_pos / count:.4f}")
        print(f"  Average Mean Haversine Distance: {total_mean_haversine / count:.2f} km")
        print("="*50)
    else:
        print("No samples were successfully evaluated.")


def main():
    df=load_data()
    groups=list(df.groupby('MMSI'))
    tr_groups,val_groups=train_test_split(groups,test_size=0.2,random_state=42)
    
    # Process data - store in lists
    print("Processing training data...")
    train_data=[process_ship(g[1], patch_size=1, randomize_gap_start=True, 
                             min_gap_start=3, max_gap_start_offset=10, fixed_gap_len=3) 
                for g in tr_groups]
    train_data = [d for d in train_data if d is not None] 
    print(f"Processed {len(train_data)} training samples.")

    print("Processing validation data...")
    val_data=[process_ship(g[1], patch_size=1, randomize_gap_start=False, 
                           fixed_gap_start=5, fixed_gap_len=3) 
              for g in val_groups]
    val_data = [d for d in val_data if d is not None]
    print(f"Processed {len(val_data)} validation samples.")

    if not train_data:
        print("No valid training data processed. Exiting.")
        return
    if not val_data:
        print("No valid validation data processed. Exiting.")
        return

    train_loader=DataLoader(train_data,batch_size=16,shuffle=True)
    val_loader=DataLoader(val_data,batch_size=16,shuffle=False)
    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Determine input dimension AFTER processing and potential patching
    in_dim=train_data[0].x.size(1) 
    print(f"Input feature dimension: {in_dim}")

    # Define model architecture (ensure parameters match saved model)
    model=ShipTrajectoryTransformerModel(
        in_dim=in_dim, hidden_dim=128, # Increased from 64
        ship_emb=4, nav_emb=4, gap_len=3, # gap_len must match process_ship fixed_gap_len
        nhead=8, 
        enc_layers=4, # Increased from 2
        dec_layers=4, # Increased from 2
        dropout=0.1, max_seq=500 # Adjust max_seq if needed without patching
    ).to(device)
    
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
    best_val_loss=float('inf')
    best_model_path = 'JakeTransMain.pt'

    print("Starting training...")
    for e in range(100): # Use the selected number of epochs
        tr_loss=train(model,train_loader,opt,device)
        val_loss=evaluate(model,val_loader,device)
        print(f"Epoch {e}: Train Loss={tr_loss:.4f}, Val Loss={val_loss:.4f}")
        if val_loss<best_val_loss:
            best_val_loss=val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Saved new best model to {best_model_path} (Val Loss: {best_val_loss:.4f})")
    print("Training finished.")

    # --- Add Detailed Evaluation ---
    print(f"\nLoading best model from {best_model_path} for detailed evaluation...")
    # Re-initialize model architecture before loading state_dict
    model_eval = ShipTrajectoryTransformerModel(
        in_dim=in_dim, hidden_dim=128, ship_emb=4, nav_emb=4, gap_len=3,
        nhead=8, enc_layers=4, dec_layers=4, dropout=0.1, max_seq=500
    ).to(device)
    try:
        model_eval.load_state_dict(torch.load(best_model_path, map_location=device))
        evaluate_detailed(model_eval, val_data, device, num_samples=50) # Evaluate 50 random ships
    except FileNotFoundError:
        print(f"Error: Best model file '{best_model_path}' not found. Skipping detailed evaluation.")
    except Exception as load_err:
         print(f"Error loading model state_dict: {load_err}. Skipping detailed evaluation.")
    # --- End Detailed Evaluation ---


if __name__=='__main__':
    main()
