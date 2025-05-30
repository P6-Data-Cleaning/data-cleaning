import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from preproccess import AISTrajectoryDataset
import numpy as np
from math import cos, radians, sin, sqrt, atan2
from fastdtw import fastdtw

# Utility function: batched
def batched(seq, size):
    """Yield successive chunks from seq of length 'size'."""
    return [seq[i:i + size] for i in range(0, len(seq), size)]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    phi1, phi2 = radians(lat1), radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)
    a = sin(delta_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Used by the fastdtw library
def haversine_distance(p1, p2):
    return haversine(p1[0], p1[1], p2[0], p2[1])

def is_inside_bbox(latitudes, longitudes, bbox):
    return all(
        bbox["min_lat"] <= lat <= bbox["max_lat"] and
        bbox["min_lon"] <= lon <= bbox["max_lon"]
        for lat, lon in zip(latitudes, longitudes)
    )

class AISTransformer(nn.Module):
    def __init__(self, feature_dim=5, hidden_dim=64, num_heads=2, num_layers=1, dropout=0.3, max_len=60):
        super().__init__()
        self.encoder_embedding = nn.Linear(feature_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(0.01 * torch.randn(1, max_len, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)  # Add LayerNorm
        self.decoder_input_proj = nn.Linear(2, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        self.output_head = nn.Linear(hidden_dim, 2)

        # New LayerNorm layers for GRU inputs and outputs
        self.gru_input_norm = nn.LayerNorm(hidden_dim)
        self.gru_output_norm = nn.LayerNorm(hidden_dim)

        # Initialize positional encoding and encoder embedding
        self.positional_encoding.data = 0.01 * torch.randn_like(self.positional_encoding)
        nn.init.xavier_uniform_(self.encoder_embedding.weight)

    def forward(self, x, decoder_input=None, teacher_forcing=False, targets=None):
        # Encode input sequence
        encoder_input = self.encoder_embedding(x) + self.positional_encoding[:, :x.size(1), :]
        memory = self.encoder(encoder_input)
        memory = self.norm(memory)  # Normalize encoder output

        # Use the mean of encoder memory as context to initialize the GRU
        encoder_summary = memory.mean(dim=1).unsqueeze(0)  # (1, B, H)
        decoder_hidden = encoder_summary
        outputs = []

        # Start token: use last input position as initial decoder input
        if decoder_input is None:
            decoder_input = x[:, -1:, :2]  # (B, 1, 2)

        seq_len = targets.size(1) if targets is not None else 5  # Default to 5 if not given

        for t in range(seq_len):
            # Apply normalization in the forward method
            decoder_input_proj = self.gru_input_norm(self.decoder_input_proj(decoder_input))
            out, decoder_hidden = self.gru(decoder_input_proj, decoder_hidden)
            out = self.gru_output_norm(out)

            out = torch.clamp(out, -10, 10)  # Clamp GRU outputs
            step_output = self.output_head(out)
            step_output = torch.clamp(step_output, -5, 5)  # Clamp final predictions
            if not torch.isfinite(step_output).all():
                print("❌ GRU output is non-finite")
            step_output = torch.clamp(step_output, -5, 5)
            outputs.append(step_output)

            if teacher_forcing and targets is not None:
                decoder_input = targets[:, t:t+1, :]
            else:
                decoder_input = step_output.detach()

        outputs = torch.cat(outputs, dim=1)  # (B, seq_len, 2)
        return outputs

class HaversineDistanceLoss(nn.Module):
    def __init__(self, lat_std, lat_mean, lon_std, lon_mean, ref_lat=56.0, ref_lon=10.0):
        super().__init__()
        self.lat_std = lat_std
        self.lat_mean = lat_mean
        self.lon_std = lon_std
        self.lon_mean = lon_mean
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.R = 6371000  # Earth radius in meters
        
    def forward(self, preds, targets):
        if not torch.isfinite(preds).all():
            print("❌ preds contain non-finite values")
            print("Min:", torch.min(preds), "Max:", torch.max(preds))
            print("Any NaN:", torch.isnan(preds).any())
            print("Any Inf:", torch.isinf(preds).any())
        if not torch.isfinite(targets).all():
            print("❌ targets contain non-finite values")
            print("Min:", torch.min(targets), "Max:", torch.max(targets))
            print("Any NaN:", torch.isnan(targets).any())
            print("Any Inf:", torch.isinf(targets).any())
            raise ValueError("⚠️ Non-finite value in preds or targets before loss calculation.")

        # Clamp normalized predictions and targets to [-5, 5] (adjust as needed)
        preds = torch.clamp(preds, -5, 5)
        targets = torch.clamp(targets, -5, 5)

        # Denormalize predictions and targets for real-world coordinates
        pred_lat = preds[..., 0] * self.lat_std + self.lat_mean
        pred_lon = preds[..., 1] * self.lon_std + self.lon_mean
        true_lat = targets[..., 0] * self.lat_std + self.lat_mean
        true_lon = targets[..., 1] * self.lon_std + self.lon_mean
        
        # Calculate Haversine distance using torch tensor operations
        # Convert degrees to radians (math.pi / 180 = 0.017453292519943295)
        deg_to_rad = 0.017453292519943295
        
        phi1 = pred_lat * deg_to_rad
        phi2 = true_lat * deg_to_rad
        delta_phi = (true_lat - pred_lat) * deg_to_rad
        delta_lambda = (true_lon - pred_lon) * deg_to_rad
        
        a = torch.sin(delta_phi / 2) ** 2 + torch.cos(phi1) * torch.cos(phi2) * torch.sin(delta_lambda / 2) ** 2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a + 1e-8))
        distance = self.R * c
        
        # Return mean of finite values only
        return torch.mean(distance[torch.isfinite(distance)])

# Utility function to compute and log gradient norms
def log_gradient_norms(model, logger=None):
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)  # Compute L2 norm
            total_norm += param_norm.item() ** 2
            if logger:
                logger.info(f"Gradient norm for {name}: {param_norm:.4f}")
            else:
                print(f"Gradient norm for {name}: {param_norm:.4f}")
    total_norm = total_norm ** 0.5
    if logger:
        logger.info(f"Total gradient norm: {total_norm:.4f}")
    else:
        print(f"Total gradient norm: {total_norm:.4f}")

def main():
    # DDP setup
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    gpus = torch.cuda.device_count()
    print(f"🧠 Using {gpus} GPUs")

    # === Hyperparameters ===
    EPOCHS = 100
    BATCH_SIZE = 120 * gpus
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    RUN_TRAINING = False
    PATIENCE = 20 # Number of epochs with no improvement before stopping
    MIN_DELTA = 1e-4 # Minimum change to qualify as an improvement
    DECAY_LENGTH = 20  # how many epochs to decay over
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", "TransformerWithGRU_year_20_v3.pt")

    # === Training False strategy ===
    SCHEDULED_SAMPLING = True  # Use scheduled sampling during training (use less teacher forcing over time)

    # === Validation decoding strategy ===
    VAL_TEACHER_FORCING = False  # Use autoregressive decoding during validation if False

    # === Evaluation decoding strategy ===
    EVAL_TEACHER_FORCING = False  # Use autoregressive decoding during evaluation if False

    # === Dataset and DataLoader with DistributedSampler ===
    data = np.load("data/processed_data_year_splits_20_v2.npz", allow_pickle=True)

    train_sequences, val_sequences, eval_sequences, train_targets, val_targets, eval_targets, norm_params = (
        data["train_sequences"],
        data["val_sequences"],
        data["eval_sequences"],
        data["train_targets"],
        data["val_targets"],
        data["eval_targets"],
        data["norm_params"].item(),
    )

    if np.isnan(train_targets).any():
        print("⚠️ NaN detected in train_targets!")
    if np.isnan(val_targets).any():
        print("⚠️ NaN detected in val_targets!")
    if np.isnan(eval_targets).any():
        print("⚠️ NaN detected in eval_targets!")

    train_dataset = AISTrajectoryDataset(train_sequences, train_targets)
    val_dataset = AISTrajectoryDataset(val_sequences, val_targets)
    eval_dataset = AISTrajectoryDataset(eval_sequences, eval_targets)

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    eval_sampler = DistributedSampler(eval_dataset)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, pin_memory=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, pin_memory=True, num_workers=8)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, sampler=eval_sampler, pin_memory=True, num_workers=8)

    # === Model, optimizer, loss ===
    model = AISTransformer().to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=PATIENCE, factor=0.5)

    lat_std = norm_params['std']['Latitude']
    lat_mean = norm_params['mean']['Latitude']
    lon_std = norm_params['std']['Longitude']
    lon_mean = norm_params['mean']['Longitude']

    criterion = HaversineDistanceLoss(lat_std, lat_mean, lon_std, lon_mean).to(device)

    # === Resume from checkpoint if exists ===
    if os.path.exists(save_path):
        if local_rank == 0:
            print(f"🔄 Resuming training from checkpoint: {save_path}")
        model.module.load_state_dict(torch.load(save_path, map_location=device))

    # Add this diagnostic code to your main function to check data
    if local_rank == 0:
        print("Data diagnostics:")
        # Check the first few samples of training data
        for i in range(3):
            sample_input = train_sequences[i][0]  # First timestep of sequence i
            sample_target = train_targets[i][0]   # First target point of sequence i
            
            # Denormalize to inspect raw values
            input_lat = sample_input[0] * lat_std + lat_mean
            input_lon = sample_input[1] * lon_std + lon_mean
            target_lat = sample_target[0] * lat_std + lat_mean
            target_lon = sample_target[1] * lon_std + lon_mean
            
            print(f"Sample {i}: Input (lat={input_lat:.6f}, lon={input_lon:.6f}), "
                  f"Target (lat={target_lat:.6f}, lon={target_lon:.6f})")

    # === Training loop ===
    best_val_loss = float('inf')
    epoch_no_improvement = 0
    if RUN_TRAINING:
        print("Training started...")
        for epoch in range(EPOCHS):
            p_teacher_forcing = max(0.0, 1.0 - epoch / DECAY_LENGTH)
            model.train()
            train_sampler.set_epoch(epoch)
            total_loss = 0.0
            scaler = torch.cuda.amp.GradScaler()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = model(inputs, teacher_forcing=True, targets=targets)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            val_preds, val_targets = [], []
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    if VAL_TEACHER_FORCING:
                        outputs = model(inputs, teacher_forcing=True, targets=targets)
                    else:
                        outputs = model(inputs, teacher_forcing=False, targets=targets)

                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    if local_rank == 0:
                        val_preds.append(outputs.cpu().numpy())
                        val_targets.append(targets.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)

            if local_rank == 0:
                if (SCHEDULED_SAMPLING):
                    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}m | Val Loss: {avg_val_loss:.4f}m | Teacher Forcing prob: {p_teacher_forcing:.2f}")
                else:
                    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}m | Val Loss: {avg_val_loss:.4f}m")

                # Check for improvement
                if avg_val_loss < best_val_loss - MIN_DELTA:
                    best_val_loss = avg_val_loss
                    epoch_no_improvement = 0
                    torch.save(model.module.state_dict(), save_path)
                    print(f"✅ New best model saved to {save_path}")
                else:
                    epoch_no_improvement += 1
                    print(f"⏳ No improvement for {epoch_no_improvement} epoch(s)")

                # Early stopping condition
                if epoch_no_improvement >= PATIENCE:
                    print(f"🛑 Early stopping triggered after {PATIENCE} epochs without improvement.")
                    break

    # === Evaluation on a few ships (only on rank 0) ===
    if local_rank == 0:
        print("\n--- Evaluation started ---")
        model.module.load_state_dict(torch.load(save_path))
        model.eval()

        # Track metrics
        total_eval_loss = 0.0
        ship_metrics = {}
        all_pred_coords = []
        all_true_coords = []

        total_points = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(eval_loader):
                # Store MMSI values before removing them
                mmsi_values = inputs[:, 0, -1].cpu().numpy().astype(int)

                # Remove MMSI from inputs for model compatibility (only keep the first 5 features)
                model_inputs = inputs[:, :, :-1].to(device)
                targets = targets.to(device)

                if EVAL_TEACHER_FORCING:
                    outputs = model(model_inputs, teacher_forcing=True, targets=targets)
                else:
                    outputs = model(model_inputs, teacher_forcing=False, targets=targets)

                loss = criterion(outputs, targets)
                total_eval_loss += loss.item()

                # Denormalize predictions and targets for real-world coordinates
                pred_lat = outputs[..., 0].cpu().numpy() * lat_std + lat_mean
                pred_lon = outputs[..., 1].cpu().numpy() * lon_std + lon_mean
                true_lat = targets[..., 0].cpu().numpy() * lat_std + lat_mean
                true_lon = targets[..., 1].cpu().numpy() * lon_std + lon_mean

                # Denormalize inputs for better understanding
                input_lat = inputs[..., 0].cpu().numpy() * lat_std + lat_mean
                input_lon = inputs[..., 1].cpu().numpy() * lon_std + lon_mean
                

                # Store for later analysis
                all_pred_coords.extend(zip(pred_lat.flatten(), pred_lon.flatten()))
                all_true_coords.extend(zip(true_lat.flatten(), true_lon.flatten()))

                # Calculate error for each prediction in meters
                for i in range(len(mmsi_values)):
                    mmsi = int(mmsi_values[i])
                    errors = np.array([haversine(pred_lat[i][j], pred_lon[i][j], true_lat[i][j], true_lon[i][j])
                                        for j in range(len(pred_lat[i]))])
                    
                    gap_length = haversine(input_lat[i][19], input_lon[i][19], input_lat[i][40], input_lon[i][40])
                    pred_length = haversine(pred_lat[i][0], pred_lon[i][0], pred_lat[i][-1], pred_lon[i][-1])
                    true_length = haversine(true_lat[i][0], true_lon[i][0], true_lat[i][-1], true_lon[i][-1])
                    dtw_distance, _ = fastdtw(list(zip(pred_lat[i], pred_lon[i])), list(zip(true_lat[i], true_lon[i])), dist=haversine_distance)

                    # Add to specific threshold sets
                    if 'ships_by_threshold' not in locals():
                        ships_by_threshold = {
                            100: 0,  # 100 m
                            200: 0,  # 200 m
                            500: 0,  # 500 m
                            1000: 0,  # 1 km
                            2000: 0,  # 2 km
                            5000: 0,  # 5 km
                            10000: 0,  # 10 km
                            20000: 0,  # 20 km
                            50000: 0,  # 50 km
                            100000: 0  # 100 km
                        }
                    
                    for error in errors:
                        total_points += 1
                        for threshold in ships_by_threshold.keys():
                            if error > threshold:
                                ships_by_threshold[threshold] += 1
                    

                    if mmsi not in ship_metrics:
                        ship_metrics[mmsi] = {
                            'errors': [errors.tolist()],
                            'count': 1,
                            'input_sequence': [list(zip(input_lat[i], input_lon[i]))],
                            'pred_coor': [list(zip(pred_lat[i], pred_lon[i]))],
                            'true_coor': [list(zip(true_lat[i], true_lon[i]))],
                            'gap_length': [gap_length],
                            'pred_length': [pred_length],
                            'true_length': [true_length],
                            'dtw_distance': [dtw_distance]
                        }
                    else:
                        ship_metrics[mmsi]['errors'].append(errors.tolist())
                        ship_metrics[mmsi]['count'] += 1
                        ship_metrics[mmsi]['input_sequence'].append(list(zip(input_lat[i], input_lon[i])))
                        ship_metrics[mmsi]['pred_coor'].append(list(zip(pred_lat[i], pred_lon[i])))
                        ship_metrics[mmsi]['true_coor'].append(list(zip(true_lat[i], true_lon[i])))
                        ship_metrics[mmsi]['gap_length'].append(gap_length)
                        ship_metrics[mmsi]['pred_length'].append(pred_length)
                        ship_metrics[mmsi]['true_length'].append(true_length)
                        ship_metrics[mmsi]['dtw_distance'].append(dtw_distance)

                # Only print detailed info for a few batches to avoid flooding output
                if batch_idx < 3:
                    mmsi = int(mmsi_values[0])
                    print(f"\nShip MMSI: {mmsi}")
                    print(f"Batch loss: {loss.item():.4f}m")
                    print(f"Average error for first ship in batch: {np.mean(errors[0]):.2f}m")

                    # Convert both predictions and ground truth to lat/lon for better understanding
                    pred_lats, pred_lons = [], []
                    true_lats, true_lons = [], []

                    for plat, plon, tlat, tlon in zip(pred_lat[0], pred_lon[0], true_lat[0], true_lon[0]):
                        pred_lats.append(plat)
                        pred_lons.append(plon)
                        true_lats.append(tlat)
                        true_lons.append(tlon)

                    print("\nLatitude/Longitude Coordinates:")
                    print(f"Prediction points (Lat): {np.array(pred_lats)}")
                    print(f"Prediction points (Lon): {np.array(pred_lons)}")
                    print(f"True points (Lat): {np.array(true_lats)}")
                    print(f"True points (Lon): {np.array(true_lons)}")

                    # Calculate point-by-point Haversine distances for more accurate error
                    haversine_errors = [haversine(plat, plon, tlat, tlon)
                                        for plat, plon, tlat, tlon
                                        in zip(pred_lats, pred_lons, true_lats, true_lons)]

                    print(f"\nHaversine errors (meters): {np.array(haversine_errors)}")
                    print(f"Average Haversine error: {np.mean(haversine_errors):.2f}m")
        
        # Calculate and print overall metrics
        avg_eval_loss = total_eval_loss / len(eval_loader)
        print(f"\n--- Evaluation Results ---")
        print(f"Overall evaluation loss: {avg_eval_loss:.4f}m")
        
        # Calculate overall errors using Haversine
        all_errors = np.array([haversine(plat, plon, tlat, tlon) 
                                for (plat, plon), (tlat, tlon) in zip(all_pred_coords, all_true_coords)])

        print(f"Mean error: {np.mean(all_errors):.2f}m")
        print(f"Median error: {np.median(all_errors):.2f}m")
        print(f"Min error: {np.min(all_errors):.2f}m")
        print(f"Max error: {np.max(all_errors):.2f}m")

        # Calculate overall DTW distance
        all_dtw_distances = []
        for mmsi, data in ship_metrics.items():
            for seq_idx in range(len(data['pred_coor'])):
                pred_coords = data['pred_coor'][seq_idx]
                true_coords = data['true_coor'][seq_idx]
                dtw_distance, _ = fastdtw(pred_coords, true_coords, dist=haversine_distance)
                all_dtw_distances.append(dtw_distance)
        all_dtw_distances = np.array(all_dtw_distances)

        print(f"\nMean DTW distance: {np.mean(all_dtw_distances):.2f}m")
        print(f"Median DTW distance: {np.median(all_dtw_distances):.2f}m")
        print(f"Min DTW distance: {np.min(all_dtw_distances):.2f}m")
        print(f"Max DTW distance: {np.max(all_dtw_distances):.2f}m")

        # Calculate average error for each sequence
        sequence_errors = []
        for mmsi, data in ship_metrics.items():
            for seq_idx, seq_errors in enumerate(data['errors']):
                avg_seq_error = np.mean(seq_errors)

                sequence_errors.append({
                    'mmsi': mmsi,
                    'seq_idx': seq_idx,
                    'avg_error': avg_seq_error,
                    'seq_errors': seq_errors,
                    'pred_coords': data['pred_coor'][seq_idx],
                    'true_coords': data['true_coor'][seq_idx],
                    'input_sequence': data['input_sequence'][seq_idx],
                    'gap_length': data['gap_length'][seq_idx],
                    'pred_length': data['pred_length'][seq_idx],
                    'true_length': data['true_length'][seq_idx],
                    'dtw_distance': data['dtw_distance'][seq_idx]
                })

        # Sort sequences by abs(gap_length - pred_length) first, then by avg_error
        sorted_sequences = sorted(sequence_errors, key=lambda x: (x['avg_error']))

        ships_by_threshold_gaps = {
            100: 0,  # 100 m
            200: 0,  # 200 m
            500: 0,  # 500 m
            1000: 0,  # 1 km
            2000: 0,  # 2 km
            5000: 0,  # 5 km
            10000: 0,  # 10 km
            20000: 0,  # 20 km
            50000: 0,  # 50 km
            100000: 0  # 100 km
        }

        ships_by_threshold_dtw = {
            100: 0,  # 100 m
            200: 0,  # 200 m
            500: 0,  # 500 m
            1000: 0,  # 1 km
            2000: 0,  # 2 km
            5000: 0,  # 5 km
            10000: 0,  # 10 km
            20000: 0,  # 20 km
            50000: 0,  # 50 km
            100000: 0  # 100 km
        }

        for seq in sorted_sequences:
            dtw_distance = seq['dtw_distance']
            gap_length = seq['gap_length']

            for threshold in ships_by_threshold_dtw.keys():
                if dtw_distance > threshold:
                    ships_by_threshold_dtw[threshold] += 1
            for threshold in ships_by_threshold_gaps.keys():
                if gap_length > threshold:
                    ships_by_threshold_gaps[threshold] += 1

        # Define the bounding box for the Aalborg canal
        AALBORG_CANAL_BBOX = {
            "min_lat": 56.957090,
            "max_lat": 57.111658,
            "min_lon": 9.867279,
            "max_lon": 10.285714
        }

        KIEL_CANAL_BBOX = {
            "min_lat": 53.897604,
            "max_lat": 54.437330,
            "min_lon": 9.177850,
            "max_lon": 10.164003
        }

        best_sequences = [seq for seq in sorted_sequences if is_inside_bbox([coord[0] for coord in seq['true_coords']], [coord[1] for coord in seq['true_coords']], AALBORG_CANAL_BBOX)]
        
        ships_by_threshold_avg_error = {
            100: 0,  # 100 m
            200: 0,  # 200 m
            500: 0,  # 500 m
            1000: 0,  # 1 km
            2000: 0,  # 2 km
            5000: 0,  # 5 km
            10000: 0,  # 10 km
            20000: 0,  # 20 km
            50000: 0,  # 50 km
            100000: 0  # 100 km
        }
        ships_by_threshold_dtw_aalborg = {
            100: 0,  # 100 m
            200: 0,  # 200 m
            500: 0,  # 500 m
            1000: 0,  # 1 km
            2000: 0,  # 2 km
            5000: 0,  # 5 km
            10000: 0,  # 10 km
            20000: 0,  # 20 km
            50000: 0,  # 50 km
            100000: 0  # 100 km
        }

        total_seq_aalborg = len(best_sequences)
        total_points_aalborg = 0
        for seqeuence in best_sequences:
            dtw_distance = seqeuence['dtw_distance']
            seq_errors = seqeuence['seq_errors']

            for threshold in ships_by_threshold_dtw_aalborg.keys():
                if dtw_distance > threshold:
                    ships_by_threshold_dtw_aalborg[threshold] += 1

            for seq_error in seq_errors:
                total_points_aalborg += 1
                for threshold in ships_by_threshold_avg_error.keys():
                    if seq_error > threshold:
                        ships_by_threshold_avg_error[threshold] += 1
        
        best_sequences = best_sequences[:200]
        worst_sequences = sorted_sequences[-10:]



        # Iterate through input sequences and filter those inside the Aalborg canal
        """ print("\n--- Sequences inside Aalborg canal ---")
        for rank, seq in enumerate(filtered_sequences, start=1):
            # Extract true coordinates
            true_coords = seq['true_coords']
            true_lads = [coord[0] for coord in true_coords]  # Extract all latitudes
            true_lons = [coord[1] for coord in true_coords]  # Extract all longitudes
            # Check if the sequence is inside the bounding box
            if is_inside_bbox(true_lads, true_lons, AALBORG_CANAL_BBOX):
                print(f"Ship MMSI: {seq['mmsi']}, Sequence Index: {seq['seq_idx']}, Avg Error: {seq['avg_error']:.2f}m")
                
                gap_length = seq['gap_length']
                input_sequence = seq['input_sequence']
                dtw_distance = seq['dtw_distance']
                
                for idx, (pred, true) in enumerate(zip(seq['pred_coords'], seq['true_coords']), start=1):
                    print(f"  Point {idx}:")
                    print(f"    Predicted: Lat {pred[0]:.6f}, Lon {pred[1]:.6f}")
                    print(f"    True:      Lat {true[0]:.6f}, Lon {true[1]:.6f}")
                    print(f"  Error: {haversine(pred[0], pred[1], true[0], true[1]):.2f}m")
                
                print(f"  Distance from first to last point: {haversine(pred_coords[0][0], pred_coords[0][1], pred_coords[-1][0], pred_coords[-1][1]):.2f}m")
                print(f"  Distance from first to last true point: {haversine(true_coords[0][0], true_coords[0][1], true_coords[-1][0], true_coords[-1][1]):.2f}m")
                print(f"  Length of missing segment: {gap_length:.2f}m")
                print(f"  Input sequence: {input_sequence}")
                print(f"  DTW distance: {dtw_distance:.2f}m")
                print("-" * 50) """
        
        # Display the 10 best sequences
        print("\nTop 100 Best Predicted Sequences:")
        for rank, seq in enumerate(best_sequences, start=1):
            print(f"{rank}. MMSI: {seq['mmsi']}, Sequence Index: {seq['seq_idx']}, Avg Error: {seq['avg_error']:.2f}m")

        # Display the 10 worst sequences
        print("\nTop 10 Worst Predicted Sequences:")
        for rank, seq in enumerate(reversed(worst_sequences), start=1):
            print(f"{rank}. MMSI: {seq['mmsi']}, Sequence Index: {seq['seq_idx']}, Avg Error: {seq['avg_error']:.2f}m")

        # Detailed information for the best and worst sequences
        for title, sequences in [("Best Sequences", best_sequences), ("Worst Sequences", worst_sequences)]:
            print(f"\n--- Detailed Information for {title} ---")
            for rank, seq in enumerate(sequences, start=1):
                print(f"\nSequence Rank {rank}: MMSI {seq['mmsi']}, Sequence Index: {seq['seq_idx']}, Avg Error: {seq['avg_error']:.2f}m")
                
                true_coords = seq['true_coords']
                pred_coords = seq['pred_coords']
                gap_length = seq['gap_length']
                input_sequence = seq['input_sequence']
                dtw_distance = seq['dtw_distance']
                
                for idx, (pred, true) in enumerate(zip(pred_coords, true_coords), start=1):
                    print(f"  Point {idx}:")
                    print(f"    Predicted: Lat {pred[0]:.6f}, Lon {pred[1]:.6f}")
                    print(f"    True:      Lat {true[0]:.6f}, Lon {true[1]:.6f}")
                    print(f"  Error: {haversine(pred[0], pred[1], true[0], true[1]):.2f}m")
                
                print(f"  Distance from first to last point: {haversine(pred_coords[0][0], pred_coords[0][1], pred_coords[-1][0], pred_coords[-1][1]):.2f}m")
                print(f"  Distance from first to last true point: {haversine(true_coords[0][0], true_coords[0][1], true_coords[-1][0], true_coords[-1][1]):.2f}m")
                print(f"  Length of missing segment: {gap_length:.2f}m")
                print(f"  Input sequence: {input_sequence}")
                print(f"  DTW distance: {dtw_distance:.2f}m")
       
            

        # Print error statistics by threshold
        print("\nError statistics by threshold:")
        for threshold, count in ships_by_threshold.items():
            threshold_km = threshold / 1000
            percentage = (count / total_points) * 100 if total_points > 0 else 0
            print(f"Sequences with errors > {threshold_km} km: {count} points ({percentage:.1f}%)")
        
        print(f"\nTotal predicted coordinates: {total_points}")

        # Print gap statistics by threshold
        print("\nGap statistics by threshold:")
        for threshold, count in ships_by_threshold_gaps.items():
            threshold_km = threshold / 1000
            percentage = (count / len(sorted_sequences)) * 100 if len(sorted_sequences) > 0 else 0
            print(f"Sequences with gaps > {threshold_km} km: {count} points ({percentage:.1f}%)")

        # Print dtw statistics by threshold
        print("\nDTW statistics by threshold:")
        for threshold, count in ships_by_threshold_dtw.items():
            threshold_km = threshold / 1000
            percentage = (count / len(sorted_sequences)) * 100 if len(sorted_sequences) > 0 else 0
            print(f"Sequences with DTW > {threshold_km} km: {count} points ({percentage:.1f}%)")

        print(f"\nTotal predicted sequences: {len(sorted_sequences)}")

        # Print avg error statistics by threshold for Aalborg canal
        print("\nAvg error statistics by threshold for Aalborg canal:")
        for threshold, count in ships_by_threshold_avg_error.items():
            threshold_km = threshold / 1000
            percentage = (count / total_points_aalborg) * 100 if total_points_aalborg > 0 else 0
            print(f"Sequences with avg error > {threshold_km} km: {count} points ({percentage:.1f}%)")

        print(f"\nTotal predicted coordinates in Aalborg: {total_points_aalborg}")

        # Print dtw statistics by threshold for Aalborg canal
        print("\nDTW statistics by threshold for Aalborg canal:")
        for threshold, count in ships_by_threshold_dtw_aalborg.items():
            threshold_km = threshold / 1000
            percentage = (count / total_seq_aalborg) * 100 if total_seq_aalborg > 0 else 0
            print(f"Sequences with DTW > {threshold_km} km: {count} points ({percentage:.1f}%)")

        print(f"\nTotal predicted coordinates in Aalborg canal: {total_seq_aalborg}")

        # Save evaluation results
        np.save("eval_results.npy", {
            'ship_metrics': ship_metrics,
            'overall_metrics': {
                'mean_error': np.mean(all_errors),
                'median_error': np.median(all_errors),
                'min_error': np.min(all_errors),
                'max_error': np.max(all_errors)
            }
        })
        print("\nEvaluation results saved to eval_results.npy")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()