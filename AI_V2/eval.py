import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import pandas as pd
from math import cos, radians, sin, sqrt, atan2
import random

def latlon_to_xy(lat, lon, ref_lat=56.0, ref_lon=10.0):
    R = 6371000  # Earth radius in meters
    x = R * radians(lon - ref_lon) * cos(radians(ref_lat))
    y = R * radians(lat - ref_lat)
    return x, y

# Inverse of equirectangular projection: convert X, Y back to latitude and longitude
def xy_to_latlon(x, y, ref_lat=56.0, ref_lon=10.0):
    R = 6371000  # Earth radius in meters
    lat = y / R + ref_lat
    lon = x / (R * cos(radians(ref_lat))) + ref_lon
    return lat, lon

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    phi1, phi2 = radians(lat1), radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)
    a = sin(delta_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

class AISTransformerSeq2Seq(nn.Module):
    def __init__(self, feature_dim=5, hidden_dim=256, num_heads=8, num_layers=4, dropout=0.2, max_len=20):
        super().__init__()
        self.encoder_embedding = nn.Linear(feature_dim, hidden_dim)
        self.decoder_embedding = nn.Linear(2, hidden_dim)  # Assume decoder input is (X, Y)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_head = nn.Linear(hidden_dim, 2)

    def forward(self, x, decoder_input=None):
        # x: (batch_size, seq_len=20, feature_dim)
        batch_size = x.size(0)
        device = x.device

        encoder_input = self.encoder_embedding(x) + self.positional_encoding[:, :x.size(1), :]
        memory = self.encoder(encoder_input)

        if decoder_input is None:
            decoder_input = torch.zeros((batch_size, 5, 2), device=device)

        decoder_input_emb = self.decoder_embedding(decoder_input) + self.positional_encoding[:, :decoder_input.size(1), :]
        out = self.decoder(tgt=decoder_input_emb, memory=memory)
        pred = self.output_head(out)
        return pred

def main():
    # DDP setup
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    gpus = torch.cuda.device_count()
    print(f"🧠 Using {gpus} GPUs")

    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", "best_model2000epochB5.pt")

    # === Model ===
    model = AISTransformerSeq2Seq().to(device)
    model = DDP(model, device_ids=[local_rank])

    # === For preprocessing ===
    df = pd.read_csv("data/cleaned_data_feb_without_traj_reducer.csv", parse_dates=["# Timestamp"])
    df.sort_values(by=["MMSI", "# Timestamp"], inplace=True)
    # === Preprocessing to match the training data ===
    df["X"], df["Y"] = zip(*df.apply(lambda row: latlon_to_xy(row["Latitude"], row["Longitude"]), axis=1))
    x_mean, x_std = df["X"].mean(), df["X"].std()
    y_mean, y_std = df["Y"].mean(), df["Y"].std()
    df["DeltaTime"] = df.groupby("MMSI")["# Timestamp"].diff().dt.total_seconds()
    df["DeltaTime"] = df["DeltaTime"].fillna(0)
    
    # === Evaluation on a few ships (only on rank 0) ===
    if local_rank == 0:
        print("\n--- Evaluation started ---")
        model.module.load_state_dict(torch.load(save_path))
        model.eval()
        print("\n--- Evaluation on 3 random ships ---")

        # Get MMSIs with at least 20 data points
        valid_mmsis = df.groupby("MMSI").filter(lambda x: len(x) >= 20)['MMSI'].unique()

        # Filter passenger ships
        passenger_ships = df[df["MMSI"].isin(valid_mmsis) & (df["Ship type"] == "Passenger")]["MMSI"].unique()

        # Filter not passenger ships
        not_passenger_ships = df[df["MMSI"].isin(valid_mmsis) & (df["Ship type"] != "Passenger")]["MMSI"].unique()
        
        # Select x random MMSIs if possible from passenger ships
        print("\n--- Evaluation on 25 random ships (Passenger) ---")
        num_ships_to_evaluate = min(25, len(passenger_ships))
        if num_ships_to_evaluate > 0:
            random_mmsis = random.sample(list(passenger_ships), num_ships_to_evaluate)
        else:
            random_mmsis = []
            print("Warning: Not enough ships with sufficient data (>= 20 points) for evaluation.")
        avg_haversine_dist_overall = 0.0
        for mmsi in random_mmsis:
            group = df[df["MMSI"] == mmsi].reset_index(drop=True)
            # No need to check length again, already filtered
            sample = group.iloc[:20]
            feature_columns = ["X", "Y", "SOG", "COG", "DeltaTime"]
            original = sample[feature_columns].copy().values
            input_seq = original.copy()
            input_seq[10:15, 0:2] = 0  # Mask X/Y
            input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                prediction = model.module(input_tensor).squeeze(0).cpu().numpy()
            print(f"\nShip MMSI: {mmsi}")
            avg_haversine_dist = 0.0
            for i in range(5):
                true_vals = original[10 + i, 0:2]
                # Convert prediction from normalized XY to lat/lon
                pred_x = prediction[i][0] * x_std + x_mean
                pred_y = prediction[i][1] * y_std + y_mean
                pred_lat, pred_lon = xy_to_latlon(pred_x, pred_y)
                pred_vals = [pred_lat, pred_lon]
                # Convert true values from XY to lat/lon
                true_x = true_vals[0]
                true_y = true_vals[1]
                true_lat, true_lon = xy_to_latlon(true_x, true_y)
                true_vals_latlon = [true_lat, true_lon]
                # Compute Haversine distance
                haversine_dist = haversine(true_lat, true_lon, pred_lat, pred_lon)
                avg_haversine_dist += haversine_dist
                print(f"Step {10+i}: True: {true_vals_latlon}, Pred: {pred_vals}, Haversine Distance: {haversine_dist:.2f} meters")
            avg_haversine_dist /= 5
            print(f"Average Haversine Distance for MMSI {mmsi}: {avg_haversine_dist:.2f} meters")
            avg_haversine_dist_overall += avg_haversine_dist
        
        avg_haversine_dist_overall /= num_ships_to_evaluate
        print(f"\nOverall Average Haversine Distance for {num_ships_to_evaluate} ships: {avg_haversine_dist_overall:.2f} meters")
        
        print("\n--- Evaluation on 25 random ships (not passenger) ---")
        # Select x random MMSIs if possible from not passenger ships
        num_ships_to_evaluate = min(25, len(not_passenger_ships))
        if num_ships_to_evaluate > 0:
            random_mmsis = random.sample(list(not_passenger_ships), num_ships_to_evaluate)
        else:
            random_mmsis = []
            print("Warning: Not enough ships with sufficient data (>= 20 points) for evaluation.")
        avg_haversine_dist_overall = 0.0
        for mmsi in random_mmsis:
            group = df[df["MMSI"] == mmsi].reset_index(drop=True)
            # No need to check length again, already filtered
            sample = group.iloc[:20]
            feature_columns = ["X", "Y", "SOG", "COG", "DeltaTime"]
            original = sample[feature_columns].copy().values
            input_seq = original.copy()
            input_seq[10:15, 0:2] = 0  # Mask X/Y
            input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                prediction = model.module(input_tensor).squeeze(0).cpu().numpy()
            print(f"\nShip MMSI: {mmsi}")
            avg_haversine_dist = 0.0
            for i in range(5):
                true_vals = original[10 + i, 0:2]
                # Convert prediction from normalized XY to lat/lon
                pred_x = prediction[i][0] * x_std + x_mean
                pred_y = prediction[i][1] * y_std + y_mean
                pred_lat, pred_lon = xy_to_latlon(pred_x, pred_y)
                pred_vals = [pred_lat, pred_lon]
                # Convert true values from XY to lat/lon
                true_x = true_vals[0]
                true_y = true_vals[1]
                true_lat, true_lon = xy_to_latlon(true_x, true_y)
                true_vals_latlon = [true_lat, true_lon]
                # Compute Haversine distance
                haversine_dist = haversine(true_lat, true_lon, pred_lat, pred_lon)
                avg_haversine_dist += haversine_dist
                print(f"Step {10+i}: True: {true_vals_latlon}, Pred: {pred_vals}, Haversine Distance: {haversine_dist:.2f} meters")
            avg_haversine_dist /= 5
            print(f"Average Haversine Distance for MMSI {mmsi}: {avg_haversine_dist:.2f} meters")
            avg_haversine_dist_overall += avg_haversine_dist
        
        avg_haversine_dist_overall /= num_ships_to_evaluate
        print(f"\nOverall Average Haversine Distance for {num_ships_to_evaluate} ships: {avg_haversine_dist_overall:.2f} meters")

        print("\n--- Evaluation completed ---")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()