import time
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from normalization import normalization
from splitting import splitting
from dynamic_graph_model import DynamicGraphModel

# Add this function to display timing information consistently
def log_time(start_time, description, show_total_time=True, total_start_time=None):
    elapsed = time.time() - start_time
    print(f"Time for {description}: {elapsed:.2f} seconds", flush=True)
    if show_total_time and total_start_time:
        total_elapsed = time.time() - total_start_time
        print(f"Total time so far: {total_elapsed:.2f} seconds", flush=True)
    return time.time()

class ShipTrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, df, sequence_length=10):
        self.df = df
        self.sequence_length = sequence_length
        
        # Precompute sequence indices for faster access
        self.mmsi_groups = list(df.groupby('mmsi'))
        self.sequence_indices = []
        
        for mmsi, group in self.mmsi_groups:
            group = group.sort_values('timestamp')
            total_rows = len(group)
            for i in range(0, total_rows - sequence_length + 1):
                self.sequence_indices.append((mmsi, i))
    
    def __len__(self):
        return len(self.sequence_indices)
    
    def __getitem__(self, idx):
        mmsi, start_idx = self.sequence_indices[idx]
        # Find the group with this mmsi
        for group_mmsi, group_df in self.mmsi_groups:
            if group_mmsi == mmsi:
                group = group_df
                break
                
        group = group.sort_values('timestamp')
        seq = group.iloc[start_idx:start_idx+self.sequence_length]
        seq_array = seq.values
        
        # Handle NaN values
        if np.isnan(seq_array).any():
            seq_array = np.nan_to_num(seq_array, nan=0.0)
        
        # Convert to tensor (keep on CPU)
        seq_tensor = torch.tensor(seq_array, dtype=torch.float32)
        
        # Extract positions and headings
        positions = seq_tensor[:, [3, 2]]  # [longitude, latitude]
        headings = seq_tensor[:, 6]
        
        return seq_tensor, positions, headings

def process_data_into_sequences(df, sequence_length=10):
    """Convert flat dataframe into sequences grouped by ship."""
    sequences = []
    mmsi_groups = df.groupby('mmsi')
    
    for mmsi, group in mmsi_groups:
        # Sort by timestamp
        group = group.sort_values('timestamp')
        
        # Create sequences of specified length
        total_rows = len(group)
        for i in range(0, total_rows - sequence_length + 1):
            seq = group.iloc[i:i+sequence_length]
            if len(seq) == sequence_length:  # Ensure complete sequence
                sequences.append(seq.values)
    
    # Stack all sequences into a 3D array
    if sequences:
        return np.stack(sequences)
    return np.array([])

def to_tensor(df, sequence_length=10):
    """Convert dataframe to tensor with sequences."""
    # Process into sequences
    sequences = process_data_into_sequences(df, sequence_length)
    
    if np.isnan(sequences).any():
        print("Warning: NaN values found in the data. Filling with 0.", flush=True)
        sequences = np.nan_to_num(sequences, nan=0.0)
    
    return torch.tensor(sequences, dtype=torch.float32).to('cuda')

def extract_positions(data):
    # For 3D data with shape [batch, sequence, features]
    # Extract longitude and latitude (indices 3 and 2)
    positions = data[:, :, [3, 2]]  # [longitude, latitude]
    return positions

def extract_headings(data):
    # Extract rate of turn (index 6)
    headings = data[:, :, 6]
    return headings

def calculate_metrics(predictions, targets):
    mse = torch.mean((predictions - targets) ** 2)
    mae = torch.mean(torch.abs(predictions - targets))
    
    # Calculate distance error (Euclidean distance)
    dist_error = torch.sqrt(torch.sum((predictions - targets) ** 2, dim=1))
    mean_dist_error = torch.mean(dist_error)
    
    # Calculate direction error
    # This is a simplified version and can be improved
    pred_dirs = torch.atan2(predictions[:, 1], predictions[:, 0])
    target_dirs = torch.atan2(targets[:, 1], targets[:, 0])
    dir_error = torch.abs(pred_dirs - target_dirs)
    # Handle circular wrap-around
    dir_error = torch.min(dir_error, 2 * torch.pi - dir_error)
    mean_dir_error = torch.mean(dir_error)
    
    return {
        "MSE": mse.item(),
        "MAE": mae.item(),
        "Mean Distance Error": mean_dist_error.item(),
        "Mean Direction Error": mean_dir_error.item()
    }

def visualize_predictions(model, test_loader, device, num_samples=5):
    """Visualize model predictions against ground truth."""
    model.eval()
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 15))
    
    with torch.no_grad():
        for i, (batch_data, batch_positions, batch_headings) in enumerate(test_loader):
            if i >= num_samples:
                break
                
            batch_data = batch_data.to(device)
            batch_positions = batch_positions.to(device)
            batch_headings = batch_headings.to(device)
            
            headings = batch_headings[:, -1]
            
            output = model(
                x_seq=batch_data,
                pos_seq=batch_positions,
                headings=headings,
                heading_threshold=45.0
            )
            
            predictions = output['predictions']
            targets = batch_positions[:, -1, :]
            
            # Plot the first example in batch
            axes[i].scatter(targets[0, 0].cpu(), targets[0, 1].cpu(), 
                           color='blue', label='Ground Truth')
            axes[i].scatter(predictions[0, 0].cpu(), predictions[0, 1].cpu(), 
                           color='red', label='Prediction')
            axes[i].set_title(f'Sample {i+1}')
            axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('prediction_visualization5.png')
    plt.close()

def main():
    total_start_time = time.time()
    stage_start_time = total_start_time
    
    if torch.cuda.is_available():
        print(f"CUDA available. Found {torch.cuda.device_count()} GPUs.", flush=True)
        device = 'cuda:0'  # Primary GPU
        
        # Print GPU info
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_properties(i).name}", flush=True)
    else: 
        print("CUDA not available")
        return
    
    stage_start_time = log_time(stage_start_time, "GPU initialization", True, total_start_time)

    # Load or prepare data
    if (os.path.exists('training_set.csv') and os.path.exists('validation_set.csv') and os.path.exists('test_set.csv')):
        print("Splitting already done, reading from files", flush=True)
        
        data_load_start = time.time()
        training_df = pd.read_csv('training_set.csv')
        validation_df = pd.read_csv('validation_set.csv')
        test_df = pd.read_csv('test_set.csv')
        stage_start_time = log_time(data_load_start, "loading existing datasets", True, total_start_time)
    else:
        print("Creating new training, validation and test sets", flush=True)
        
        normalization_start = time.time()
        df, scaler = normalization('cleaned_data_reduced_all.csv')
        stage_start_time = log_time(normalization_start, "normalization", True, total_start_time)

        splitting_start = time.time()
        training_df, validation_df, test_df = splitting(df, scaler)
        
        # Save the datasets
        save_start = time.time()
        training_df.to_csv('training_set.csv', index=False)
        validation_df.to_csv('validation_set.csv', index=False)
        test_df.to_csv('test_set.csv', index=False)
        log_time(save_start, "saving datasets", False)
        
        stage_start_time = log_time(splitting_start, "splitting data", True, total_start_time)

    print("Training dataframe shape:", training_df.shape, flush=True)
    print("Training dataframe columns:", training_df.columns, flush=True)
    print("First few rows:", training_df.head(), flush=True)

    # Set sequence length
    sequence_length = 10  # Adjust based on your needs
    
    # Create datasets
    print("Creating datasets...", flush=True)
    dataset_start = time.time()
    train_dataset = ShipTrajectoryDataset(training_df, sequence_length)
    val_dataset = ShipTrajectoryDataset(validation_df, sequence_length)
    test_dataset = ShipTrajectoryDataset(test_df, sequence_length)
    
    stage_start_time = log_time(dataset_start, "dataset creation", True, total_start_time)
    print(f"Created datasets with {len(train_dataset)} training samples, {len(val_dataset)} validation samples, and {len(test_dataset)} test samples", flush=True)
    
    # Print GPU memory info
    print("\nGPU Memory Before Training:", flush=True)
    for i in range(torch.cuda.device_count()):
        total_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        allocated_mem = torch.cuda.memory_allocated(i) / 1e9
        free_mem = total_mem - allocated_mem
        print(f"GPU {i}: {free_mem:.2f} GB free of {total_mem:.2f} GB total", flush=True)
    
    # Use a small batch size per GPU
    num_gpus = torch.cuda.device_count()
    batch_size_per_gpu = 1600
    total_batch_size = batch_size_per_gpu * num_gpus
    
    print(f"Using batch size {batch_size_per_gpu} per GPU, total batch size: {total_batch_size}", flush=True)
    
    dataloader_start = time.time()
    train_loader = DataLoader(
        train_dataset, 
        batch_size=total_batch_size, 
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=total_batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=total_batch_size, num_workers=0)
    
    stage_start_time = log_time(dataloader_start, "DataLoader creation", True, total_start_time)
    
    # Get a sample batch to determine input dimensions
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch[0].size(2)
    
    # Create the model on the primary GPU first
    model_creation_start = time.time()
    model = DynamicGraphModel(
        input_dim=input_dim, 
        hidden_dim=64,      
        fusion_dim=128,     
        max_radius=1.0
    ).to(device)
    
    # Enable Multi-GPU 
    if num_gpus > 1:
        print(f"Using DataParallel with {num_gpus} GPUs", flush=True)
        model = nn.DataParallel(model)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    stage_start_time = log_time(model_creation_start, "model initialization", True, total_start_time)

    # Training parameters
    num_epochs = 4
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    # Lists to track metrics
    train_losses = []
    val_losses = []
    
    # Training loop
    print("\n=== Starting Training ===", flush=True)
    training_start = time.time()
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        
        batch_time_start = time.time()
        for batch_idx, (batch_data, batch_positions, batch_headings) in enumerate(train_loader):
            # Move data to GPU
            batch_data = batch_data.to(device)
            batch_positions = batch_positions.to(device)
            batch_headings = batch_headings.to(device)
            
            # Get the latest heading for each sequence
            headings = batch_headings[:, -1]
            
            # Clear gradients
            optimizer.zero_grad()
            
            try:
                # Forward pass - with DataParallel now
                output = model(
                    x_seq=batch_data,
                    pos_seq=batch_positions,
                    headings=headings,
                    heading_threshold=45.0
                )
                
                # Get predictions
                predictions = output['predictions']
                
                # Target is the next position
                targets = batch_positions[:, -1, :]
                
                # Compute loss
                loss = criterion(predictions, targets)
                
                # Backpropagation
                loss.backward()
                
                # Update weights
                optimizer.step()
                
            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {e}", flush=True)
                # Emergency cleanup
                torch.cuda.empty_cache()
                continue
            
            # Add explicit memory cleanup every few batches
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
            
            epoch_loss += loss.item() * batch_data.size(0)
            
            # Print progress and memory usage
            if batch_idx % 20 == 0:
                batch_time = time.time() - batch_time_start
                print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}, Batch time: {batch_time:.2f}s", flush=True)
                # Print current GPU memory usage
                for i in range(torch.cuda.device_count()):
                    used_mem = torch.cuda.memory_allocated(i) / 1e9
                    print(f"  GPU {i}: {used_mem:.2f} GB used", flush=True)
                batch_time_start = time.time()
        
        # Calculate average training loss
        avg_train_loss = epoch_loss / len(train_dataset)
        train_losses.append(avg_train_loss)
        
        # Validation
        validation_start = time.time()
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_data, batch_positions, batch_headings in val_loader:
                batch_data = batch_data.to(device)
                batch_positions = batch_positions.to(device)
                batch_headings = batch_headings.to(device)
                
                # Get the latest heading for each sequence
                headings = batch_headings[:, -1]
                
                output = model(
                    x_seq=batch_data,
                    pos_seq=batch_positions,
                    headings=headings,
                    heading_threshold=45.0
                )
                
                predictions = output['predictions']
                targets = batch_positions[:, -1, :]
                
                loss = criterion(predictions, targets)
                val_loss += loss.item() * batch_data.size(0)
        
        validation_time = time.time() - validation_start
        
        # More memory cleanup
        torch.cuda.empty_cache()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_dataset)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Print progress
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | "
              f"Time: {epoch_time:.2f}s (Training: {epoch_time - validation_time:.2f}s, Validation: {validation_time:.2f}s)", flush=True)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_start = time.time()
            torch.save(model.state_dict(), 'best_model5.pt')
            save_time = time.time() - save_start
            print(f"Best model saved with validation loss: {best_val_loss:.6f} (Save time: {save_time:.2f}s)", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {patience} epochs without improvement", flush=True)
                break
    
    total_training_time = time.time() - training_start
    print(f"\n=== Training Complete (total time: {total_training_time:.2f}s) ===", flush=True)
    print(f"Best validation loss: {best_val_loss:.6f}", flush=True)
    
    # Plot training curve
    plot_start = time.time()
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_curve5.png')
    plt.close()
    stage_start_time = log_time(plot_start, "plotting loss curves", True, total_start_time)
    
    # Load best model for testing
    test_start = time.time()
    model.load_state_dict(torch.load('best_model5.pt'))
    
    # Test evaluation
    model.eval()
    test_loss = 0.0
    all_metrics = {}
    
    with torch.no_grad():
        for batch_data, batch_positions, batch_headings in test_loader:
            batch_data = batch_data.to(device)
            batch_positions = batch_positions.to(device)
            batch_headings = batch_headings.to(device)
            
            sequence_length = batch_data.size(1)
            pos_seq = batch_positions
            
            headings = batch_headings
            
            output = model(
                x_seq=batch_data,
                pos_seq=pos_seq,
                headings=headings,
                heading_threshold=45.0
            )
            
            predictions = output['predictions']
            targets = batch_positions[:, -1, :]
            
            loss = criterion(predictions, targets)
            test_loss += loss.item() * batch_data.size(0)
            
            # Calculate additional metrics
            metrics = calculate_metrics(predictions, targets)
            for key, value in metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = 0.0
                all_metrics[key] += value * batch_data.size(0)
    
    stage_start_time = log_time(test_start, "model testing", True, total_start_time)
    
    # Calculate average test loss and metrics
    avg_test_loss = test_loss / len(test_dataset)
    for key in all_metrics:
        all_metrics[key] /= len(test_dataset)
    
    print("\n=== Test Results ===", flush=True)
    print(f"Test Loss: {avg_test_loss:.6f}", flush=True)
    for key, value in all_metrics.items():
        print(f"Test {key}: {value:.6f}", flush=True)
    
    # Visualize predictions
    vis_start = time.time()
    visualize_predictions(model, test_loader, device)
    stage_start_time = log_time(vis_start, "prediction visualization", True, total_start_time)
    
    total_time = time.time() - total_start_time
    print(f"\n=== Execution Summary ===", flush=True)
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)", flush=True)

if __name__ == "__main__":
    main()