import time
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dynamic_graph_model import DynamicGraphModel

# Helper functions
def log_time(start_time, description, show_total_time=True, total_start_time=None):
    elapsed = time.time() - start_time
    print(f"Time for {description}: {elapsed:.2f} seconds")
    if show_total_time and total_start_time:
        total_elapsed = time.time() - total_start_time
        print(f"Total time so far: {total_elapsed:.2f} seconds")
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

def calculate_metrics(predictions, targets):
    mse = torch.mean((predictions - targets) ** 2)
    mae = torch.mean(torch.abs(predictions - targets))
    
    # Calculate distance error (Euclidean distance)
    dist_error = torch.sqrt(torch.sum((predictions - targets) ** 2, dim=1))
    mean_dist_error = torch.mean(dist_error)
    
    # Calculate direction error
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
            
            # Print coordinates for better analysis
            true_x, true_y = targets[0, 0].item(), targets[0, 1].item()
            pred_x, pred_y = predictions[0, 0].item(), predictions[0, 1].item()
            error_distance = torch.sqrt(((predictions[0] - targets[0]) ** 2).sum()).item()
            
            print(f"\nSample {i+1} Coordinates:")
            print(f"  Ground Truth: ({true_x:.6f}, {true_y:.6f})")
            print(f"  Prediction:   ({pred_x:.6f}, {pred_y:.6f})")
            print(f"  Error Distance: {error_distance:.6f}")
            
            # Plot the first example in batch
            axes[i].scatter(true_x, true_y, 
                           color='blue', marker='o', s=100, label='Ground Truth')
            axes[i].scatter(pred_x, pred_y, 
                           color='red', marker='x', s=100, label='Prediction')
            
            # Add coordinate annotations
            axes[i].annotate(f"({true_x:.4f}, {true_y:.4f})", 
                            (true_x, true_y), 
                            textcoords="offset points",
                            xytext=(5,5), 
                            ha='left')
            axes[i].annotate(f"({pred_x:.4f}, {pred_y:.4f})", 
                            (pred_x, pred_y), 
                            textcoords="offset points", 
                            xytext=(5,5), 
                            ha='left')
            
            # Plot the trajectory (past positions)
            past_positions = batch_positions[0, :-1].cpu()
            axes[i].plot(past_positions[:, 0], past_positions[:, 1], 'b-', alpha=0.5, label='Past Trajectory')
            axes[i].scatter(past_positions[:, 0], past_positions[:, 1], color='lightblue', s=50)
            
            # Add arrows to show direction
            axes[i].arrow(past_positions[-1, 0].item(), past_positions[-1, 1].item(),
                         true_x - past_positions[-1, 0].item(),
                         true_y - past_positions[-1, 1].item(),
                         head_width=0.01, head_length=0.02, fc='blue', ec='blue', alpha=0.5)
            
            axes[i].arrow(past_positions[-1, 0].item(), past_positions[-1, 1].item(),
                         pred_x - past_positions[-1, 0].item(),
                         pred_y - past_positions[-1, 1].item(),
                         head_width=0.01, head_length=0.02, fc='red', ec='red', alpha=0.5)
            
            axes[i].set_title(f'Sample {i+1} (Error: {error_distance:.6f})')
            handles, labels = axes[i].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))  # Remove duplicate labels
            axes[i].legend(by_label.values(), by_label.keys())
            axes[i].set_aspect('equal')  # Equal aspect ratio
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('prediction_visualization.png')
    
    # Display interactive figure
    plt.ion()  # Turn on interactive mode
    plt.show()
    
    print("\nInteractive plot opened. Close the plot window to continue.")
    
    # Keep the plot open until user closes it
    try:
        plt.fignum_exists(fig.number)  # This is just to keep the reference
        input("Press Enter to continue after closing the plot window...")
    except Exception:
        pass
    
    plt.close(fig)

def find_max_batch_size(model, test_dataset, device, start_batch_size=100, step_size=100, max_attempts=20):
    """Find the maximum batch size that fits in GPU memory."""
    model.eval()
    
    # Get a single sample for dimension info
    sample_data, sample_pos, sample_heading = test_dataset[0]
    
    # Define test function that mimics your actual processing
    def test_batch_size(batch_size):
        print(f"Testing batch size: {batch_size}")
        try:
            # Create batch
            batch_data = sample_data.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
            batch_pos = sample_pos.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
            batch_headings = sample_heading.unsqueeze(0).repeat(batch_size, 1).to(device)
            
            # Run forward pass
            with torch.no_grad():
                output = model(
                    x_seq=batch_data,
                    pos_seq=batch_pos,
                    headings=batch_headings[:, -1],
                    heading_threshold=45.0
                )
            
            # Clear memory
            del batch_data, batch_pos, batch_headings, output
            torch.cuda.empty_cache()
            return True
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                torch.cuda.empty_cache()
                print(f"  Batch size {batch_size} is too large. Error: {str(e)[:100]}...")
                return False
            else:
                print(f"  Unexpected error: {e}")
                raise e
    
    # Binary search approach
    batch_size = start_batch_size
    max_successful_batch_size = 0
    
    # First find a batch size that fails
    success = True
    attempts = 0
    
    while success and attempts < max_attempts:
        success = test_batch_size(batch_size)
        if success:
            max_successful_batch_size = batch_size
            batch_size += step_size
        attempts += 1
    
    if attempts >= max_attempts:
        print(f"Reached maximum attempts. Using batch size {max_successful_batch_size}")
        return max_successful_batch_size
    
    # Now refine with binary search
    upper_bound = batch_size
    lower_bound = max_successful_batch_size
    
    while upper_bound - lower_bound > step_size // 2:
        mid = (upper_bound + lower_bound) // 2
        if test_batch_size(mid):
            lower_bound = mid
            max_successful_batch_size = mid
        else:
            upper_bound = mid
    
    # Return a slightly conservative value (95% of max) to account for variations
    recommended_batch_size = int(max_successful_batch_size * 0.95)
    recommended_batch_size = max(recommended_batch_size, 1)  # Ensure at least batch size 1
    
    per_gpu_batch_size = recommended_batch_size // torch.cuda.device_count()
    print(f"\nRecommended batch size: {recommended_batch_size} (about {per_gpu_batch_size} per GPU)")
    
    return recommended_batch_size

def main():
    total_start_time = time.time()
    
    # Set device
    if torch.cuda.is_available():
        print(f"CUDA available. Found {torch.cuda.device_count()} GPUs.")
        device = 'cuda'  # Changed to use all GPUs
    else:
        print("CUDA not available. Using CPU.")
        device = 'cpu'
        return
    
    # Print GPU information
    print("GPU Information:")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    
    # Load test data
    print("Loading test dataset...")
    test_df = pd.read_csv('test_set.csv')
    print(f"Test dataset loaded with shape: {test_df.shape}")
    
    # Create test dataset
    sequence_length = 10
    test_dataset = ShipTrajectoryDataset(test_df, sequence_length)
    
    # Get input dimension from the first sample
    sample_batch = test_dataset[0][0]
    input_dim = sample_batch.size(1)
    print(f"Input dimension: {input_dim}")
    
    # Create model
    print("Creating model...")
    model = DynamicGraphModel(
        input_dim=input_dim, 
        hidden_dim=64,      
        fusion_dim=128,     
        max_radius=1.0
    )
    
    # Load saved model weights
    model_path = 'best_model5.pt'
    print(f"Loading model from {model_path}...")
    
    # Load the state dictionary
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Remove the "module." prefix if it exists
    if list(state_dict.keys())[0].startswith('module.'):
        print("Removing 'module.' prefix from state dict keys...")
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Fix key mapping for fusion module
    key_mapping = {
        "fusion_fc.weight": "fusion_module.fc.weight",
        "fusion_fc.bias": "fusion_module.fc.bias"
    }

    for old_key, new_key in key_mapping.items():
        if old_key in state_dict:
            print(f"Remapping {old_key} to {new_key}")
            state_dict[new_key] = state_dict.pop(old_key)

    model.load_state_dict(state_dict)
    
    # Enable multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for data parallel processing")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # Find optimal batch size
    print("\nFinding optimal batch size for your GPU configuration...")
    #batch_size = find_max_batch_size(model, test_dataset, device, start_batch_size=500, step_size=100)
    batch_size = 1600

    # Enable multi-processing data loading
    num_workers = min(8, os.cpu_count())
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    print(f"Test dataset created with {len(test_dataset)} samples")
    print(f"Using batch size {batch_size} with {num_workers} data loader workers")
    
    # Test evaluation
    print("Running test evaluation...")
    model.eval()
    test_loss = 0.0
    all_metrics = {}
    num_samples_processed = 0
    
    processing_start = time.time()
    with torch.no_grad():
        for batch_idx, (batch_data, batch_positions, batch_headings) in enumerate(test_loader):
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
            
            batch_size = batch_data.size(0)
            num_samples_processed += batch_size
            criterion = nn.MSELoss()
            loss = criterion(predictions, targets)
            test_loss += loss.item() * batch_size
            
            # Calculate additional metrics
            metrics = calculate_metrics(predictions, targets)
            for key, value in metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = 0.0
                all_metrics[key] += value * batch_size
                
            # Report progress
            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - processing_start
                samples_per_second = num_samples_processed / elapsed
                print(f"Processed {num_samples_processed} samples ({samples_per_second:.2f} samples/sec)")
    
    # Calculate average test loss and metrics
    avg_test_loss = test_loss / len(test_dataset)
    for key in all_metrics:
        all_metrics[key] /= len(test_dataset)
    
    # Display results
    print("\n=== Test Results ===")
    print(f"Test Loss: {avg_test_loss:.6f}")
    for key, value in all_metrics.items():
        print(f"Test {key}: {value:.6f}")
    
    # Visualize predictions
    print("Generating prediction visualizations...")
    # Use the first GPU for visualization to avoid any issues
    visualize_predictions(model.module if isinstance(model, nn.DataParallel) else model, 
                          DataLoader(test_dataset, batch_size=10, shuffle=False),
                          'cuda:0')
    
    total_time = time.time() - total_start_time
    print(f"\n=== Execution Summary ===")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Processed {len(test_dataset)} samples at {len(test_dataset)/total_time:.2f} samples/second")

if __name__ == "__main__":
    main()
