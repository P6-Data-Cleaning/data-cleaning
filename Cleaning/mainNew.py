import time
import os
import pandas as pd
import sys
from clusterDetection import detect_and_remove_anomalies
from trajectoryReducer import trajectory_reducer

def mainNew():
    start_time = time.time()
    
    # Make sure output directories exist
    os.makedirs('outputs', exist_ok=True)
    
    # Define file paths for the pipeline
    input_file = 'outputs/cleaned_data.csv'
    reduced_file = 'outputs/trajectory_reduced.csv'
    final_file = 'outputs/cluster_detection_filtered.csv'
    
    # Parameters for cluster detection
    density_threshold = 10
    radius_km = 1.0
    
    # Step 1: Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found!")
        return
    
    # Step 2: Run trajectory reducer
    print(f"\n=== STEP 1: TRAJECTORY REDUCTION ===")
    print(f"Processing file: {input_file}")
    
    # Read the CSV file
    print("Reading input file...")
    try:
        df = pd.read_csv(input_file)
        print(f"Read {len(df)} rows from input file")
        
        # Apply trajectory reduction
        print("Applying trajectory reduction...")
        reduced_df = trajectory_reducer(df)
        
        # Save reduced trajectory to CSV
        print(f"Saving reduced trajectory to {reduced_file}...")
        reduced_df.to_csv(reduced_file, index=False)
        
        print(f"Trajectory reduction completed. Original: {len(df)} rows, Reduced: {len(reduced_df)} rows")
        percent_reduction = ((len(df) - len(reduced_df)) / len(df)) * 100
        print(f"Reduced by {percent_reduction:.2f}%")
        
    except Exception as e:
        print(f"Error during trajectory reduction: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Run cluster detection on the reduced data
    print(f"\n=== STEP 2: CLUSTER DETECTION ===")
    print(f"Processing file: {reduced_file}")
    
    try:
        # Call the detect_and_remove_anomalies function from clusterDetection.py
        detect_and_remove_anomalies(
            reduced_file, 
            final_file, 
            density_threshold=density_threshold,
            radius_km=radius_km
        )
        print(f"Cluster detection completed successfully")   
    except Exception as e:
        print(f"Error during cluster detection: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Final summary
    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    print(f"Processing pipeline:")
    print(f"1. Original data: {input_file}")
    print(f"2. After trajectory reduction: {reduced_file}")
    print(f"3. Final cleaned data: {final_file}")

if __name__ == "__main__":
    mainNew()