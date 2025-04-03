import pandas as pd
import numpy as np
import sys
import os
import dask.dataframe as dd
from distributed import Client, LocalCluster
import math

META = {
    'MMSI': 'int64',
    'Latitude': 'float64',
    'Longitude': 'float64',
    'SOG': 'float64',
    'COG': 'float64',
    'Heading': 'float64',
    'ROT': 'float64',
    'Navigational status': 'object',
    'IMO': 'object',
    'Callsign': 'object',
    'Name': 'object',
    'Ship type': 'object',
    'Cargo type': 'object',
    'Width': 'float64',
    'Length': 'float64',
    'Type of position fixing device': 'object',
    'Draught': 'float64',
    'Destination': 'object',
    'ETA': 'object',
    'Data source type': 'object',
    'A': 'float64',
    'B': 'float64',
    'C': 'float64',
    'D': 'float64'
}

def calculate_bearing(lat1, lon1, lat2, lon2):
    try:
        y = math.sin(math.radians(lon2 - lon1)) * math.cos(math.radians(lat2))
        x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - \
            math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
            math.cos(math.radians(lon2 - lon1))
        return (math.degrees(math.atan2(y, x)) + 360) % 360
    except Exception as e:
        print(f"Error calculating bearing: {e}")
        return 0

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0 # Earth radius in kilometers
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c

def calculate_course_change(cog1, cog2):
    """Calculate absolute course change between two COG values (0-360 degrees)"""
    diff = abs(cog1 - cog2)
    if diff > 180:
        diff = 360 - diff
    return diff


def detect_backward_sailing(ship_data, min_duration=25):
    backward_count = 0
    max_backward_count = 0
    threshold = 170

    if len(ship_data) < 2:
        return False
    
    data = ship_data.sort_values('# Timestamp').reset_index(drop=True)
    
    for i in range(1, len(data)):
        current = data.iloc[i]
        previous = data.iloc[i-1]
        
        # Skip if stationary or missing heading
        if pd.isna(current['Heading']):
            backward_count = 0
            continue
        
        # Calculate actual movement direction between consecutive points
        actual_lat1, actual_lon1 = previous['Latitude'], previous['Longitude']
        actual_lat2, actual_lon2 = current['Latitude'], current['Longitude']
        
        # Skip if positions are identical
        if actual_lat1 == actual_lat2 and actual_lon1 == actual_lon2:
            continue
            
        movement_bearing = calculate_bearing(actual_lat1, actual_lon1, actual_lat2, actual_lon2)

        # Compare movement bearing with vessel heading
        heading = current['Heading']
        angle_diff = calculate_course_change(movement_bearing, heading)
        
        # If angle difference is close to 180 degrees, vessel is moving backward
        if angle_diff > threshold:
            backward_count += 1
            max_backward_count = max(max_backward_count, backward_count)
        else:
            backward_count = 0
    
    return max_backward_count >= min_duration

def detect_circular_movement(coords, threshold=300):
    if len(coords) < 3:
        return False
    # Calculate COG between consecutive points
    course_changes = []
    for i in range(len(coords)-2):
        lat1, lon1 = coords[i]
        lat2, lon2 = coords[i+1]
        lat3, lon3 = coords[i+2]
        # Calculate bearings
        bearing1 = calculate_bearing(lat1, lon1, lat2, lon2)
        bearing2 = calculate_bearing(lat2, lon2, lat3, lon3)
        course_diff = calculate_course_change(bearing1, bearing2)
        course_changes.append(course_diff)
    # Sum total course changes
    total_course_change = sum(course_changes)
    return total_course_change >= threshold

def detect_erratic_movement(ship_data, min_direction_changes, min_course_change):
    direction_changes = 0
    prev_cog = None
    
    # Process each point in chronological order
    for _, row in ship_data.iterrows():
        if prev_cog is not None and not pd.isna(row['COG']) and not pd.isna(prev_cog):
            course_diff = calculate_course_change(prev_cog, row['COG'])
            if course_diff >= min_course_change:
                direction_changes += 1
        prev_cog = row['COG'] if not pd.isna(row['COG']) else prev_cog
    
    return direction_changes >= min_direction_changes


def analyze_vessel_behavior(vessel_data, mmsi, min_direction_changes, min_course_change, circular_threshold):
    """Analyze a single vessel's behavior, can be run in parallel"""
    try:
        # Sort by timestamp
        vessel_data = vessel_data.sort_values('# Timestamp')
        
        # Initialize result
        result = {
            'mmsi': mmsi,
            'is_unusual': False,
            'behaviors': []
        }
        
        # Extract coordinates for trajectory analysis
        coords = vessel_data[['Latitude', 'Longitude']].values
        
        # Detect unusual behaviors
        if detect_circular_movement(coords, circular_threshold):
            result['behaviors'].append("circular_movement")
            result['is_unusual'] = True
        
        if detect_backward_sailing(vessel_data):
            result['behaviors'].append("backward_sailing")
            result['is_unusual'] = True
        
        if detect_erratic_movement(vessel_data, min_direction_changes, min_course_change):
            result['behaviors'].append("erratic_movement")
            result['is_unusual'] = True
            
        return result
    except Exception as e:
        # Handle errors gracefully
        print(f"Error analyzing vessel {mmsi}: {e}")
        return None


def detect_unusual_behavior(df,
                            min_direction_changes=40,
                            min_course_change=120,
                            circular_threshold=300,
                            exclude_ship_types = 'Passenger',
                            client=None,):
    use_existing_client = client is not None
    cluster = None
    
    try:
        import time
        start_time = time.time()
        
        if not use_existing_client:
            print("Creating new Dask cluster for analysis")
            cluster = LocalCluster(n_workers=32, threads_per_worker=4, memory_limit="300GB")
            client = Client(cluster)
        else:
            print("Using existing Dask client for analysis")

        
        exclude_ship_types = [] if exclude_ship_types is None else \
                             [exclude_ship_types] if isinstance(exclude_ship_types, str) else \
                             exclude_ship_types
            
        needed_cols = ['MMSI', '# Timestamp', 'Latitude', 'Longitude', 'SOG', 'COG', 'Ship type', 'Heading']
        unique_mmsis = df['MMSI'].unique()
        
        print(f"Analyzing {len(unique_mmsis)} unique vessels...")
        
        # Process in batches to improve monitoring and avoid memory issues
        batch_size = 5000  # Larger batch size for supercomputer
        num_batches = (len(unique_mmsis) + batch_size - 1) // batch_size
        
        all_futures = []
        skipped_vessels = 0
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(unique_mmsis))
            batch_mmsis = unique_mmsis[batch_start:batch_end]
            
            print(f"Processing batch {batch_idx+1}/{num_batches} ({len(batch_mmsis)} vessels)")
            batch_futures = []
            
            for i, mmsi in enumerate(batch_mmsis):
                # Get data for this vessel
                vessel_data = df[df['MMSI'] == mmsi][needed_cols]
                
                # Skip if insufficient data points
                if len(vessel_data) < 5:
                    skipped_vessels += 1
                    continue
                    
                # Skip excluded vessel types
                if should_skip_vessel(vessel_data, exclude_ship_types):
                    skipped_vessels += 1
                    continue
                
                # Submit analysis task to Dask
                future = client.submit(
                    analyze_vessel_behavior,
                    vessel_data,
                    mmsi,
                    min_direction_changes,
                    min_course_change,
                    circular_threshold
                )
                batch_futures.append(future)
                
                # Progress reporting within batch
                if (i+1) % 1000 == 0:
                    print(f"  Submitted {i+1}/{len(batch_mmsis)} vessels in current batch")
            
            # Add batch futures to all futures
            all_futures.extend(batch_futures)
            
        
        print(f"Analyzing {len(all_futures)} vessels (skipped {skipped_vessels} vessels)...")
        results = client.gather(all_futures)
        
        # Process results
        unusual_mmsis = set()
        behavior_types = {}
        
        for result in results:
            if result and result['is_unusual']:
                mmsi = result['mmsi']
                unusual_mmsis.add(mmsi)
                behavior_types[mmsi] = result['behaviors']
        
        processing_time = time.time() - start_time
        print(f"Analysis completed in {processing_time:.1f} seconds")
        
        unusual_data = {
            'unusual_mmsis': unusual_mmsis,
            'behavior_types': behavior_types,
            'skipped_vessels': skipped_vessels,
            'total_vessels': len(unique_mmsis),
            'processing_time': processing_time
        }
        
        return process_results(unusual_data, df)
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Clean up resources
        if not use_existing_client:
            if client:
                try:
                    client.close()
                except:
                    pass
            if cluster:
                try:
                    cluster.close()
                except:
                    pass
        else:
            print("Using existing Dask client, not closing it.")


def should_skip_vessel(ship_data, exclude_ship_types):
    """Check if vessel should be skipped based on ship type"""
    if not exclude_ship_types or 'Ship type' not in ship_data.columns:
        return False
        
    ship_types = ship_data['Ship type'].dropna().astype(str).unique()
    if len(ship_types) == 0:
        return False
        
    return any(excluded in st for st in ship_types for excluded in exclude_ship_types)


def process_results(unusual_data, df):
    """Process results and save cleaned data"""
    unusual_mmsis = unusual_data['unusual_mmsis']
    behavior_types = unusual_data['behavior_types']
    total_vessels = unusual_data['total_vessels']
    skipped_vessels = unusual_data.get('skipped_vessels', 0)
    processing_time = unusual_data.get('processing_time', 0)
    
    # Handle case where unusual vessels were found
    if unusual_mmsis:
        print(f"\nDetected {len(unusual_mmsis)} ships with unusual behavior")
 
        behavior_counts = {}
        for behaviors in behavior_types.values():
            for b in behaviors:
                behavior_counts[b] = behavior_counts.get(b, 0) + 1
        
        print("\nTotal vessels caught by each behavior:")
        for method, count in sorted(behavior_counts.items()):
            print(f"- {method}: {count} vessels")
        

        # Remove unusual vessels
        try:
            print(f"Removing unusual ships from dataset...")
            cleaned_df = df[~df['MMSI'].isin(list(unusual_mmsis))]

            percent_removed = (len(unusual_mmsis) / total_vessels) * 100 if total_vessels > 0 else 0
            print(f"\nCleaning summary:")
            print(f"- Total vessels analyzed: {total_vessels}")
            print(f"- Vessels skipped: {skipped_vessels}")
            print(f"- Unusual vessels removed: {len(unusual_mmsis)} ({percent_removed:.2f}%)")
            if processing_time > 0:
                print(f"- Processing time: {processing_time:.1f} seconds")
            
            return cleaned_df
        except Exception as e:
            print(f"Error saving cleaned data: {e}")
            return None
    else:
        print("No ships with unusual behavior detected with current parameters")
        try:
            return df
        except Exception as e:
            print(f"Error saving data: {e}")
            return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    detect_unusual_behavior(input_file, output_file, exclude_ship_types=['Passenger'])