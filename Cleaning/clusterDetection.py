import numpy as np
import sys
import os
import dask.dataframe as dd
from distributed import Client, LocalCluster
import math
from shapely.geometry import Point
import geopandas as gpd
from dask import delayed
import tempfile

META = {
    'MMSI': 'int64',
    'Latitude': 'float64',
    'Longitude': 'float64',
    'SOG': 'float64',
    'COG': 'float64',
    'Heading': 'float64',
    'ROT': 'float64',
    'Navigational status': 'obj1558ect',
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

# Add this new function at the top of your file
def prepare_river_data(file_path):
    """Pre-process river polygons for faster lookups"""
    land_gdf = gpd.read_file(file_path)
    land_gdf["geometry"] = land_gdf.geometry.buffer(0)  # Fix invalid geometries
    land_gdf = land_gdf.explode(index_parts=False)  # Ensure proper multi-polygons
    land_gdf.sindex  # Build spatial index
    return land_gdf


def mainDetectBehavior(df,
                            min_direction_changes=40,
                            min_course_change=120,
                            circular_threshold=300,
                            exclude_ship_types = 'Passenger',
                            client=None, batch_size = 200):
    
    use_existing_client = client is not None
    cluster = None

    try:
        import time
        start_time = time.time()
        
        if not use_existing_client:
            local_dir = tempfile.gettempdir()
            cluster = LocalCluster(n_workers=32, threads_per_worker=4, memory_limit="400GB", local_directory=local_dir)
            client = Client(cluster, timeout="120s")          
        
        exclude_ship_types = [] if exclude_ship_types is None else \
                             [exclude_ship_types] if isinstance(exclude_ship_types, str) else \
                             exclude_ship_types
            
        needed_cols = ['MMSI', '# Timestamp', 'Latitude', 'Longitude', 'SOG', 'COG', 'Ship type', 'Heading']
        unique_mmsis = df['MMSI'].unique()
        
        print(f"Analyzing {len(unique_mmsis)} unique vessels...")

        # Load land polygons with a spatial index
        land_gdf = prepare_river_data("Data/river_poly.geojson")
        print(f"River polygons loaded as {land_gdf.crs}")
        
        # Process in batches to improve monitoring and avoid memory issues
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
                df_group = df[df['MMSI'] == mmsi][needed_cols]

                df_group_len = df_group.size.compute()
            
                if df_group_len < 5:
                    skipped_vessels += 1
                    continue
                    
                if should_skip_vessel(df_group, exclude_ship_types):
                    skipped_vessels += 1
                    continue
                
                delayed_analysis = delayed(analyze_vessel_behavior)(
                    df_group,
                    land_gdf,
                    mmsi,
                    min_direction_changes,
                    min_course_change,
                    circular_threshold
                )
                future = client.compute(delayed_analysis)
                batch_futures.append(future)
                
                # Progress reporting within batch
                if (i+1) % 1000 == 0:
                    print(f"  Submitted {i+1}/{len(batch_mmsis)} vessels in current batch")
            
            all_futures.extend(batch_futures)
            
        unusual_mmsis = set()
        behavior_types = {}
        
        print(f"Analyzing {len(all_futures)} vessels (skipped {skipped_vessels} vessels)...")
        
        # Process futures in smaller batches to avoid losing all progress
        for i in range(0, len(all_futures), batch_size):
            batch = all_futures[i:i+batch_size]
            
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    batch_results = client.gather(batch)
                    valid_results = [r for r in batch_results if r is not None]
                    for result in valid_results:
                        if result and result['is_unusual']:
                            mmsi = result['mmsi']
                            unusual_mmsis.add(mmsi)
                            behavior_types[mmsi] = result['behaviors']
                    print(f"Processed {min(i+batch_size, len(all_futures))}/{len(all_futures)} results")
                    break  # Success - exit retry loop
                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        print(f"Error processing batch {i//batch_size + 1} after {max_retries} attempts: {str(e)[:100]}...")
                    else:
                        print(f"Attempt {retry_count}/{max_retries} failed for batch {i//batch_size + 1}. Retrying...")
                        import time
                        time.sleep(2)  # Add a small delay before retrying
        
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


def analyze_vessel_behavior(df, land_gdf, mmsi, min_direction_changes, min_course_change, circular_threshold):
    """Analyze a single vessel's behavior, can be run in parallel"""
    try:
        df = df.sort_values('# Timestamp')
        
        result = {
            'mmsi': mmsi,
            'is_unusual': False,
            'behaviors': []
        }
        
        coords = df[['Latitude', 'Longitude']].values

        # Vectorized point-in-polygon check - this is the critical performance improvement
        points_gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(df.Longitude, df.Latitude),
            crs=land_gdf.crs
        )
        
        # Spatial join instead of point-by-point checking
        joined = gpd.sjoin(points_gdf, land_gdf, how="left", predicate="within")
        points_outside_river = joined[joined.index_right.isna()]
        
        found_point_outside_river = len(points_outside_river) > 0

        if found_point_outside_river:
            # Only run expensive detection algorithms if needed
            if detect_circular_movement(coords, circular_threshold, area_threshold_km2=2):
                result['behaviors'].append("circular_movement")
                result['is_unusual'] = True
                        
                            
            if detect_erratic_movement(df, min_direction_changes, min_course_change):
                result['behaviors'].append("erratic_movement")
                result['is_unusual'] = True

        return result
    except Exception as e:
        print(f"Error analyzing vessel {mmsi}: {e}")
        return None


def process_results(unusual_data, df):
    """Process results and save cleaned data"""
    unusual_mmsis = unusual_data['unusual_mmsis']
    behavior_types = unusual_data['behavior_types']
    total_vessels = unusual_data['total_vessels']
    skipped_vessels = unusual_data.get('skipped_vessels', 0)
    processing_time = unusual_data.get('processing_time', 0)
    
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
            return df
    else:
        print("No ships with unusual behavior detected with current parameters")
        try:
            return df
        except Exception as e:
            print(f"Error saving data: {e}")
            return df


def detect_circular_movement(coords, threshold=300, area_threshold_km2=2):
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
    
    if total_course_change < threshold:
        return False
    
    lats = [coord[0] for coord in coords]
    lons = [coord[1] for coord in coords]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    lat_avg = (min_lat + max_lat) / 2  # Use average latitude for longitude conversion
    lon_meters_per_degree = 111320 * math.cos(math.radians(lat_avg))
    lat_meters_per_degree = 111320

    width_m = (max_lon - min_lon) * lon_meters_per_degree
    height_m = (max_lat - min_lat) * lat_meters_per_degree
    area_m2 = width_m * height_m
    area_km2 = area_m2 / 1000000  # Convert to km²




    return total_course_change <= area_threshold_km2


def detect_erratic_movement(ship_data, min_direction_changes, min_course_change):
    direction_changes = 0
    prev_cog = None
    
    # Process each point in chronological order
    for _, row in ship_data.iterrows():
        if prev_cog is not None and not row['COG'] == row['COG'] and prev_cog == prev_cog:
            course_diff = calculate_course_change(prev_cog, row['COG'])
            if course_diff >= min_course_change:
                direction_changes += 1
        prev_cog = row['COG'] if row['COG'] == row['COG'] else prev_cog
    
    return direction_changes >= min_direction_changes



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


def calculate_course_change(cog1, cog2):
    """Calculate absolute course change between two COG values (0-360 degrees)"""
    diff = abs(cog1 - cog2)
    if diff > 180:
        diff = 360 - diff
    return diff


def should_skip_vessel(ship_data, exclude_ship_types):
    """Check if vessel should be skipped based on ship type"""
    if not exclude_ship_types or 'Ship type' not in ship_data.columns:
        return False

    if isinstance(ship_data, dd.DataFrame):
        ship_types = ship_data['Ship type'].dropna().unique().compute()
    else:
        ship_types = ship_data['Ship type'].dropna().unique()
        
    if len(ship_types) == 0:
        return False
    
    ship_types = [str(st) for st in ship_types]
        
    return any(excluded in st for st in ship_types for excluded in exclude_ship_types)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    mainDetectBehavior(input_file, output_file, exclude_ship_types=['Passenger'])
