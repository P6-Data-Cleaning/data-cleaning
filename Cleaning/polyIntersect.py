import geopandas as gpd
import pandas as pd
from shapely import Point
from shapely.geometry import LineString
import time
import numpy as np

def process_ship(mmsi, group, land_gdf):
    """Checks if a ship's path intersects with land and returns points to remove."""
    # Select only necessary columns to reduce data size
    group = group[["# Timestamp", "Longitude", "Latitude"]].copy()
    
    # Sort by timestamp to ensure proper ordering
    group = group.sort_values(by='# Timestamp')
    
    # Get coordinates
    coordinates = list(zip(group["Longitude"].round(6), group["Latitude"].round(6)))
    if len(coordinates) < 2:
        return None, []  # Skip ships with too few points
    
    ship_path = LineString(coordinates)
    points_to_remove = []
    
    # Faster intersection check using spatial index
    possible_matches = land_gdf.sindex.query(ship_path, predicate="intersects")
    if len(possible_matches) > 0:
        candidate_polygons = land_gdf.iloc[possible_matches].geometry
        ship_crosses_land = False
        
        # Track furthest intersection points at start and end
        max_start_intersection = -1
        min_end_intersection = len(coordinates)
        
        # Check individual line segments for intersections
        for i in range(len(coordinates) - 1):
            segment = LineString([coordinates[i], coordinates[i+1]])
            
            # Only check beginning (first 5 points) or end (last 5 points)
            if i < 5:
                for polygon in candidate_polygons:
                    if segment.intersects(polygon):
                        max_start_intersection = max(max_start_intersection, i+1)
            elif i >= len(coordinates) - 6:
                for polygon in candidate_polygons:
                    if segment.intersects(polygon):
                        min_end_intersection = min(min_end_intersection, i)
            else:
                for polygon in candidate_polygons:
                    if segment.intersects(polygon):
                        ship_crosses_land = True
        
        # If any middle segment crosses land, mark the whole ship for removal
        if ship_crosses_land:
            return mmsi, []
        
        # Process start intersections - remove points from 0 to max_start_intersection
        if max_start_intersection > -1:
            points_to_remove.extend(group.iloc[:max_start_intersection].index)
                
        # Process end intersections - remove points from min_end_intersection to end
        if min_end_intersection < len(coordinates):
            points_to_remove.extend(group.iloc[min_end_intersection:].index)
        
        if points_to_remove:
            return None, points_to_remove
    
    return None, []  # This ship doesn't cross land or only at endpoints that we'll remove

def poly_intersect(df):
    start_time = time.time()

    # Verify that the required columns exist
    required_columns = {"MMSI", "# Timestamp", "Longitude", "Latitude"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    # Load land polygons with a spatial index
    land_gdf = gpd.read_file("Data/land_poly.geojson")
    print(f"Land polygons loaded as {land_gdf.crs}")
    land_gdf["geometry"] = land_gdf.geometry.buffer(0)  # Fix invalid geometries
    land_gdf = land_gdf.explode(index_parts=False)  # Ensure proper multi-polygons
    land_gdf.sindex  # Build spatial index

    # Split the DataFrame into batches based on unique MMSI values
    unique_mmsi = df["MMSI"].unique()
    mmsi_batches = np.array_split(unique_mmsi, 100)  # Adjust the number of batches as needed

    all_results = []
    print("Processing ships in batches...")
    
    # Process in batches without using Dask
    for batch_idx, batch in enumerate(mmsi_batches):
        print(f"Processing batch {batch_idx+1}/{len(mmsi_batches)}")
        batch_df = df[df["MMSI"].isin(batch)]
        batch_results = []
        
        for mmsi, group in batch_df.groupby("MMSI"):
            if not group.empty:
                # Process directly without using delayed
                result = process_ship(mmsi, group, land_gdf)
                batch_results.append(result)
        
        all_results.extend(batch_results)
    
    # Separate ships to remove and points to remove
    crossing_ships = [res[0] for res in all_results if res[0] is not None]
    points_to_remove = []
    
    # Track which ships have points removed at start/end
    ships_with_trimmed_points = set()
    for res in all_results:
        _, pts = res
        if pts:
            # Get unique MMSI values for points to be removed
            if len(pts) > 0:
                pts_df = df.loc[df.index.isin(pts)]
                if not pts_df.empty:
                    ships_with_trimmed_points.update(pts_df['MMSI'].unique())
            points_to_remove.extend(pts)
    
    print(f"Ships crossing land (to be removed): {crossing_ships}")
    print(f"Ships with points trimmed at start/end: {list(ships_with_trimmed_points)}")
    print(f"Number of individual points intersecting land at start/end: {len(points_to_remove)}")
    print("Time taken:", time.time() - start_time)

    # Create a cleaned dataset
    if crossing_ships or points_to_remove:
        df_clean = df[~df["MMSI"].isin(crossing_ships)]
        if points_to_remove:
            df_clean = df_clean.drop(index=points_to_remove, errors='ignore')
        
        print(f"Cleaned data saved. Removed {len(crossing_ships)} ships and {len(points_to_remove)} individual points.")
        return df_clean
    else:
        print("No ships or points needed removal.")
        return df

if __name__ == "__main__":
    poly_intersect()