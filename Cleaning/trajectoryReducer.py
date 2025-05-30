import pandas as pd
import sys
import os
from shapely.geometry import Point
import geopandas as gpd

def trajectory_reducer(df):
    start_rows = len(df)
    result_dfs = []

    # Load land polygons with a spatial index
    land_gdf = gpd.read_file("Data/river_poly.geojson")
    print(f"River polygons loaded as {land_gdf.crs}")
    land_gdf["geometry"] = land_gdf.geometry.buffer(0)  # Fix invalid geometries
    land_gdf = land_gdf.explode(index_parts=False)  # Ensure proper multi-polygons
    land_gdf.sindex  # Build spatial index
    
    for mmsi, group in df.groupby('MMSI'):
        group = group.sort_values('# Timestamp')
        
        # Always keep first point
        reduced_group = [group.iloc[0]]
        prevRow = group.iloc[0]
        
        # Process middle points (all except first and last)
        for _, row in group.iloc[1:-1].iterrows():
            speed = row.SOG
            
            # Create Point object for current location
            point = Point(row.Longitude, row.Latitude)
            
            # Check if point is within any river polygon
            possible_matches_idx = list(land_gdf.sindex.intersection(point.bounds))
            if possible_matches_idx:
                possible_matches = land_gdf.iloc[possible_matches_idx]
                within_river = any(possible_matches.contains(point))
                
                # Use much lower threshold for points inside river areas
                if within_river:
                    threshold = 1  # Lower threshold for river points
                else:
                    # Regular threshold logic for non-river points
                    if speed < 10:
                        threshold = 3
                    elif speed < 20:
                        threshold = 6
                    else:
                        threshold = 9
            else:
                # Default threshold for points clearly outside river areas
                if speed < 10:
                    threshold = 3
                elif speed < 20:
                    threshold = 6
                else:
                    threshold = 9
            
            delta = abs(prevRow.COG - row.COG)
            if delta >= threshold:
                reduced_group.append(row)
                prevRow = row
        
        # Always add the last point if it exists and is different from the first
        if len(group) > 1 and not group.iloc[-1].equals(group.iloc[0]):
            reduced_group.append(group.iloc[-1])

        if (len(reduced_group) == 2):
            print(f"Warning: Only one row remaining in the DataFrame after trajectory reducer (removing all): {reduced_group[0]['MMSI']}")
            result_dfs.append(pd.DataFrame(columns=df.columns))
            continue

        # Convert list of rows back to DataFrame
        result_df = pd.DataFrame(reduced_group)
        result_dfs.append(result_df)
    
    # Combine all reduced groups
    final_df = pd.concat(result_dfs)
    end_rows = len(final_df)
    reduction_percent = ((start_rows - end_rows) / start_rows) * 100
    print(f"Reduced dataset from {start_rows} to {end_rows} rows ({reduction_percent:.2f}% reduction)")
    return final_df

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python trajectory_reducer.py <filename>")
        sys.exit(1)

    fileName = sys.argv[1]
    df = pd.read_csv(fileName)
    result_df = trajectory_reducer(df)
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        base_name = os.path.basename(fileName)
        name_parts = os.path.splitext(base_name)
        output_file = f"{name_parts[0]}_trajectory_reduced{name_parts[1]}"
    result_df.to_csv(output_file, index=False)
