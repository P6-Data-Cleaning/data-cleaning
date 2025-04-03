import geopandas as gpd
import pandas as pd
from shapely import Point
from shapely.geometry import LineString
import time
import dask.dataframe as dd
from dask import delayed

def setup_dask():
    from distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=28, threads_per_worker=4, memory_limit="300GB")
    return Client(cluster)

def process_ship(mmsi, group, land_gdf):
    """Checks if a ship's path intersects with land and returns points to remove."""
    # Convert group to DataFrame if it's a Series
    if isinstance(group, pd.Series):
        group = pd.DataFrame([group])
    
    # Sort by timestamp to ensure proper ordering
    group = group.sort_values(by='timestamp')
    
    # Get coordinates
    coordinates = list(zip(group["longitude"].round(6), group["latitude"].round(6)))
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
                        # Remember the furthest intersecting segment at the start
                        max_start_intersection = max(max_start_intersection, i+1)
            elif i >= len(coordinates) - 6:
                for polygon in candidate_polygons:
                    if segment.intersects(polygon):
                        # Remember the earliest intersecting segment at the end
                        min_end_intersection = min(min_end_intersection, i)
            else:
                # For middle segments, just check if they cross land
                for polygon in candidate_polygons:
                    if segment.intersects(polygon):
                        ship_crosses_land = True
        
        # If any middle segment crosses land, mark the whole ship for removal
        if ship_crosses_land:
            return mmsi, []
        
        # Process start intersections - remove points from 0 to max_start_intersection
        if max_start_intersection > -1:
            for i in range(max_start_intersection):
                points_to_remove.append(group.iloc[i].name)
                
        # Process end intersections - remove points from min_end_intersection to end
        if min_end_intersection < len(coordinates):
            for i in range(min_end_intersection, len(coordinates)):
                points_to_remove.append(group.iloc[i].name)
        
        if points_to_remove:
            # Only return points to remove if there are some
            return None, points_to_remove
    
    return None, []  # This ship doesn't cross land or only at endpoints that we'll remove

def poly_intersect(df):
    start_time = time.time()

    # Load land polygons with a spatial index
    land_gdf = gpd.read_file("Data/land_poly.geojson")
    print(f"Land polygons loaded as {land_gdf.crs}")
    land_gdf["geometry"] = land_gdf.geometry.buffer(0)  # Fix invalid geometries
    land_gdf = land_gdf.explode(index_parts=False)  # Ensure proper multi-polygons
    land_gdf.sindex  # Build spatial index

    # Load ship data
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['mmsi', 'timestamp'])

    # Convert ship data to a GeoDataFrame with EPSG:4326
    df["geometry"] = df.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
    ship_gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    print(f"Ship data loaded with CRS: {ship_gdf.crs}")

    # Process each ship in parallel
    delayed_results = [
        delayed(process_ship)(mmsi, group, land_gdf)
        for mmsi, group in df.groupby("mmsi")
    ]

    # Compute results
    results = dd.compute(*delayed_results)
    
    # Separate ships to remove and points to remove
    crossing_ships = [res[0] for res in results if res[0] is not None]
    points_to_remove = []
    
    # Track which ships have points removed at start/end
    ships_with_trimmed_points = set()
    for res in results:
        _, pts = res
        if pts:  # If this ship has points to remove
            # Get the MMSI for these points
            point_indices = pts
            if point_indices:
                # Find the corresponding MMSIs for these points
                for idx in point_indices:
                    if idx in df.index:
                        ships_with_trimmed_points.add(df.loc[idx, 'mmsi'])
                points_to_remove.extend(pts)
    
    print(f"Ships crossing land (to be removed): {crossing_ships}")
    print(f"Ships with points trimmed at start/end: {list(ships_with_trimmed_points)}")
    print(f"Number of individual points intersecting land at start/end: {len(points_to_remove)}")
    print("Time taken:", time.time() - start_time)

    # Create a cleaned dataset
    if crossing_ships or points_to_remove:
        # Remove whole ships that crossed land in the middle
        df_clean = df[~df["mmsi"].isin(crossing_ships)]
        
        # Remove individual points at the start/end that crossed land
        if points_to_remove:
            df_clean = df_clean.drop(points_to_remove)
        
        print(f"Cleaned data saved. Removed {len(crossing_ships)} ships and {len(points_to_remove)} individual points.")
        return df_clean
    else:
        print("No ships or points needed removal.")

if __name__ == "__main__":
    poly_intersect()