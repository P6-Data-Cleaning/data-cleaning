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
    """Checks if a ship's path intersects with land."""
    coordinates = list(group.geometry)
    # Ensure that ship coordinates are in sufficient precision
    coordinates = list(zip(group["longitude"].round(6), group["latitude"].round(6)))
    if len(coordinates) < 2:
        return None  # Skip ships with too few points
    
    ship_path = LineString(coordinates)
    
    # Faster intersection check using spatial index
    possible_matches = land_gdf.sindex.query(ship_path, predicate="intersects")
    if len(possible_matches) > 0:
        candidate_polygons = land_gdf.iloc[possible_matches].geometry
        for idx, polygon in zip(possible_matches, candidate_polygons):
            if ship_path.intersects(polygon):
                # Get the actual intersection points
                intersection = ship_path.intersection(polygon)
                print(f"Ship {mmsi} crossed land at polygon index {idx}")
                print(f"Intersection coordinatess: {intersection}")

                if (mmsi == 211718360):
                    # Save ship path as a GeoDataFrame
                    ship_gdf = gpd.GeoDataFrame(geometry=[ship_path], crs="EPSG:4326")
                    ship_gdf.to_file("outputs/csv/ship_path.geojson", driver="GeoJSON")

                    # Save intersection as a GeoDataFrame
                    intersection_gdf = gpd.GeoDataFrame(geometry=[intersection], crs="EPSG:4326")
                    intersection_gdf.to_file("outputs/csv/ship_intersection.geojson", driver="GeoJSON")
        return mmsi  # This ship crosses land
    
    return None

def poly_intersect():
    client = setup_dask()
    
    start_time = time.time()

    # Load land polygons with a spatial index
    land_gdf = gpd.read_file("Data/land_poly.geojson")
    print(f"Land polygons loaded as {land_gdf.crs}")
    land_gdf["geometry"] = land_gdf.geometry.buffer(0)  # Fix invalid geometries
    land_gdf = land_gdf.explode(index_parts=False)  # Ensure proper multi-polygons
    land_gdf.sindex  # Build spatial index

    # Load ship data
    df = pd.read_csv("outputs/csv/cleaned_data_without_reduced.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['mmsi', 'timestamp'])

    # Convert ship data to a GeoDataFrame with EPSG:4326
    df["geometry"] = df.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
    ship_gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    print(f"Ship data loaded with CRS: {ship_gdf.crs}")

    # Convert to Dask DataFrame for parallel processing
    ddf = dd.from_pandas(df, npartitions=8)

    # Process each ship in parallel
    delayed_results = [
        delayed(process_ship)(mmsi, group, land_gdf)
        for mmsi, group in df.groupby("mmsi")
    ]

    # Compute results
    crossing_ships = [res for res in dd.compute(*delayed_results) if res is not None]

    print("The following ships crossed land:", crossing_ships)
    print("Time taken:", time.time() - start_time)

    # Remove ships that crossed land
    if crossing_ships:
        df = df[~df["mmsi"].isin(crossing_ships)]
        df.to_csv("outputs/csv/cleaned_data_no_land.csv", index=False)
        print("Cleaned data saved to 'outputs/csv/cleaned_data_no_land2.csv'.")

if __name__ == "__main__":
    poly_intersect()