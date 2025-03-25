import geopandas as gpd

# Load landmass polygon (from Natural Earth or Geofabrik)
land = gpd.read_file("Cleaning/land_poly.geojson")

# Load the Kiel Canal polygon
kiel_canal = gpd.read_file("Cleaning/anadrom.geojson")

# Ensure CRS matches (important for spatial operations)
land = land.to_crs("EPSG:4326")
kiel_canal = kiel_canal.to_crs("EPSG:4326")

# Subtract the canal from the land polygon
updated_land = land.overlay(kiel_canal, how="difference")

# Save the new land polygon
updated_land.to_file("land_poly.geojson", driver="GeoJSON")

print("Updated land polygon saved as 'land_poly.geojson'")

""" import geopandas as gpd
from shapely.geometry import LineString

# Load the full land polygon
land = gpd.read_file("Cleaning/kiel.shp")

# Define bounding box for Denmark (approximately)
denmark_bbox = (9, 53.86, 10.27, 54.48)  # (minx, miny, maxx, maxy)

#1 Clip land polygon to the bounding box for Denmark
land_clipped = land.cx[denmark_bbox[0]:denmark_bbox[2], denmark_bbox[1]:denmark_bbox[3]]

# save the clipped land polygon
land_clipped.to_file("keil.geojson", driver="GeoJSON") """

""" import geopandas as gpd
from shapely.geometry import LineString

# Load the full land polygon
land = gpd.read_file("Cleaning/keil.geojson")

land_filtered = land[land["name"].isna() | (~land["name"].str.lower().str.contains("kiel", na=False))]

# Save the filtered land polygon
land_filtered.to_file("kiel_filtered.geojson", driver="GeoJSON")
print("Filtered land polygon saved as 'kiel_filtered.geojson'") """