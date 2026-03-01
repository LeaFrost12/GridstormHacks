import os
import geopandas as gpd

IN_GJ = "proximity/power_lines_clipped.geojson"
OUT_GJ = "outputs/power_lines_clipped_utm32617.geojson"
DST_CRS = "EPSG:32617"

os.makedirs("outputs", exist_ok=True)

gdf = gpd.read_file(IN_GJ)
if gdf.crs is None:
    gdf = gdf.set_crs("EPSG:4326", allow_override=True)

gdf = gdf.to_crs(DST_CRS)
gdf.to_file(OUT_GJ, driver="GeoJSON")

print("✅ Wrote", OUT_GJ, "rows:", len(gdf), "crs:", gdf.crs)