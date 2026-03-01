import geopandas as gpd
import rasterio
from shapely.geometry import box

LINES_IN = "proximity/south_carolina_power_lines.geojson"
NDVI_TIF = "NDVI/ndvi_output.tif"
LINES_OUT = "proximity/power_lines_clipped.geojson"

lines = gpd.read_file(LINES_IN)

with rasterio.open(NDVI_TIF) as src:
    raster_crs = src.crs
    b = src.bounds
    raster_box = box(b.left, b.bottom, b.right, b.top)

lines_proj = lines.to_crs(raster_crs)
clipped = gpd.clip(lines_proj, raster_box)

clipped_4326 = clipped.to_crs("EPSG:4326")
clipped_4326.to_file(LINES_OUT, driver="GeoJSON")

print("✅ Wrote:", LINES_OUT)
print("Features before:", len(lines), "after:", len(clipped_4326))
