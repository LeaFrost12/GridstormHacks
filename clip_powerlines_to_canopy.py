import geopandas as gpd
import rasterio
from shapely.geometry import box

CANOPY_TIF = "height/data/columbia_canopy_height.tif"
IN_LINES = "proximity/south_carolina_power_lines.geojson"
OUT_LINES = "proximity/power_lines_clipped.geojson"

with rasterio.open(CANOPY_TIF) as src:
    bounds = src.bounds
    crs = src.crs

bbox = gpd.GeoDataFrame(
    geometry=[box(bounds.left, bounds.bottom, bounds.right, bounds.top)],
    crs=crs
)

lines = gpd.read_file(IN_LINES)

# Ensure same CRS
if lines.crs is None:
    # If your source has no CRS, it's usually EPSG:4326
    lines = lines.set_crs("EPSG:4326", allow_override=True)

if lines.crs != crs:
    lines = lines.to_crs(crs)

clipped = gpd.clip(lines, bbox)

clipped.to_file(OUT_LINES, driver="GeoJSON")
print("✅ Wrote:", OUT_LINES)
print("Rows:", len(clipped))