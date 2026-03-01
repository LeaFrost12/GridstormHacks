import geopandas as gpd
import rasterio
import numpy as np
from shapely.geometry import mapping, box
from rasterio.mask import mask

POWER_LINES = "proximity/south_carolina_power_lines.geojson"
NDVI_TIF = "NDVI/ndvi_output.tif"
OUTPUT = "proximity/powerline_risk.geojson"

BUFFER_METERS = 50

print("Loading power lines...")
lines = gpd.read_file(POWER_LINES)
print("Loaded features:", len(lines), "CRS:", lines.crs)

if len(lines) == 0:
    raise SystemExit("ERROR: Input power lines file has 0 features.")

print("Opening NDVI raster...")
with rasterio.open(NDVI_TIF) as src:
    print("Raster CRS:", src.crs)
    print("Raster bounds:", src.bounds)

    # Reproject powerlines to raster CRS
    lines_proj = lines.to_crs(src.crs)
    print("Reprojected features:", len(lines_proj))

    # Clip powerlines to raster extent to guarantee overlap
    raster_extent = box(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
    lines_clip = gpd.clip(lines_proj, raster_extent)
    print("Clipped-to-raster features:", len(lines_clip))

    if len(lines_clip) == 0:
        raise SystemExit(
            "ERROR: After clipping to NDVI raster extent, 0 powerline features remain.\n"
            "This means your NDVI tile is not covering the same region as your powerline data."
        )

    risk_scores = []
    kept_geoms = 0

    for geom in lines_clip.geometry:
        if geom is None or geom.is_empty:
            risk_scores.append(None)
            continue

        buffered = geom.buffer(BUFFER_METERS)

        try:
            out_image, _ = mask(src, [mapping(buffered)], crop=True, filled=False)
        except ValueError:
            # no overlap for this geometry
            risk_scores.append(None)
            continue

        # IMPORTANT FIX: .copy() prevents "output array is read-only"
        data = out_image[0].astype(np.float32).copy()

        # Convert any non-finite values to NaN, then drop NaNs
        data = np.where(np.isfinite(data), data, np.nan)
        data = data[~np.isnan(data)]

        if data.size == 0:
            risk_scores.append(None)
            continue

        mean_ndvi = float(np.nanmean(data))
        risk_scores.append(mean_ndvi)
        kept_geoms += 1

    lines_clip["risk_score"] = risk_scores

print("Computed risk for features:", kept_geoms, "out of", len(lines_clip))

# Save output in EPSG:4326 for frontend
lines_out = lines_clip.to_crs("EPSG:4326")
print("Saving GeoJSON...", OUTPUT)
lines_out.to_file(OUTPUT, driver="GeoJSON")

print("✅ Risk model complete:", OUTPUT)
print("Output features:", len(lines_out))