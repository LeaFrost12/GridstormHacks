import os
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt


def normalize(arr, nodata_mask=None):
    """Normalize array to 0..1 ignoring nodata/NaNs."""
    a = arr.astype("float32").copy()

    if nodata_mask is not None:
        a[nodata_mask] = np.nan

    if np.isnan(a).all():
        return np.zeros_like(a, dtype="float32")

    mn = np.nanmin(a)
    mx = np.nanmax(a)

    if mx - mn < 1e-9:
        return np.zeros_like(a, dtype="float32")

    out = (a - mn) / (mx - mn)
    out = np.nan_to_num(out, nan=0.0)
    return out.astype("float32")


def main():
    # --------- INPUT FILES ----------
    CANOPY_TIF = "outputs/canopy_utm32617.tif"
    POWERLINES_GEOJSON = "outputs/power_lines_clipped_utm32617.geojson"
    NDVI_TIF = "outputs/ndvi_on_canopy_utm.tif"
    USE_NDVI = True

    # --------- OUTPUT FILES ----------
    OUT_RISK_TIF = "outputs/risk_map.tif"
    OUT_DIST_TIF = "outputs/distance_to_powerlines_m.tif"

    os.makedirs("outputs", exist_ok=True)

    # --------- WEIGHTS ----------
    # Encroachment score replaces raw canopy height in the risk equation
    if USE_NDVI:
        W_ENCROACH, W_DISTANCE, W_NDVI = 0.4, 0.4, 0.2
    else:
        W_ENCROACH, W_DISTANCE, W_NDVI = 0.5, 0.5, 0.0

    # --------- ENCROACHMENT PARAMETERS ----------
    # These are defensible defaults; tune later if desired.
    CLEARANCE_M = 6.0     # baseline clearance required near lines
    SLOPE = 0.15          # increases required clearance with distance (m per m)

    # --------- LOAD CANOPY ----------
    with rasterio.open(CANOPY_TIF) as src:
        canopy = src.read(1).astype("float32")
        profile = src.profile
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    canopy_mask = np.zeros_like(canopy, dtype=bool)
    if nodata is not None:
        canopy_mask |= (canopy == nodata)
    canopy_mask |= ~np.isfinite(canopy)

    # --------- LOAD POWERLINES ----------
    lines = gpd.read_file(POWERLINES_GEOJSON)
    if len(lines) == 0:
        raise ValueError("Powerlines GeoJSON contains 0 features.")
    if lines.crs != crs:
        lines = lines.to_crs(crs)

    # --------- RASTERIZE POWERLINES ----------
    shapes = [(geom, 1) for geom in lines.geometry if geom is not None]
    line_raster = rasterize(
        shapes=shapes,
        out_shape=canopy.shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )

    # --------- DISTANCE TRANSFORM ----------
    distance_pixels = distance_transform_edt(line_raster == 0)

    pixel_size_x = abs(transform.a)
    pixel_size_y = abs(transform.e)
    pixel_size = (pixel_size_x + pixel_size_y) / 2.0

    distance_m = (distance_pixels * pixel_size).astype("float32")

    # --------- SAVE DISTANCE ----------
    dist_profile = profile.copy()
    dist_profile.update(dtype="float32", count=1, nodata=None, compress="lzw")
    with rasterio.open(OUT_DIST_TIF, "w", **dist_profile) as dst:
        dst.write(distance_m, 1)

    # --------- NDVI ----------
    if USE_NDVI:
        with rasterio.open(NDVI_TIF) as nd:
            ndvi = nd.read(1).astype("float32")

            if nd.shape != canopy.shape:
                raise ValueError(f"NDVI shape {nd.shape} does not match canopy shape {canopy.shape}.")
            if nd.crs != crs:
                raise ValueError("NDVI CRS does not match canopy CRS.")
            if nd.transform != transform:
                raise ValueError("NDVI transform does not match canopy transform.")

        ndvi_mask = ~np.isfinite(ndvi)
        ndvi_norm = normalize(ndvi, nodata_mask=ndvi_mask)
    else:
        ndvi_norm = None

    # --------- NORMALIZE INPUTS ----------
    # Proximity: closer to lines => higher score
    distance_norm = normalize(distance_m)
    proximity_score = (1.0 - distance_norm).astype("float32")

    # Encroachment potential: only vegetation tall enough to violate clearance matters
    # encroach = max(0, canopy - (CLEARANCE_M + SLOPE*distance))
    encroach = canopy - (CLEARANCE_M + SLOPE * distance_m)
    encroach = np.maximum(encroach, 0.0).astype("float32")

    encroach_norm = normalize(encroach, nodata_mask=canopy_mask)

    # --------- RISK ----------
    risk = (W_ENCROACH * encroach_norm) + (W_DISTANCE * proximity_score)
    if ndvi_norm is not None:
        risk += (W_NDVI * ndvi_norm)

    risk = risk.astype("float32")
    risk[canopy_mask] = np.nan

    # --------- SAVE RISK ----------
    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1, nodata=None, compress="lzw")
    with rasterio.open(OUT_RISK_TIF, "w", **out_profile) as dst:
        dst.write(risk, 1)

    print("✅ Wrote:")
    print(" -", OUT_DIST_TIF)
    print(" -", OUT_RISK_TIF)
    print("CRS:", crs)
    print("Shape:", canopy.shape)
    print("Encroachment params:", {"CLEARANCE_M": CLEARANCE_M, "SLOPE": SLOPE})
    print("Weights:", {"encroach": W_ENCROACH, "distance": W_DISTANCE, "ndvi": W_NDVI})


if __name__ == "__main__":
    main()