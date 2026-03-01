import os
import numpy as np
import rasterio

RISK_TIF = "outputs/risk_map.tif"
DIST_TIF = "outputs/distance_to_powerlines_m.tif"
OUT_HOTSPOT_TIF = "outputs/hotspots_risk_gt_0p75_within_30m.tif"

RISK_THRESH = 0.75
DIST_THRESH_M = 30

os.makedirs("outputs", exist_ok=True)

with rasterio.open(RISK_TIF) as rsrc, rasterio.open(DIST_TIF) as dsrc:
    risk = rsrc.read(1).astype("float32")
    dist = dsrc.read(1).astype("float32")

    assert rsrc.crs == dsrc.crs
    assert rsrc.transform == dsrc.transform
    assert risk.shape == dist.shape

    mask = np.isfinite(risk) & np.isfinite(dist)

    hotspots = (mask & (risk >= RISK_THRESH) & (dist <= DIST_THRESH_M)).astype("uint8")

    # pixel area in m^2 (UTM)
    px_w = abs(rsrc.transform.a)
    px_h = abs(rsrc.transform.e)
    pixel_area_m2 = px_w * px_h

    hotspot_pixels = int(hotspots.sum())
    hotspot_area_m2 = hotspot_pixels * pixel_area_m2
    hotspot_area_km2 = hotspot_area_m2 / 1e6

    profile = rsrc.profile.copy()
    profile.update(dtype="uint8", count=1, nodata=0, compress="lzw")

    with rasterio.open(OUT_HOTSPOT_TIF, "w", **profile) as out:
        out.write(hotspots, 1)

print("✅ Wrote:", OUT_HOTSPOT_TIF)
print("Hotspot pixels:", hotspot_pixels)
print("Hotspot area (km^2):", hotspot_area_km2)