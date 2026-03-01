import os
import numpy as np
import rasterio

RISK_TIF = "outputs/risk_map.tif"
DIST_TIF = "outputs/distance_to_powerlines_m.tif"
OUT_TIF = "outputs/hotspots_top5pct_within_100m.tif"

TOP_PCT = 95          # top 5%
DIST_THRESH_M = 100   # 3-4 pixels at your resolution

os.makedirs("outputs", exist_ok=True)

with rasterio.open(RISK_TIF) as rsrc, rasterio.open(DIST_TIF) as dsrc:
    risk = rsrc.read(1).astype("float32")
    dist = dsrc.read(1).astype("float32")

    mask = np.isfinite(risk) & np.isfinite(dist)

    # threshold = percentile of all valid risk pixels
    thr = float(np.percentile(risk[mask], TOP_PCT))

    hotspots = (mask & (risk >= thr) & (dist <= DIST_THRESH_M)).astype("uint8")

    px_w = abs(rsrc.transform.a)
    px_h = abs(rsrc.transform.e)
    area_km2 = float(hotspots.sum() * (px_w * px_h) / 1e6)

    profile = rsrc.profile.copy()
    profile.update(dtype="uint8", count=1, nodata=0, compress="lzw")

    with rasterio.open(OUT_TIF, "w", **profile) as out:
        out.write(hotspots, 1)

print("✅ Wrote:", OUT_TIF)
print("Risk threshold (percentile):", TOP_PCT, "=>", thr)
print("Corridor distance (m):", DIST_THRESH_M)
print("Hotspot pixels:", int(hotspots.sum()))
print("Hotspot area (km^2):", area_km2)