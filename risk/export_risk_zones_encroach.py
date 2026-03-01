import os, json
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape as shapely_shape

# -----------------------------
# Inputs
# -----------------------------
RISK_TIF = "outputs/risk_map.tif"
DIST_TIF = "outputs/distance_to_powerlines_m.tif"
CANOPY_TIF = "outputs/canopy_utm32617.tif"

# -----------------------------
# Outputs
# -----------------------------
OUT_CLASS_TIF = "outputs/risk_classes.tif"     # 0 safe, 1 concern, 2 high, 255 nodata
OUT_GEOJSON   = "outputs/risk_zones.geojson"   # dissolved polygons
OUT_SUMMARY   = "outputs/risk_summary.json"
OUT_LEGEND    = "outputs/risk_legend.txt"

# -----------------------------
# Rule parameters
# -----------------------------
HIGH_DIST_M = 200
CONCERN_DIST_M = 300
P95 = 95
P75 = 75

# Encroachment parameters (must match build_risk.py)
CLEARANCE_M = 6.0
SLOPE = 0.15

# Simplification (meters; EPSG:32617 is meters)
SIMPLIFY_M = 20.0   # try 10–50

COLORS = {
    2: {"label": "HIGH",    "hex": "#E53935"},  # red
    1: {"label": "CONCERN", "hex": "#FB8C00"},  # orange
    0: {"label": "SAFE",    "hex": "#43A047"},  # green
}

def main():
    os.makedirs("outputs", exist_ok=True)

    with rasterio.open(RISK_TIF) as rsrc, rasterio.open(DIST_TIF) as dsrc, rasterio.open(CANOPY_TIF) as csrc:
        risk = rsrc.read(1).astype("float32")
        dist = dsrc.read(1).astype("float32")
        canopy = csrc.read(1).astype("float32")

        # Alignment checks
        assert rsrc.crs == dsrc.crs == csrc.crs, "CRS mismatch"
        assert rsrc.transform == dsrc.transform == csrc.transform, "Transform mismatch"
        assert risk.shape == dist.shape == canopy.shape, "Shape mismatch"

        crs = rsrc.crs
        transform = rsrc.transform

        valid = np.isfinite(risk) & np.isfinite(dist) & np.isfinite(canopy)

        thr95 = float(np.percentile(risk[valid], P95))
        thr75 = float(np.percentile(risk[valid], P75))

        # Encroachment
        encroach = np.maximum(canopy - (CLEARANCE_M + SLOPE * dist), 0.0).astype("float32")

        # Class raster
        cls = np.zeros(risk.shape, dtype="uint8")
        cls[~valid] = 255

        high_mask = valid & (dist <= HIGH_DIST_M) & ((encroach > 0) | (risk >= thr95))
        concern_mask = valid & (~high_mask) & (dist <= CONCERN_DIST_M) & (risk >= thr75)

        cls[concern_mask] = 1
        cls[high_mask] = 2

        # Save classified raster
        profile = rsrc.profile.copy()
        profile.update(dtype="uint8", count=1, nodata=255, compress="lzw")
        with rasterio.open(OUT_CLASS_TIF, "w", **profile) as out:
            out.write(cls, 1)

        # Area stats
        px_w = abs(transform.a)
        px_h = abs(transform.e)
        pixel_area_km2 = float((px_w * px_h) / 1e6)

        counts = {
            "SAFE": int(np.sum(cls == 0)),
            "CONCERN": int(np.sum(cls == 1)),
            "HIGH": int(np.sum(cls == 2)),
        }
        areas_km2 = {k: v * pixel_area_km2 for k, v in counts.items()}

        # Polygonize each class separately, then dissolve to 1 geometry per class
        records = []
        for geom_mapping, val in shapes(cls, mask=(cls != 255), transform=transform):
            v = int(val)
            geom = shapely_shape(geom_mapping)  # ✅ make shapely geometry
            if geom.is_empty:
                continue
            records.append({
                "geometry": geom,
                "class_id": v,
            })

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)

    # Dissolve into one polygon per class_id
    dissolved = gdf.dissolve(by="class_id", as_index=False)

    # Fix geometry & simplify
    dissolved["geometry"] = dissolved["geometry"].buffer(0)
    if SIMPLIFY_M and SIMPLIFY_M > 0:
        dissolved["geometry"] = dissolved["geometry"].simplify(SIMPLIFY_M, preserve_topology=True)

    # Add class labels/colors
    dissolved["class"] = dissolved["class_id"].map(lambda i: COLORS[int(i)]["label"])
    dissolved["color"] = dissolved["class_id"].map(lambda i: COLORS[int(i)]["hex"])

    # Write GeoJSON
    dissolved.to_file(OUT_GEOJSON, driver="GeoJSON")

    # Write summary JSON
    summary = {
        "inputs": {"risk": RISK_TIF, "distance": DIST_TIF, "canopy": CANOPY_TIF},
        "crs": str(crs),
        "rules": {
            "HIGH": f"dist <= {HIGH_DIST_M}m AND (encroach > 0 OR risk >= p{P95})",
            "CONCERN": f"dist <= {CONCERN_DIST_M}m AND risk >= p{P75} AND not HIGH",
            "SAFE": "everything else",
        },
        "thresholds": {
            "risk_p95": thr95,
            "risk_p75": thr75,
            "high_distance_m": HIGH_DIST_M,
            "concern_distance_m": CONCERN_DIST_M,
            "encroach_params": {"CLEARANCE_M": CLEARANCE_M, "SLOPE": SLOPE},
            "simplify_m": SIMPLIFY_M,
        },
        "classes": {
            "SAFE": {"id": 0, "color": COLORS[0]["hex"], "pixels": counts["SAFE"], "area_km2": areas_km2["SAFE"]},
            "CONCERN": {"id": 1, "color": COLORS[1]["hex"], "pixels": counts["CONCERN"], "area_km2": areas_km2["CONCERN"]},
            "HIGH": {"id": 2, "color": COLORS[2]["hex"], "pixels": counts["HIGH"], "area_km2": areas_km2["HIGH"]},
        },
    }
    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)

    # Legend text
    legend = (
        "RISK ZONES LEGEND\n"
        f"SAFE (0)    {COLORS[0]['hex']}  default / lower priority\n"
        f"CONCERN (1) {COLORS[1]['hex']}  dist <= {CONCERN_DIST_M}m AND risk >= p{P75}\n"
        f"HIGH (2)    {COLORS[2]['hex']}  dist <= {HIGH_DIST_M}m AND (encroach > 0 OR risk >= p{P95})\n"
        f"\nEncroachment = max(0, canopy - (CLEARANCE + SLOPE*distance))\n"
        f"CLEARANCE={CLEARANCE_M}m, SLOPE={SLOPE}\n"
        f"Simplify={SIMPLIFY_M}m\n"
    )
    with open(OUT_LEGEND, "w") as f:
        f.write(legend)

    print("✅ Wrote (dissolved):")
    print(" -", OUT_CLASS_TIF)
    print(" -", OUT_GEOJSON)
    print(" -", OUT_SUMMARY)
    print(" -", OUT_LEGEND)

    print("\nThresholds:")
    print(" - risk p95 =", thr95)
    print(" - risk p75 =", thr75)
    print("Areas (km^2):", areas_km2)


if __name__ == "__main__":
    main()