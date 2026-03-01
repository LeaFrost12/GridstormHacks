import numpy as np
import rasterio

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


# -----------------------------
# Helpers
# -----------------------------
def read_raster(path: str) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        meta = {
            "crs": src.crs,
            "shape": (src.height, src.width),
            "transform": src.transform,
            "nodata": src.nodata,
        }
    return arr, meta


def check_align(meta_base: dict, meta_other: dict, name: str):
    assert meta_other["crs"] == meta_base["crs"], f"{name} CRS mismatch"
    assert meta_other["shape"] == meta_base["shape"], f"{name} shape mismatch"
    assert meta_other["transform"] == meta_base["transform"], f"{name} transform mismatch"


# -----------------------------
# Main
# -----------------------------
def main():
    # ✅ Use the aligned/UTM rasters (all 788x706 EPSG:32617)
    canopy_path = "outputs/canopy_utm32617.tif"
    dist_path   = "outputs/distance_to_powerlines_m.tif"
    ndvi_path   = "outputs/ndvi_on_canopy_utm.tif"
    risk_path   = "outputs/risk_map.tif"

    canopy, meta_c = read_raster(canopy_path)
    dist,   meta_d = read_raster(dist_path)
    ndvi,   meta_n = read_raster(ndvi_path)
    risk,   meta_r = read_raster(risk_path)

    # Hard alignment checks
    check_align(meta_c, meta_d, "distance")
    check_align(meta_c, meta_n, "ndvi")
    check_align(meta_c, meta_r, "risk")
    print("✅ Inputs aligned:", meta_c["crs"], meta_c["shape"])

    # Encroachment params (must match build_risk.py)
    CLEARANCE_M = 6.0
    SLOPE = 0.15

    # Encroachment (meters)
    encroach = np.maximum(canopy - (CLEARANCE_M + SLOPE * dist), 0.0).astype("float32")

    # -----------------------------
    # Labels = "High Risk" using your rule
    # -----------------------------
    p95 = np.nanpercentile(risk[np.isfinite(risk)], 95)

    y = ((dist <= 200) & ((encroach > 0) | (risk >= p95))).astype("int8")

    # -----------------------------
    # Feature matrix
    # -----------------------------
    mask = (
        np.isfinite(canopy) &
        np.isfinite(dist) &
        np.isfinite(ndvi) &
        np.isfinite(risk)
    )

    X = np.column_stack([
        dist[mask],
        encroach[mask],
        ndvi[mask],
        canopy[mask],
    ]).astype("float32")

    y = y[mask]

    # Safety: if label is too imbalanced or empty
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    print(f"Samples: total={y.size} pos={pos} neg={neg}")
    if pos == 0 or neg == 0:
        raise RuntimeError("Label has only one class. Loosen thresholds or check inputs.")

    # -----------------------------
    # Train/test split + model
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    print("ROC AUC:", round(float(roc_auc_score(y_test, y_prob)), 4))

    print("\nModel Coefficients (direction of influence):")
    for name, coef in zip(["dist", "encroach", "ndvi", "canopy"], model.coef_[0]):
        print(f"  {name:10s} {coef:+.4f}")

    # -----------------------------
    # Optional: export probability raster
    # -----------------------------
    prob = np.full(canopy.shape, np.nan, dtype="float32")
    prob_vals = model.predict_proba(X)[:, 1].astype("float32")
    prob[mask] = prob_vals

    out_path = "outputs/pred_highrisk_prob.tif"
    with rasterio.open(risk_path) as src:
        profile = src.profile.copy()
    profile.update(dtype="float32", count=1, nodata=None, compress="lzw")

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(prob, 1)

    print("\n✅ Wrote probability map:", out_path)


if __name__ == "__main__":
    main()