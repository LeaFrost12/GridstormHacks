import os
import json
import glob
import subprocess
from datetime import datetime
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

# ----------------------------
# SETTINGS (edit these)
# ----------------------------

# Your AOI raster to match (canopy in UTM)
AOI_TIF = "outputs/canopy_utm32617.tif"

# Folder where you store MODIS HDF downloads per date
# Example you already have: NDVI/data/2026-02-28-d2bbd4/*.hdf
MODIS_HDF_ROOT = "NDVI/data"

# Output
OUT_DIR = "outputs/ndvi_10yr"
OUT_CSV = "outputs/ndvi_10yr/ndvi_timeseries.csv"

# Years you want
START_YEAR = 2016
END_YEAR   = 2025   # inclusive (10 years)

# ----------------------------
# HELPERS
# ----------------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def list_hdfs():
    # any .hdf under NDVI/data/**
    return sorted(glob.glob(os.path.join(MODIS_HDF_ROOT, "**", "*.hdf"), recursive=True))

def parse_date_from_filename(fname):
    # MOD13Q1.AYYYYDDD.hXXvYY...hdf
    # returns YYYYDDD as int
    base = os.path.basename(fname)
    # find "A" then 7 digits
    # ex: MOD13Q1.A2023353...
    try:
        a = base.split(".A")[1]
        yyyyddd = a[:7]
        year = int(yyyyddd[:4])
        ddd = int(yyyyddd[4:])
        return year, ddd
    except Exception:
        return None

def ddd_to_date(year, ddd):
    # day-of-year -> YYYY-MM-DD
    return datetime.strptime(f"{year}-{ddd:03d}", "%Y-%j").date()

def hdf_to_ndvi_tif(hdf_path, out_tif):
    # Uses your existing converter script if you have it.
    # You already have NDVI/modis/hdf_to_ndvi_tif.py in your repo.
    cmd = ["python3", "NDVI/modis/hdf_to_ndvi_tif.py", hdf_path, out_tif]
    subprocess.run(cmd, check=True)

def align_to_aoi(src_tif, dst_tif, aoi_profile):
    with rasterio.open(src_tif) as src:
        src_data = src.read(1).astype("float32")
        src_profile = src.profile

        dst = np.full((aoi_profile["height"], aoi_profile["width"]), np.nan, dtype="float32")

        reproject(
            source=src_data,
            destination=dst,
            src_transform=src_profile["transform"],
            src_crs=src_profile["crs"],
            dst_transform=aoi_profile["transform"],
            dst_crs=aoi_profile["crs"],
            resampling=Resampling.bilinear,
        )

    prof = aoi_profile.copy()
    prof.update(dtype="float32", count=1, nodata=None, compress="lzw")

    with rasterio.open(dst_tif, "w", **prof) as out:
        out.write(dst, 1)

def summarize_ndvi(ndvi_arr, aoi_mask=None):
    a = ndvi_arr.copy()
    if aoi_mask is not None:
        a = np.where(aoi_mask, a, np.nan)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None

    return {
        "count": int(a.size),
        "mean": float(np.mean(a)),
        "p10": float(np.percentile(a, 10)),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
    }

def main():
    ensure_dir(OUT_DIR)

    # Load AOI raster template
    with rasterio.open(AOI_TIF) as aoi:
        aoi_profile = aoi.profile
        aoi_data = aoi.read(1).astype("float32")
        aoi_mask = np.isfinite(aoi_data)

    # Find all HDFs you have locally
    hdfs = list_hdfs()

    # Filter to only the 10-year range
    keep = []
    for h in hdfs:
        parsed = parse_date_from_filename(h)
        if not parsed:
            continue
        year, ddd = parsed
        if START_YEAR <= year <= END_YEAR:
            keep.append((h, year, ddd))

    keep.sort(key=lambda x: (x[1], x[2]))

    if not keep:
        print("No HDF files found in range. You need to download the 10 years first.")
        return

    rows = []
    for hdf_path, year, ddd in keep:
        date = ddd_to_date(year, ddd)
        stamp = f"{date}"

        raw_tif = os.path.join(OUT_DIR, f"ndvi_raw_{stamp}.tif")
        aligned_tif = os.path.join(OUT_DIR, f"ndvi_aligned_{stamp}.tif")

        # Convert HDF -> NDVI tif (raw)
        if not os.path.exists(raw_tif):
            print("Converting:", os.path.basename(hdf_path))
            hdf_to_ndvi_tif(hdf_path, raw_tif)

        # Align to AOI grid
        if not os.path.exists(aligned_tif):
            align_to_aoi(raw_tif, aligned_tif, aoi_profile)

        # Summarize
        with rasterio.open(aligned_tif) as src:
            nd = src.read(1).astype("float32")
        s = summarize_ndvi(nd, aoi_mask=aoi_mask)
        if s is None:
            continue

        rows.append({
            "date": str(date),
            "year": int(year),
            **s
        })

    # Write CSV
    ensure_dir(os.path.dirname(OUT_CSV))
    with open(OUT_CSV, "w") as f:
        f.write("date,year,count,mean,p10,p50,p90\n")
        for r in rows:
            f.write(f"{r['date']},{r['year']},{r['count']},{r['mean']},{r['p10']},{r['p50']},{r['p90']}\n")

    print("✅ Wrote:", OUT_CSV)
    print("Rows:", len(rows))

if __name__ == "__main__":
    main()