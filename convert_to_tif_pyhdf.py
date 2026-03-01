import re
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pyhdf.SD import SD, SDC
from pathlib import Path

# ---- Choose latest MOD13Q1 HDF ----
hdfs = sorted(Path("data").rglob("MOD13Q1*.hdf"))
if not hdfs:
    raise FileNotFoundError("No MOD13Q1 .hdf files found under ./data")
HDF_PATH = str(hdfs[-1])
OUT_TIF = "NDVI/ndvi_output.tif"
print("Using HDF:", HDF_PATH)

# ---- MODIS Sinusoidal tiling constants (MODIS Land) ----
R = 6371007.181  # meters
TILE_SIZE_M = 1111950.5196666666  # 10 degrees at equator in sinusoidal grid
PIXELS_250M = 4800                # MOD13Q1 250m tile is 4800 x 4800
PIXEL_SIZE = TILE_SIZE_M / PIXELS_250M  # ~231.656 m

# Global upper-left of the MODIS sinusoidal grid
X_MIN = -20015109.354  # meters
Y_MAX =  10007554.677  # meters

def hv_from_filename(path: str):
    m = re.search(r"\.h(\d{2})v(\d{2})\.", path)
    if not m:
        raise ValueError("Could not parse h/v from filename")
    return int(m.group(1)), int(m.group(2))

def main():
    h, v = hv_from_filename(HDF_PATH)
    print("Parsed tile:", f"h{h:02d}v{v:02d}")

    # Compute tile upper-left in MODIS sinusoidal meters
    ulx = X_MIN + h * TILE_SIZE_M
    uly = Y_MAX - v * TILE_SIZE_M

    transform = from_origin(ulx, uly, PIXEL_SIZE, PIXEL_SIZE)

    # CRS: MODIS Sinusoidal (spherical)
    crs = "+proj=sinu +R=6371007.181 +nadgrids=@null +wktext +units=m +no_defs"

    # Read NDVI
    hdf = SD(HDF_PATH, SDC.READ)
    ndvi_ds = hdf.select("250m 16 days NDVI")
    raw = ndvi_ds.get().astype(np.float32)

    # Scale factor
    ndvi = raw * 0.0001

    # Replace fill values (typical MODIS fill is -3000 or -2000 range)
    ndvi[raw <= -2000] = np.nan

    height, width = ndvi.shape
    if (height, width) != (PIXELS_250M, PIXELS_250M):
        print("WARNING: Unexpected shape:", (height, width))

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
        "nodata": None,
    }

    with rasterio.open(OUT_TIF, "w", **profile) as dst:
        dst.write(ndvi.astype(np.float32), 1)

    print("✅ Wrote GeoTIFF:", OUT_TIF)
    print("UL corner (m):", (ulx, uly))
    print("Pixel size (m):", PIXEL_SIZE)

if __name__ == "__main__":
    main()
