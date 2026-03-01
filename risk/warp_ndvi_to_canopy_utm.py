import os
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np

CANOPY_UTM = "outputs/canopy_utm32617.tif"
NDVI_IN = "ndvi_output.tif"
NDVI_OUT = "outputs/ndvi_on_canopy_utm.tif"

os.makedirs("outputs", exist_ok=True)

with rasterio.open(CANOPY_UTM) as c:
    dst_crs = c.crs
    dst_transform = c.transform
    dst_height, dst_width = c.height, c.width
    profile = c.profile.copy()
    profile.update(dtype="float32", nodata=None, compress="lzw", count=1)

with rasterio.open(NDVI_IN) as n:
    src = n.read(1).astype("float32")
    src_transform = n.transform
    src_crs = n.crs

dst = np.empty((dst_height, dst_width), dtype="float32")

reproject(
    source=src,
    destination=dst,
    src_transform=src_transform,
    src_crs=src_crs,
    dst_transform=dst_transform,
    dst_crs=dst_crs,
    resampling=Resampling.bilinear,
)

with rasterio.open(NDVI_OUT, "w", **profile) as out:
    out.write(dst, 1)

print("✅ Wrote", NDVI_OUT)
print("Shape:", dst.shape, "CRS:", dst_crs)