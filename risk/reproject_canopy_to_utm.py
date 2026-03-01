import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np

IN_TIF = "height/data/columbia_canopy_height.tif"
OUT_TIF = "outputs/canopy_utm32617.tif"
DST_CRS = "EPSG:32617"

with rasterio.open(IN_TIF) as src:
    transform, width, height = calculate_default_transform(
        src.crs, DST_CRS, src.width, src.height, *src.bounds
    )
    profile = src.profile.copy()
    profile.update(
        crs=DST_CRS,
        transform=transform,
        width=width,
        height=height,
        dtype="float32",
        nodata=None,
        compress="lzw",
        count=1,
    )

    dst = np.empty((height, width), dtype="float32")

    reproject(
        source=rasterio.band(src, 1),
        destination=dst,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=transform,
        dst_crs=DST_CRS,
        resampling=Resampling.bilinear,
    )

with rasterio.open(OUT_TIF, "w", **profile) as out:
    out.write(dst, 1)

print("✅ Wrote", OUT_TIF, "shape:", dst.shape, "crs:", DST_CRS)