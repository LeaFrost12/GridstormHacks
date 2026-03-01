import numpy as np
import rasterio
from pathlib import Path

# Automatically find first HDF file
hdf_files = list(Path("data").rglob("*.hdf"))

if not hdf_files:
    print("No HDF files found.")
    exit()

hdf_path = hdf_files[0]
print("Using HDF:", hdf_path)

subdataset = f'HDF4_EOS:EOS_GRID:"{hdf_path}":MODIS_Grid_16DAY_250m_500m_VI:250m 16 days NDVI'

with rasterio.open(subdataset) as src:
    ndvi_raw = src.read(1).astype(np.float32)
    ndvi = ndvi_raw * 0.0001

    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        compress="lzw"
    )

    with rasterio.open("ndvi_output.tif", "w", **profile) as dst:
        dst.write(ndvi, 1)

print("GeoTIFF created: ndvi_output.tif")