import rasterio

hdf_path = "data/2026-02-28-d2bbd4/MOD13Q1.A2023353.h17v06.061.2024005131728.hdf"

with rasterio.open(hdf_path) as src:
    print("Subdatasets:")
    for s in src.subdatasets:
        print(s)