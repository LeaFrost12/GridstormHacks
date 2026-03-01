import rasterio
import matplotlib.pyplot as plt
import numpy as np

paths = [
    ("Distance to power lines (m)", "outputs/distance_to_powerlines_m.tif"),
    ("Risk map (0-1)", "outputs/risk_map.tif"),
]

for title, path in paths:
    with rasterio.open(path) as src:
        a = src.read(1).astype("float32")
        a = np.where(np.isfinite(a), a, np.nan)

    plt.figure()
    plt.imshow(a)
    plt.title(title)
    plt.colorbar()
    plt.show()