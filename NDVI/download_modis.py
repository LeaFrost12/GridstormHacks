import earthaccess

# Login to NASA Earthdata
earthaccess.login()

# Search MOD13Q1 for a date range
results = earthaccess.search_data(
    short_name="MOD13Q1",
    version="061",
    temporal=("2024-01-01", "2024-01-16"),
    cloud_hosted=True,
)

print("Total results found:", len(results))

# Filter ONLY South Carolina tile (h11v05)
sc_tile = [r for r in results if "h11v05" in str(r)]

print("South Carolina tile results:", len(sc_tile))

if len(sc_tile) == 0:
    print("No h11v05 tile found for this date range.")
else:
    print("Downloading South Carolina NDVI tile...")
    files = earthaccess.download(sc_tile[:1])
    print("Downloaded:")
    print(files)