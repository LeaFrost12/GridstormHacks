import os
import requests

print("Starting MODIS connection test (archives endpoint)...")

TOKEN = os.environ.get("LAADS_TOKEN")
if not TOKEN:
    print("No LAADS_TOKEN found. Run: export LAADS_TOKEN='...'")
    raise SystemExit(1)

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "X-Requested-With": "XMLHttpRequest",
}

url = "https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/archives"

params = {
    "products": "MOD13Q1",               
    "collections": "61",                  
    "temporalRanges": "2024-01-01..2024-02-01",
    "regions": "[BBOX]W-84.5 N39.5 E-75.0 S32.0",
    "page": 1
}

r = requests.get(url, headers=headers, params=params, timeout=60)
print("Status Code:", r.status_code)
print("First 800 chars:\n", r.text[:800])
r.raise_for_status()