
import requests
import pandas as pd
import numpy as np
import datetime
import json

TOKEN = "qDMWPIYGHTKRKkhCWbKjwkUQxNMinWWd"
headers = {"token": TOKEN}

end_date = datetime.datetime.now().date()
start_date = end_date - pd.Timedelta(days=365)

all_results = []
offset = 1

while True:
    params = {
        "datasetid": "GHCND",
        "locationid": "FIPS:45079",  # Columbia, SC (Richland County)
        "startdate": start_date.isoformat(),
        "enddate": end_date.isoformat(),
        "datatypeid": "PRCP",
        "limit": 1000,
        "offset": offset,
        "units": "metric"
    }

    response = requests.get(
        "https://www.ncei.noaa.gov/cdo-web/api/v2/data",
        headers=headers,
        params=params,
        timeout=10
    )

    batch = response.json().get("results", [])
    if not batch:
        break

    all_results.extend(batch)
    print(f"Fetched {len(all_results)} records so far...")

    # If we got less than 1000, we've hit the last page
    if len(batch) < 1000:
        break

    offset += 1000

print(f"Total records fetched: {len(all_results)}")

df = pd.DataFrame(all_results)

if df.empty:
    print("No data returned for this date range")
else:
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = df["value"] / 10  # tenths of mm → mm

    daily_rain = df.groupby("date")["value"].sum().reset_index()
    daily_rain = daily_rain.sort_values("date").reset_index(drop=True)

    rain_values = daily_rain["value"]
    total_rain = rain_values.sum()

    entries = []

    for i, row in daily_rain.iterrows():
        current_date = row["date"]
        current_mm = row["value"]

        # 30-day rolling rainfall up to this day
        window_start_30 = current_date - pd.Timedelta(days=29)
        rain_30 = daily_rain[
            (daily_rain["date"] >= window_start_30) &
            (daily_rain["date"] <= current_date)
        ]["value"].sum()

        # Cumulative heavy rain days up to this date
        heavy_rain_days_cumulative = int(daily_rain[
            (daily_rain["date"] <= current_date) & (daily_rain["value"] > 25)
        ].shape[0])

        # 60-day anomaly: recent 60 avg vs previous 60 avg
        recent_start = current_date - pd.Timedelta(days=59)
        prev_start = current_date - pd.Timedelta(days=119)
        prev_end = current_date - pd.Timedelta(days=60)

        recent_60_vals = daily_rain[
            (daily_rain["date"] >= recent_start) & (daily_rain["date"] <= current_date)
        ]["value"]
        prev_60_vals = daily_rain[
            (daily_rain["date"] >= prev_start) & (daily_rain["date"] <= prev_end)
        ]["value"]

        recent_60 = recent_60_vals.mean() if not recent_60_vals.empty else 0
        previous_60 = prev_60_vals.mean() if not prev_60_vals.empty else 0
        anomaly = round(float(recent_60 - previous_60), 2)

        # Seasonal growth factor based on month
        month = current_date.month
        if month in [3, 4, 5]:
            seasonal = 1.2
        elif month in [6, 7, 8]:
            seasonal = 1.1
        elif month in [9, 10, 11]:
            seasonal = 1.0
        else:
            seasonal = 0.7

        # Weather stress score
        rain_norm = rain_30 / total_rain if total_rain > 0 else 0
        storm_norm = heavy_rain_days_cumulative / len(daily_rain) if len(daily_rain) > 0 else 0

        val_max = rain_values.max()
        val_min = rain_values.min()
        anomaly_norm = (anomaly - val_min) / (val_max - val_min) if val_max != val_min else 0

        weather_stress = round(0.5 * rain_norm + 0.3 * storm_norm + 0.2 * float(anomaly_norm), 4)

        entries.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "location": "Columbia, SC",
            "rainfall_mm": round(float(current_mm), 2),
            "heavy_rain_days": heavy_rain_days_cumulative,
            "anomaly_60d_mm": anomaly,
            "seasonal_growth_factor": seasonal,
            "weather_stress_score": weather_stress
        })

    output = {
        "location": "Columbia, SC",
        "units": "millimeters",
        "days": entries
    }

    output_path = "/Users/eharshyne24/hackathon/GridstormHacks-/weather/sc_weather.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"✅ Saved {len(entries)} daily entries to {output_path}")
    print("\nSample (first 3 entries):")
    print(json.dumps(entries[:3], indent=2))
