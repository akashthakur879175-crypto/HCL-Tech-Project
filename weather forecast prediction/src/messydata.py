"""Step 1: Generate a realistic MESSY weather dataset (messy_weather.csv)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 730  # two years of daily data
dates = pd.date_range("2023-01-01", periods=n, freq="D")
doy = dates.day_of_year.values

# --- Realistic underlying weather (seasonal + noise) ---
temp = 25 + 10 * np.sin(2 * np.pi * (doy - 100) / 365) + rng.normal(0, 1.8, n)
humidity = 60 + 20 * np.sin(2 * np.pi * (doy - 190) / 365) + rng.normal(0, 6, n)
pressure = 1012 - 0.3 * (temp - 25) + rng.normal(0, 2, n)
wind = np.abs(8 + rng.normal(0, 3, n))
rain = np.where(rng.random(n) < (humidity - 40) / 120, rng.gamma(2, 4, n), 0.0)

df = pd.DataFrame({
    "date": dates, "city": "Delhi",
    "temperature": temp.round(1), "humidity": humidity.round(1),
    "pressure": pressure.round(1), "wind_speed": wind.round(1),
    "rainfall": rain.round(1),
})

# --- Make it MESSY ---
df["date"] = df["date"].dt.strftime("%Y-%m-%d")
idx = rng.choice(n, 120, replace=False)                      # mixed date formats
df.loc[idx[:60], "date"] = pd.to_datetime(df.loc[idx[:60], "date"]).dt.strftime("%d/%m/%Y")
df.loc[idx[60:], "date"] = pd.to_datetime(df.loc[idx[60:], "date"]).dt.strftime("%b %d, %Y")

df["city"] = rng.choice(["Delhi", "delhi", " DELHI ", "New Delhi", "Delhi "], n)  # inconsistent text
df["temperature"] = df["temperature"].astype(object)
t_idx = rng.choice(n, 80, replace=False)                      # units mixed into text
df.loc[t_idx, "temperature"] = df.loc[t_idx, "temperature"].astype(str) + "°C"
df = df.astype({"humidity": object, "wind_speed": object})
w_idx = rng.choice(n, 40, replace=False)
df.loc[w_idx, "wind_speed"] = df.loc[w_idx, "wind_speed"].astype(str) + " km/h"

for col, k in [("temperature", 50), ("humidity", 60), ("pressure", 45),
               ("wind_speed", 40), ("rainfall", 35)]:         # missing values
    df.loc[rng.choice(n, k, replace=False), col] = np.nan
df.loc[rng.choice(n, 10, replace=False), "temperature"] = "N/A"
df.loc[rng.choice(n, 8, replace=False), "humidity"] = "--"

df.loc[rng.choice(n, 6, replace=False), "temperature"] = 999  # impossible sensor errors
df.loc[rng.choice(n, 6, replace=False), "humidity"] = -50
df.loc[rng.choice(n, 5, replace=False), "humidity"] = 180
df.loc[rng.choice(n, 5, replace=False), "pressure"] = 0
df.loc[rng.choice(n, 5, replace=False), "rainfall"] = -9

df = pd.concat([df, df.sample(25, random_state=1)])           # duplicate rows
df = df.sample(frac=1, random_state=7).reset_index(drop=True) # shuffle order

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

raw_output = RAW_DATA_DIR / "messy_weather.csv"
df.to_csv(raw_output, index=False)
print(f"{raw_output.name} created:", df.shape)
