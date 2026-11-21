"""Step 2: Clean messy_weather.csv -> cleaned_weather.csv"""
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "messy_weather.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_weather.csv"

df = pd.read_csv(RAW_DATA_PATH)
print("Raw shape:", df.shape)
print("Missing values before:\n", df.isna().sum(), "\n")

# 1. Remove exact duplicate rows
df = df.drop_duplicates()

# 2. Parse mixed date formats (try each known format)
def parse_date(s):
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%b %d, %Y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.NaT

df["date"] = df["date"].astype(str).apply(parse_date)
df = df.dropna(subset=["date"])

# 3. Standardise city names
df["city"] = "Delhi"

# 4. Strip units/text, convert to numbers ("N/A", "--" become NaN)
for col in ["temperature", "humidity", "pressure", "wind_speed", "rainfall"]:
    df[col] = (df[col].astype(str)
                      .str.replace(r"[^0-9.\-]", "", regex=True)
                      .replace("", np.nan))
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 5. Replace impossible values with NaN (physical limits)
limits = {"temperature": (-10, 55), "humidity": (0, 100),
          "pressure": (950, 1060), "wind_speed": (0, 150), "rainfall": (0, 500)}
for col, (lo, hi) in limits.items():
    df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

# 6. Detect statistical outliers using the 1.5 x IQR rule
for col in ["temperature", "humidity", "pressure", "wind_speed", "rainfall"]:
    values = df[col].dropna()
    if col == "rainfall":
        values = values[values > 0]
    if len(values) < 4:
        continue

    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = (df[col] < lower) | (df[col] > upper)
    outlier_count = int(outliers.sum())

    if outlier_count:
        if col == "rainfall":
            in_range = values[(values >= lower) & (values <= upper)]
            df.loc[outliers, col] = in_range.median()
        else:
            df.loc[outliers, col] = np.nan
        print(f"Replaced {outlier_count} {col} outlier(s)")

# 7. One row per day, sorted by date, then fill the gaps
df = df.groupby("date", as_index=False).first().sort_values("date")
df = df.set_index("date").asfreq("D")           # make sure no day is missing
df["city"] = "Delhi"
num_cols = ["temperature", "humidity", "pressure", "wind_speed"]
df[num_cols] = df[num_cols].interpolate(method="time").bfill().ffill()
df["rainfall"] = df["rainfall"].fillna(0)       # no record = no rain

df = df.round(1).reset_index()
print("Missing values after:\n", df.isna().sum(), "\n")
df.to_csv(PROCESSED_DATA_PATH, index=False)
print(f"{PROCESSED_DATA_PATH.name} created:", df.shape)
print(df.describe().round(2))
