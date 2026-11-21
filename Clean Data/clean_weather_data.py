import re
import numpy as np
import pandas as pd


def parse_mixed_date(val):
    """Parse multiple date string formats safely into pandas Timestamp."""
    if pd.isna(val):
        return pd.NaT
    s = str(val).strip()
    if s == "" or s.lower() in {"n/a", "na", "null", "unknown", "missing"}:
        return pd.NaT
    formats = ["%Y-%m-%d", "%d/%m/%Y", "%d-%b-%Y", "%Y/%m/%d", "%d.%m.%Y"]
    for fmt in formats:
        try:
            return pd.to_datetime(s, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.to_datetime(s, errors="coerce")


def extract_numeric(val):
    """Extract numeric float value from strings with units or return NaN."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s in {"", "n/a", "na", "null", "missing", "?", "-", "--", "unknown"}:
        return np.nan
    if s == "none":
        return 0.0
    if s == "trace":
        return 0.1  # Standard meteorological convention for trace rainfall
    match = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(match.group()) if match else np.nan


def clean_weather_dataset(
    input_path="weather_forecast_messy.csv",
    output_path="weather_forecast_clean.csv",
):
    # 1. Load raw messy dataset
    df = pd.read_csv(input_path, dtype=str, keep_default_na=True)
    initial_rows = len(df)

    # 2. Standardize column names to clean snake_case
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[()?]", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
        .str.replace("/", "_per_", regex=False)
    )
    df = df.rename(
        columns={
            "city_name": "city",
            "temperature_c": "temperature_c",
            "humidity_%": "humidity_pct",
            "wind_speed_km_per_h": "wind_speed_kmh",
            "pressure_hpa": "pressure_hpa",
            "rainfall_mm": "rainfall_mm",
            "cloud_cover_%": "cloud_cover_pct",
            "rain_tomorrow": "rain_tomorrow",
        }
    )

    # 3. Remove exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # 4. Parse and standardize dates; drop rows with invalid/missing dates
    df["date"] = df["date"].apply(parse_mixed_date)
    df = df.dropna(subset=["date"])

    # 5. Clean and standardize city names
    city_map = {
        "mumbai": "Mumbai",
        "bombay": "Mumbai",
        "mumbay": "Mumbai",
        "delhi": "Delhi",
        "new delhi": "Delhi",
        "delhii": "Delhi",
        "bengaluru": "Bengaluru",
        "bangalore": "Bengaluru",
        "chennai": "Chennai",
        "madras": "Chennai",
        "kolkata": "Kolkata",
        "calcutta": "Kolkata",
    }
    df["city"] = (
        df["city"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(city_map)
    )
    # Fill missing city with the most frequent city (mode)
    df["city"] = df["city"].fillna(df["city"].mode()[0])

    # 6. Extract numeric values from columns with mixed units/strings
    numeric_cols = [
        "temperature_c",
        "humidity_pct",
        "wind_speed_kmh",
        "pressure_hpa",
        "rainfall_mm",
        "cloud_cover_pct",
    ]
    for col in numeric_cols:
        df[col] = df[col].apply(extract_numeric)

    # 7. Replace physically impossible sensor outliers with NaN
    valid_ranges = {
        "temperature_c": (-10.0, 55.0),
        "humidity_pct": (0.0, 100.0),
        "wind_speed_kmh": (0.0, 150.0),
        "pressure_hpa": (950.0, 1060.0),
        "rainfall_mm": (0.0, 300.0),
        "cloud_cover_pct": (0.0, 100.0),
    }
    for col, (low, high) in valid_ranges.items():
        df.loc[(df[col] < low) | (df[col] > high), col] = np.nan

    # 8. Impute missing numeric values using city-level median (fallback to global median)
    for col in numeric_cols:
        city_medians = df.groupby("city")[col].transform("median")
        df[col] = df[col].fillna(city_medians).fillna(df[col].median())
        df[col] = df[col].round(1)

    # 9. Clean and standardize weather_condition categories
    condition_map = {
        "sunny": "Sunny",
        "clear": "Sunny",
        "suny": "Sunny",
        "rainy": "Rainy",
        "rain": "Rainy",
        "cloudy": "Cloudy",
        "partly cloudy": "Cloudy",
        "clody": "Cloudy",
        "stormy": "Stormy",
        "thunderstorm": "Stormy",
        "storm": "Stormy",
        "foggy": "Foggy",
        "fog": "Foggy",
        "misty": "Foggy",
    }
    df["weather_condition"] = (
        df["weather_condition"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(condition_map)
    )
    # Impute missing weather_condition using logical weather rules
    missing_cond = df["weather_condition"].isna()
    df.loc[missing_cond & (df["rainfall_mm"] >= 20) & (df["wind_speed_kmh"] >= 22), "weather_condition"] = "Stormy"
    df.loc[missing_cond & df["weather_condition"].isna() & (df["rainfall_mm"] > 0.5), "weather_condition"] = "Rainy"
    df.loc[missing_cond & df["weather_condition"].isna() & (df["cloud_cover_pct"] >= 70), "weather_condition"] = "Cloudy"
    df.loc[missing_cond & df["weather_condition"].isna(), "weather_condition"] = "Sunny"

    # 10. Clean binary target column (rain_tomorrow -> 1 / 0) and drop missing targets
    target_map = {
        "yes": 1, "y": 1, "1": 1, "true": 1,
        "no": 0, "n": 0, "0": 0, "false": 0,
    }
    df["rain_tomorrow"] = (
        df["rain_tomorrow"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(target_map)
    )
    df = df.dropna(subset=["rain_tomorrow"])
    df["rain_tomorrow"] = df["rain_tomorrow"].astype(int)

    # 11. Sort chronologically, format date cleanly, and remove post-cleaning duplicates
    df = df.sort_values(["date", "city"]).drop_duplicates().reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # 12. Save clean CSV
    df.to_csv(output_path, index=False)

    print(f"Cleaning complete: {initial_rows} raw rows -> {len(df)} clean rows")
    print(f"Remaining missing values: {df.isna().sum().sum()}")
    return df


if __name__ == "__main__":
    clean_df = clean_weather_dataset()
    print(clean_df.head(10).to_string())
