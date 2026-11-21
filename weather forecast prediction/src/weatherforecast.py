"""Step 3: Weather forecasting project (NumPy + Pandas only).

Predicts NEXT DAY's temperature from the past 7 days of weather using
Linear Regression built from scratch with NumPy, compares it to a naive
baseline, and forecasts the next 7 days.
"""
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_weather.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "temperature"
FEATURES = ["temperature", "humidity", "pressure", "wind_speed", "rainfall"]
LAGS = 7

df = pd.read_csv(CLEANED_DATA_PATH, parse_dates=["date"]).set_index("date")

# ---------- Feature engineering ----------
def make_features(data):
    X = pd.DataFrame(index=data.index)
    for col in FEATURES:
        for lag in range(1, LAGS + 1):
            X[f"{col}_lag{lag}"] = data[col].shift(lag)
    X["temp_roll7_mean"] = data[TARGET].shift(1).rolling(7).mean()
    X["day_sin"] = np.sin(2 * np.pi * data.index.day_of_year / 365)
    X["day_cos"] = np.cos(2 * np.pi * data.index.day_of_year / 365)
    return X

X_all = make_features(df)
data = X_all.join(df[TARGET].rename("y")).dropna()

# ---------- Time-based train/test split (80/20, no shuffling!) ----------
split = int(len(data) * 0.8)
train, test = data.iloc[:split], data.iloc[split:]
X_train, y_train = train.drop(columns="y").values, train["y"].values
X_test, y_test = test.drop(columns="y").values, test["y"].values

# ---------- Standardise + Ridge linear regression (NumPy) ----------
mu, sigma = X_train.mean(0), X_train.std(0) + 1e-9
def add_bias(X): return np.hstack([np.ones((len(X), 1)), X])
Xtr, Xte = add_bias((X_train - mu) / sigma), add_bias((X_test - mu) / sigma)

lam = 1.0
I = np.eye(Xtr.shape[1]); I[0, 0] = 0
w = np.linalg.solve(Xtr.T @ Xtr + lam * I, Xtr.T @ y_train)

# ---------- Evaluation ----------
def metrics(y, p):
    err = y - p
    return (np.mean(np.abs(err)), np.sqrt(np.mean(err**2)),
            1 - np.sum(err**2) / np.sum((y - y.mean())**2))

pred = Xte @ w
baseline = test["temperature_lag1"].values        # "tomorrow = today"
print(f"Train days: {len(train)} | Test days: {len(test)}\n")
for name, p in [("Linear Regression", pred), ("Persistence baseline", baseline)]:
    mae, rmse, r2 = metrics(y_test, p)
    print(f"{name:22s} MAE={mae:.2f}°C  RMSE={rmse:.2f}°C  R²={r2:.3f}")

pd.DataFrame({"actual": y_test, "predicted": pred.round(1)},
             index=test.index).to_csv(OUTPUT_DIR / "test_predictions.csv")

# ---------- Rain probability model (ridge logistic regression) ----------
RAIN_FEATURES = ["temperature", "humidity", "pressure", "wind_speed"]
rain_x = df[RAIN_FEATURES].join(
    pd.Series(
        np.sin(2 * np.pi * df.index.day_of_year / 365),
        index=df.index,
        name="day_sin",
    )
).join(
    pd.Series(
        np.cos(2 * np.pi * df.index.day_of_year / 365),
        index=df.index,
        name="day_cos",
    )
).values
rain_y = (df["rainfall"].values > 0).astype(float)

rain_mu = rain_x.mean(0)
rain_sigma = rain_x.std(0) + 1e-9
rain_x_std = (rain_x - rain_mu) / rain_sigma
rain_x_std = add_bias(rain_x_std)

rain_w = np.zeros(rain_x_std.shape[1])
rain_I = np.eye(rain_x_std.shape[1])
rain_I[0, 0] = 0
learning_rate = 0.1
for _ in range(2000):
    logits = np.clip(rain_x_std @ rain_w, -30, 30)
    rain_prob = 1 / (1 + np.exp(-logits))
    gradient = rain_x_std.T @ (rain_prob - rain_y) / len(rain_y)
    gradient += 1.0 * rain_I @ rain_w
    rain_w -= learning_rate * gradient

def rain_probability(row, date):
    values = row[RAIN_FEATURES].values.astype(float).tolist()
    values.extend([
        np.sin(2 * np.pi * date.day_of_year / 365),
        np.cos(2 * np.pi * date.day_of_year / 365),
    ])
    x = add_bias(((np.asarray(values) - rain_mu) / rain_sigma).reshape(1, -1))
    probability = 1 / (1 + np.exp(-np.clip((x @ rain_w)[0], -30, 30)))
    return float(np.clip(probability * 100, 0, 100))

# ---------- Forecast the next 7 days (recursive) ----------
hist = df[FEATURES].copy()
future = []
for step in range(7):
    next_date = hist.index[-1] + pd.Timedelta(days=1)
    row = hist.copy()
    row.loc[next_date] = np.nan                    # placeholder for features
    feats = make_features(row).iloc[[-1]]
    x = add_bias((feats.values - mu) / sigma)
    t_pred = float((x @ w)[0])
    # other variables: assume they continue as the last 7-day average
    new_row = hist.iloc[-7:].mean()
    new_row["temperature"] = t_pred
    hist.loc[next_date] = new_row
    rain_pct = rain_probability(new_row, next_date)
    future.append((
        next_date.date(),
        round(t_pred, 1),
        round(float(new_row["wind_speed"]), 1),
        round(rain_pct, 1),
    ))

forecast = pd.DataFrame(
    future,
    columns=[
        "Date",
        "Temperature (C)",
        "Wind Speed (km/hr)",
        "Rain chance (%)",
    ],
)
forecast.to_csv(OUTPUT_DIR / "forecast_next_7_days.csv", index=False)
print("\n7-day temperature forecast:\n", forecast.to_string(index=False))
