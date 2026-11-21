import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

df = pd.read_csv("weather_forecast_clean.csv")

print("----- DATA OVERVIEW -----")
print(df.head())
print("\nRows and columns:", df.shape)
print("\nColumn names:", list(df.columns))
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year

print("\nData is from", df["date"].min().date(), "to", df["date"].max().date())

print("\n----- BASIC STATISTICS -----")
print(df[["temperature_c", "humidity_pct", "wind_speed_kmh", "pressure_hpa", "rainfall_mm", "cloud_cover_pct"]].describe().round(2))

temp = df["temperature_c"].to_numpy()
print("\nTemperature using numpy")
print("Average :", np.mean(temp).round(2))
print("Median  :", np.median(temp))
print("Lowest  :", np.min(temp))
print("Highest :", np.max(temp))
print("Std dev :", np.std(temp).round(2))

print("\n----- CITY WISE ANALYSIS -----")
print(df["city"].value_counts())

city_summary = df.groupby("city").agg(
    days=("date", "count"),
    avg_temp=("temperature_c", "mean"),
    avg_humidity=("humidity_pct", "mean"),
    avg_rainfall=("rainfall_mm", "mean"),
    rain_tomorrow_chance=("rain_tomorrow", "mean"),
)
city_summary["rain_tomorrow_chance"] = city_summary["rain_tomorrow_chance"] * 100
print(city_summary.round(2).sort_values("rain_tomorrow_chance", ascending=False))

hottest = city_summary["avg_temp"].idxmax()
coolest = city_summary["avg_temp"].idxmin()
rainiest = city_summary["avg_rainfall"].idxmax()
print("\nHottest city on average:", hottest)
print("Coolest city on average:", coolest)
print("Highest average rainfall:", rainiest)

print("\n----- MONTH WISE ANALYSIS -----")
month_summary = df.groupby("month").agg(
    days=("date", "count"),
    avg_temp=("temperature_c", "mean"),
    avg_rainfall=("rainfall_mm", "mean"),
    rain_tomorrow_chance=("rain_tomorrow", "mean"),
)
month_summary["rain_tomorrow_chance"] = month_summary["rain_tomorrow_chance"] * 100
print(month_summary.round(2))

print("\nRainiest month number:", month_summary["avg_rainfall"].idxmax())
print("Hottest month number:", month_summary["avg_temp"].idxmax())

print("\n----- WEATHER CONDITION -----")
print(df["weather_condition"].value_counts())

condition_summary = df.groupby("weather_condition").agg(
    days=("date", "count"),
    rain_tomorrow_chance=("rain_tomorrow", "mean"),
)
condition_summary["rain_tomorrow_chance"] = condition_summary["rain_tomorrow_chance"] * 100
print("\nChance of rain the next day after each type of weather")
print(condition_summary.round(2))

print("\n----- WHAT AFFECTS RAIN TOMORROW? -----")
number_cols = ["temperature_c", "humidity_pct", "wind_speed_kmh",
               "pressure_hpa", "rainfall_mm", "cloud_cover_pct", "rain_tomorrow"]
print("Correlation with rain_tomorrow")
print(df[number_cols].corr()["rain_tomorrow"].drop("rain_tomorrow").round(3))

df["humidity_level"] = pd.cut(df["humidity_pct"],
                              bins=[0, 50, 65, 80, 100],
                              labels=["0-50", "50-65", "65-80", "80-100"])
print("\nRain tomorrow chance (%) by humidity level")
print((df.groupby("humidity_level", observed=True)["rain_tomorrow"].mean() * 100).round(2))

df["cloud_level"] = pd.cut(df["cloud_cover_pct"],
                           bins=[-1, 40, 60, 80, 100],
                           labels=["0-40", "40-60", "60-80", "80-100"])
print("\nRain tomorrow chance (%) by cloud cover")
print((df.groupby("cloud_level", observed=True)["rain_tomorrow"].mean() * 100).round(2))

df["pressure_level"] = pd.cut(df["pressure_hpa"],
                              bins=[990, 1005, 1010, 1015, 1030],
                              labels=["990-1005", "1005-1010", "1010-1015", "1015+"])
print("\nRain tomorrow chance (%) by pressure")
print((df.groupby("pressure_level", observed=True)["rain_tomorrow"].mean() * 100).round(2))

print("\n----- RAIN PREDICTION USING A POINT SYSTEM -----")
print("A day gets 1 point for each of these:")
print("1. Humidity above 70")
print("2. Cloud cover above 55")
print("3. Pressure below 1010")
print("4. Rainfall today more than 0")

df["point_humidity"] = np.where(df["humidity_pct"] > 70, 1, 0)
df["point_cloud"] = np.where(df["cloud_cover_pct"] > 55, 1, 0)
df["point_pressure"] = np.where(df["pressure_hpa"] < 1010, 1, 0)
df["point_rain"] = np.where(df["rainfall_mm"] > 0, 1, 0)

df["rain_points"] = (df["point_humidity"] + df["point_cloud"]
                     + df["point_pressure"] + df["point_rain"])

points_table = df.groupby("rain_points").agg(
    days=("date", "count"),
    rain_tomorrow_chance=("rain_tomorrow", "mean"),
)
points_table["rain_tomorrow_chance"] = points_table["rain_tomorrow_chance"] * 100
print("\nPoints and how often it rained the next day")
print(points_table.round(2))

df["predicted_rain"] = np.where(df["rain_points"] >= 3, 1, 0)

correct = np.sum(df["predicted_rain"] == df["rain_tomorrow"])
total = len(df)
accuracy = correct / total * 100
print("\nCorrect predictions:", correct, "out of", total)
print("Accuracy:", round(accuracy, 2), "%")

always_no_rain = np.sum(df["rain_tomorrow"] == 0) / total * 100
print("Accuracy if we always said 'no rain':", round(always_no_rain, 2), "%")

print("\nPrediction result table (rows = actual, columns = predicted)")
print(pd.crosstab(df["rain_tomorrow"], df["predicted_rain"]))

print("\nAccuracy for each city")
df["is_correct"] = np.where(df["predicted_rain"] == df["rain_tomorrow"], 1, 0)
print((df.groupby("city")["is_correct"].mean() * 100).round(2))


def predict_rain(humidity, cloud_cover, pressure, rainfall_today):
    points = 0
    if humidity > 70:
        points = points + 1
    if cloud_cover > 55:
        points = points + 1
    if pressure < 1010:
        points = points + 1
    if rainfall_today > 0:
        points = points + 1

    chance = float(points_table.loc[points, "rain_tomorrow_chance"])
    if points >= 3:
        answer = "Rain is likely tomorrow"
    else:
        answer = "Rain is not likely tomorrow"
    return answer, points, round(chance, 1)


print("\n----- TRY IT FOR A NEW DAY -----")
print("Humid and cloudy day :", predict_rain(85, 75, 1003, 4.5))
print("Dry and clear day    :", predict_rain(45, 20, 1018, 0))
print("Medium day           :", predict_rain(75, 50, 1008, 0))

print("\n----- FINAL CONCLUSION -----")
print("1. Humidity has the strongest link with rain tomorrow (correlation",
      round(df["humidity_pct"].corr(df["rain_tomorrow"]), 2), ")")
print("2. Low pressure and high cloud cover also mean more chance of rain.")
print("3. Chennai and Mumbai have more rain chances than Delhi.")
print("4. Rain chances are highest in July and August.")
print("5. The point system predicts correctly about", round(accuracy), "% of the time.")

df.to_csv("weather_forecast_results.csv", index=False)
city_summary.round(2).to_csv("city_summary.csv")
month_summary.round(2).to_csv("month_summary.csv")
print("\nSaved weather_forecast_results.csv, city_summary.csv and month_summary.csv")
