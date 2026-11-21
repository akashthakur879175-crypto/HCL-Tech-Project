# Weather Forecasting Project (Python + NumPy + Pandas)

## Project structure

```text
src/                 Python scripts run in sequence
data/raw/            Generated messy dataset
data/processed/      Cleaned dataset
output/              Model evaluation and 7-day forecast
```

## Run

```bash
python src/messydata.py
python src/cleandata.py
python src/weatherforecast.py
```

The 7-day output includes `Temperature (C)`, `Wind Speed (km/hr)`, and
`Rain chance (%)`.

The cleaning step handles impossible values and replaces statistical outliers
using the 1.5 x IQR rule before filling missing values.

Requirements: `pip install numpy pandas`
