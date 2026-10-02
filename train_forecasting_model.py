import pandas as pd
import numpy as np
import pickle
from statsmodels.tsa.arima.model import ARIMA
import warnings
import os

warnings.filterwarnings("ignore") # Ignore statistical warnings for cleaner terminal output

print("[Aegis] Training Macro-Forecasting Model (ARIMA)...")

file_path = 'dataset/hospital_time_series.csv'

if not os.path.exists(file_path):
    print(f"Error: Could not find {file_path}. Please ensure it is in the 'dataset' folder.")
    exit()

# 1. Load the dataset
df = pd.read_csv(file_path)
print(f"-> Successfully loaded {file_path} ({len(df)} records found).")

# 2. Preprocess Time-Series Data
# Auto-detect date column and value column
date_col = df.columns[0]
target_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

# Ensure date column is properly formatted
try:
    df[date_col] = pd.to_datetime(df[date_col])
except Exception:
    print(f"-> Warning: Could not parse '{date_col}' as datetime. Creating synthetic timeline...")
    df[date_col] = pd.date_range(end=pd.Timestamp.today(), periods=len(df), freq='D')

df = df.sort_values(date_col)

# Set index and aggregate by day (in case there are multiple entries per day)
df_daily = df.set_index(date_col).resample('D').mean()

# Fill missing days with the previous day's value
ts_data = df_daily[target_col].ffill().dropna()

print(f"-> Time series ready: {len(ts_data)} days of data.")

# 3. Train the ARIMA Model
# Using order (7,1,1) to capture weekly seasonality and standard trends
arima_model = ARIMA(ts_data, order=(7, 1, 1))
fitted_model = arima_model.fit()

print("\n=== ARIMA Model Summary ===")
print(f"AIC: {fitted_model.aic:.2f} | BIC: {fitted_model.bic:.2f}")

# 4. Forecast the Next 7 Days
forecast = fitted_model.forecast(steps=7)
print("\nNext 7-Day Occupancy Forecast:")
print(forecast.round().astype(int).to_string())

# 5. Export Model & History for Streamlit UI
with open('arima_bed_model.pkl', 'wb') as f:
    pickle.dump({'model': fitted_model, 'history': ts_data}, f)
print("\n-> Serialized: arima_bed_model.pkl\n")