import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
import numpy as np
from loguru import logger
import sys
from typing import Dict, List, Union, Tuple, Any, Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data' / 'raw'
PREDICTIONS_DIR = BASE_DIR / 'predictions'
PLOTS_DIR = BASE_DIR / 'plots' / 'task_2' # Folder to store plots

PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logger
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{module}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

def geo_dist(lat1 : float, lon1 : float, lat2 : float, lon2 : float) -> float:
    """
        Geographical distance between two geographic coordinates. Could use an euclidean distance too.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * np.arcsin(np.sqrt(a)) * 6371

def weighted_centered_ma(window : np.ndarray, weights : np.ndarray) -> float:
    """
        Weighted moving average function.
    """
    if len(window) < len(weights):
        if len(window) == 1: 
            w = np.array([weights[1]])
        elif len(window) == 2:
             return np.mean(window)
        else:
             return np.mean(window)
    return np.sum(window * weights) / np.sum(weights)

measurement_file = DATA_DIR / 'measurement_data.csv'
instrument_file = DATA_DIR / 'instrument_data.csv'
pollutant_file = DATA_DIR / 'pollutant_data.csv'

logger.info(f"Loading data from {measurement_file}...")
df_measure = pd.read_csv(measurement_file)
logger.info(f"Loading data from {instrument_file}...")
df_instrument = pd.read_csv(instrument_file)
logger.info(f"Loading data from {pollutant_file}...")
df_pollutant_map = pd.read_csv(pollutant_file)

pollutant_to_code = df_pollutant_map.set_index('Item name')['Item code'].to_dict()

logger.info("Preprocessing measurement data...")
df_measure['Measurement date'] = pd.to_datetime(df_measure['Measurement date'])

logger.info("Preprocessing instrument data...")
df_instrument['Measurement date'] = pd.to_datetime(df_instrument['Measurement date'])
df_instrument_status = df_instrument[['Measurement date', 'Station code', 'Item code', 'Instrument status']].copy()

# --- Merge Instrument Status with Measurements ---
logger.info("Merging instrument status with measurements...")
df_measure_long = df_measure.melt(
    id_vars=['Measurement date', 'Station code', 'Latitude', 'Longitude'],
    value_vars=['SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5'],
    var_name='Item name',
    value_name='Pollutant value'
)

# Map Item name to Item code
df_measure_long['Item code'] = df_measure_long['Item name'].map(pollutant_to_code)

# Merge instrument status
df_merged = pd.merge(
    df_measure_long,
    df_instrument_status,
    on=['Measurement date', 'Station code', 'Item code'],
    how='left'
)

# Fill missing values
df_merged['Instrument status'] = df_merged['Instrument status'].fillna(0).astype(int)

df_measure = df_merged.pivot_table(
    index=['Measurement date', 'Station code', 'Latitude', 'Longitude'],
    columns='Item name',
    values=['Pollutant value', 'Instrument status']
)

df_measure.columns = ['_'.join(col).strip() for col in df_measure.columns.values]
df_measure = df_measure.reset_index()

value_cols = {f'Pollutant value_{p}': p for p in pollutant_to_code.keys()}
status_cols = {f'Instrument status_{p}': f'{p}_status' for p in pollutant_to_code.keys()}
df_measure = df_measure.rename(columns={**value_cols, **status_cols})

for p in pollutant_to_code.keys():
    if f'{p}_status' not in df_measure.columns:
        df_measure[f'{p}_status'] = 0

logger.info("Extracting station locations...")
station_locations = df_measure[['Station code', 'Latitude', 'Longitude']].drop_duplicates('Station code').set_index('Station code')
station_coords = {station: (row['Latitude'], row['Longitude']) for station, row in station_locations.iterrows()}
logger.info(f"Found {len(station_coords)} unique stations.")

logger.info("Calculating distances between stations...")
all_stations = list(station_coords.keys())
distances = pd.DataFrame(index=all_stations, columns=all_stations, dtype=float)
for s1 in all_stations:
    for s2 in all_stations:
        if s1 == s2:
            distances.loc[s1, s2] = 0.0
        else:
            lat1, lon1 = station_coords[s1]
            lat2, lon2 = station_coords[s2]
            distances.loc[s1, s2] = geo_dist(lat1, lon1, lat2, lon2)
logger.info("Distances calculated.")

pollutant_neighbor_config = {
    'SO2': {'max_neighbors': 1},
    'NO2': {'max_neighbors': 1},
    'O3': {'max_neighbors': 8},
    'CO': {'max_neighbors': 8},
    'PM10': {'max_neighbors': 15},
    'PM2.5': {'max_neighbors': 15}
}

df_measure = df_measure.set_index('Measurement date')
df_measure['hour'] = df_measure.index.hour
df_measure['year'] = df_measure.index.year

targets = [
    {'station_code': 206, 'pollutant': 'SO2', 'start': '2023-07-01 00:00:00', 'end': '2023-07-31 23:00:00'},
    {'station_code': 211, 'pollutant': 'NO2', 'start': '2023-08-01 00:00:00', 'end': '2023-08-31 23:00:00'},
    {'station_code': 217, 'pollutant': 'O3', 'start': '2023-09-01 00:00:00', 'end': '2023-09-30 23:00:00'},
    {'station_code': 219, 'pollutant': 'CO', 'start': '2023-10-01 00:00:00', 'end': '2023-10-31 23:00:00'},
    {'station_code': 225, 'pollutant': 'PM10', 'start': '2023-11-01 00:00:00', 'end': '2023-11-30 23:00:00'},
    {'station_code': 228, 'pollutant': 'PM2.5', 'start': '2023-12-01 00:00:00', 'end': '2023-12-31 23:00:00'},
]

all_predictions = {"target": {}}
epsilon = 1e-6

def generate_plot(
    station: int, 
    pollutant: str, 
    plot_hist_data: pd.Series, 
    pred_series: pd.Series, 
    max_neighbors: int, 
    pred_start: pd.Timestamp, 
    pred_end: pd.Timestamp
) -> str:
    plt.figure(figsize=(15, 6))

    plt.plot(plot_hist_data.index, 
             plot_hist_data.values, 
             label=f'Historical {pollutant} (Station {station})', 
             alpha=0.7, linewidth=1)
    plt.plot(pred_series.index, 
             pred_series.values, 
             label=f'Predicted {pollutant}', 
             linestyle='-', color='red', linewidth=1.5)

    plot_start_date = plot_hist_data.index.min() if not plot_hist_data.empty else pred_start
    if pd.isna(plot_start_date): plot_start_date = pred_start

    plt.xlim(max(plot_start_date - pd.Timedelta(days=15), pred_start - pd.Timedelta(days=30)),
             pred_end + pd.Timedelta(days=1))
    plt.ylim(bottom=0)

    plt.title(f'Station {station} - {pollutant} Pred')
    plt.xlabel('Date')
    plt.ylabel(f'{pollutant} Concentration')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    plot_filename = PLOTS_DIR / f'station_{station}_{pollutant}.png'
    plt.savefig(plot_filename)
    plt.close()
    
    return str(plot_filename)

for target in targets:
    station = target['station_code']
    pollutant = target['pollutant']
    pred_start = pd.to_datetime(target['start'])
    pred_end = pd.to_datetime(target['end'])
    pred_year = pred_start.year
    pred_month = pred_start.month
    pollutant_status_col = f'{pollutant}_status'
    max_neighbors = pollutant_neighbor_config[pollutant]['max_neighbors']

    logger.info(f"  Processing Station {station}, Pollutant {pollutant} (Max Neighbors: {max_neighbors}, Weighted MA) for {pred_start.strftime('%Y-%m')}...")

    # --- Calculate Fallback: Historical Hourly Average for Target Station ---
    hist_data_target = df_measure[
        (df_measure['Station code'] == station) &
        (df_measure['year'] < pred_year) &
        (df_measure[pollutant] >= 0) &
        (df_measure[pollutant_status_col] == 0)
    ][[pollutant, 'hour']].copy().dropna()

    hourly_avg_fallback = hist_data_target.groupby('hour')[pollutant].mean()
    overall_mean_fallback = hist_data_target[pollutant].mean()
    if pd.isna(overall_mean_fallback): overall_mean_fallback = 0
    hourly_avg_fallback = hourly_avg_fallback.reindex(range(24), fill_value=overall_mean_fallback)
    hourly_avg_fallback = hourly_avg_fallback.fillna(0)

    # --- Calculate Historical Monthly Average for Scaling ---
    hist_data_target_monthly = df_measure[
        (df_measure['Station code'] == station) &
        (df_measure.index.year < pred_year) &
        (df_measure.index.month == pred_month) &
        (df_measure[pollutant] >= 0) &
        (df_measure[pollutant_status_col] == 0)
    ][pollutant].copy().dropna()

    historical_monthly_avg = hist_data_target_monthly.mean()
    if pd.isna(historical_monthly_avg) or historical_monthly_avg < epsilon:
        historical_monthly_avg = overall_mean_fallback
    if pd.isna(historical_monthly_avg): historical_monthly_avg = 0

    pred_dates = pd.date_range(start=pred_start, end=pred_end, freq='h')

    station_predictions = {}
    predicted_values = []
    neighbor_data_start = pred_start - pd.Timedelta(hours=1)
    neighbor_data_end = pred_end + pd.Timedelta(hours=1)

    neighbor_data_filtered = df_measure[
        (df_measure.index >= neighbor_data_start) &
        (df_measure.index <= neighbor_data_end) &
        (df_measure['Station code'] != station) &
        (df_measure[pollutant] >= 0) &
        (df_measure[pollutant_status_col] == 0)
    ][[pollutant, 'Station code']].copy()

    neighbor_weights = np.array([0.25, 0.5, 0.25])
    neighbor_data_filtered['pollutant_wma3h_centered'] = neighbor_data_filtered.groupby('Station code')[pollutant]\
        .transform(lambda x: x.rolling(window=3, min_periods=1, center=True).apply(weighted_centered_ma, raw=True, kwargs={'weights': neighbor_weights})) 

    neighbor_data_filtered = neighbor_data_filtered.dropna(subset=['pollutant_wma3h_centered'])

    for date in pred_dates:
        neighbors_now_with_ma = neighbor_data_filtered[neighbor_data_filtered.index == date].copy() 
        weighted_sum = 0.0
        total_weight = 0.0
        fallback_needed = True

        if not neighbors_now_with_ma.empty:
            neighbors_now_with_ma['distance'] = neighbors_now_with_ma['Station code'].apply(lambda s: distances.loc[station, s])

            closest_neighbors = neighbors_now_with_ma.sort_values('distance').head(max_neighbors)

            for _, neighbor_row in closest_neighbors.iterrows():

                neighbor_ma_value = neighbor_row['pollutant_wma3h_centered']
                dist = neighbor_row['distance']
                if dist < epsilon: continue

                weight = 1.0 / (dist**2 + epsilon)
                weighted_sum += neighbor_ma_value * weight
                total_weight += weight

            if total_weight > 0:
                prediction = weighted_sum / total_weight
                prediction = max(0, prediction)
                fallback_needed = False

        if fallback_needed:
            prediction = hourly_avg_fallback[date.hour]
            prediction = max(0, prediction)

        predicted_values.append(prediction)
        
    predicted_monthly_avg = np.mean(predicted_values) if predicted_values else 0

    scaling_factor = 1.0
    if predicted_monthly_avg > epsilon:
        scaling_factor = historical_monthly_avg / predicted_monthly_avg

    scaled_predicted_values = [max(0, p * scaling_factor) for p in predicted_values]

    for i, date in enumerate(pred_dates):
         station_predictions[date.strftime('%Y-%m-%d %H:%M:%S')] = scaled_predicted_values[i] ## Use scaled values

    all_predictions["target"][str(station)] = station_predictions

    # --- PLOTS ---
    plot_hist_data = df_measure[
        (df_measure['Station code'] == station) &
        (df_measure[pollutant] >= 0) &
        (df_measure[pollutant_status_col] == 0)
    ][pollutant].dropna()

    pred_series = pd.Series(scaled_predicted_values, index=pred_dates)
    
    plot_filename = generate_plot(
        station=station,
        pollutant=pollutant,
        plot_hist_data=plot_hist_data,
        pred_series=pred_series,
        max_neighbors=max_neighbors,
        pred_start=pred_start,
        pred_end=pred_end
    )
    
    logger.info(f"    Plot saved to {plot_filename}")

output_file = PREDICTIONS_DIR / 'predictions_task_2.json'
logger.info(f"\nSaving predictions to {output_file}...")
with open(output_file, 'w') as f:
    json.dump(all_predictions, f, indent=2)

logger.success("Processing complete.")
