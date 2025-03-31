import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import lightgbm as lgb
import json
import os
from pathlib import Path
from math import radians, cos, sin, asin, sqrt
from collections import defaultdict
import sys
from loguru import logger

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{module}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)


# --- Configuration ---
DATA_DIR = Path("data/raw")
PREDICTIONS_DIR = Path("predictions")
INSTRUMENT_DATA_PATH = DATA_DIR / "instrument_data.csv"
POLLUTANT_DATA_PATH = DATA_DIR / "pollutant_data.csv"
MEASUREMENT_DATA_PATH = DATA_DIR / "measurement_data.csv"
OUTPUT_FILE = PREDICTIONS_DIR / "predictions_task_3.json"
N_NEIGHBORS = 3  # Number of closest neighbors to consider

# Define the target station, pollutant, and period combinations
TARGETS = [
    {
        "station": 205,
        "pollutant": "SO2",
        "period_start": "2023-11-01 00:00:00",
        "period_end": "2023-11-30 23:00:00",
    },
    {
        "station": 209,
        "pollutant": "NO2",
        "period_start": "2023-09-01 00:00:00",
        "period_end": "2023-09-30 23:00:00",
    },
    {
        "station": 223,
        "pollutant": "O3",
        "period_start": "2023-07-01 00:00:00",
        "period_end": "2023-07-31 23:00:00",
    },
    {
        "station": 224,
        "pollutant": "CO",
        "period_start": "2023-10-01 00:00:00",
        "period_end": "2023-10-31 23:00:00",
    },
    {
        "station": 226,
        "pollutant": "PM10",
        "period_start": "2023-08-01 00:00:00",
        "period_end": "2023-08-31 23:00:00",
    },
    {
        "station": 227,
        "pollutant": "PM2.5",
        "period_start": "2023-12-01 00:00:00",
        "period_end": "2023-12-31 23:00:00",
    },
]


def geo_dist(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Geographical distance between two geographic coordinates. Could use an euclidean distance too.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * np.arcsin(np.sqrt(a)) * 6371


def find_closest_neighbors(target_station_code, station_coords, n=N_NEIGHBORS):
    """Find the n closest neighbors to a target station."""
    target_coords = station_coords.get(target_station_code)
    if not target_coords:
        return []

    distances = []
    for station_code, coords in station_coords.items():
        if station_code != target_station_code:
            distance = geo_dist(
                target_coords[1], target_coords[0], coords[1], coords[0]
            )
            distances.append((station_code, distance))

    distances.sort(key=lambda x: x[1])
    return [int(station[0]) for station in distances[:n]]


def create_features(df, date_col="Measurement date"):
    """Add time-based features to the dataframe."""
    df_copy = df.copy()
    if date_col not in df_copy.columns:
        df_copy[date_col] = df_copy.index

    df_copy["hour"] = df_copy[date_col].dt.hour
    df_copy["dayofweek"] = df_copy[date_col].dt.dayofweek
    df_copy["month"] = df_copy[date_col].dt.month
    df_copy["day"] = df_copy[date_col].dt.day
    if "target_value" in df_copy.columns:
        df_copy["target_value_lag1"] = df_copy["target_value"].shift(1)
        df_copy["target_value_lag3"] = df_copy["target_value"].shift(3)

    return df_copy.drop(columns=[date_col], errors="ignore")


def load_data():
    """Load the dataset files."""
    try:
        instrument_df = pd.read_csv(
            INSTRUMENT_DATA_PATH, parse_dates=["Measurement date"]
        )
        pollutant_df = pd.read_csv(POLLUTANT_DATA_PATH)
        measurement_df = pd.read_csv(
            MEASUREMENT_DATA_PATH, parse_dates=["Measurement date"]
        )
        return instrument_df, pollutant_df, measurement_df
    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}")
        exit(1)


def get_station_coordinates(measurement_df):
    """Extract unique station coordinates from measurement data."""
    station_locations = measurement_df[
        ["Station code", "Latitude", "Longitude"]
    ].drop_duplicates("Station code")
    return {
        row["Station code"]: (row["Latitude"], row["Longitude"])
        for _, row in station_locations.iterrows()
    }


def prepare_training_data(
    instrument_df, measurement_df, station, pollutant, train_end_date, neighbors
):
    """Prepare training data for a specific target station and pollutant."""
    # Get instrument status data (target variable)
    instr_hist = instrument_df[
        (instrument_df["Station code"] == station)
        & (instrument_df["Item name"] == pollutant)
        & (instrument_df["Measurement date"] < train_end_date)
    ][["Measurement date", "Average value", "Instrument status"]].set_index(
        "Measurement date"
    )

    # Get measurement data for target and neighbor stations
    stations_to_get = [station] + neighbors
    meas_hist = measurement_df[
        (measurement_df["Station code"].isin(stations_to_get))
        & (measurement_df["Measurement date"] < train_end_date)
    ]

    meas_pivot = meas_hist.pivot_table(
        index="Measurement date", columns="Station code", values=pollutant
    )

    rename_dict = {station: "target_value"}
    for i, neighbor_code in enumerate(neighbors):
        if neighbor_code in meas_pivot.columns:
            rename_dict[neighbor_code] = f"neighbor_{i + 1}_value"

    meas_pivot_renamed = meas_pivot.rename(columns=rename_dict)
    cols_to_keep = ["target_value"] + [
        f"neighbor_{i + 1}_value" for i in range(N_NEIGHBORS)
    ]
    meas_pivot_final = meas_pivot_renamed[
        [col for col in cols_to_keep if col in meas_pivot_renamed.columns]
    ]
    combined_train = instr_hist.join(meas_pivot_final, how="left")
    combined_train.rename(
        columns={"Average value": "instrument_avg_value"}, inplace=True
    )
    combined_train = create_features(combined_train)
    potential_features = (
        [
            "target_value",
            "instrument_avg_value",
            "target_value_lag1",
            "target_value_lag3",
        ]
        + [f"neighbor_{i + 1}_value" for i in range(N_NEIGHBORS)]
        + ["hour", "dayofweek", "month", "day"]
    )

    current_features = [f for f in potential_features if f in combined_train.columns]

    X_train = combined_train[current_features].ffill().fillna(0)
    y_train = combined_train["Instrument status"]

    y_train_series = y_train.copy()
    y_train_series.index = pd.to_datetime(y_train_series.index)
    status_1_counts_weekly = y_train_series[y_train_series == 1].resample("W").count()
    if not status_1_counts_weekly.empty:
        historical_weekly_avg_status1 = status_1_counts_weekly.mean()
    else:
        historical_weekly_avg_status1 = 0.0

    return X_train, y_train, historical_weekly_avg_status1


def prepare_prediction_data(
    measurement_df, station, pollutant, start_date, end_date, neighbors, model_features
):
    """Prepare prediction data for a specific target station and pollutant."""
    # Get measurement data for target and neighbor stations
    stations_to_get = [station] + neighbors
    meas_pred = measurement_df[
        (measurement_df["Station code"].isin(stations_to_get))
        & (measurement_df["Measurement date"] >= start_date)
        & (measurement_df["Measurement date"] <= end_date)
    ]

    # Pivot and rename columns
    meas_pivot_pred = meas_pred.pivot_table(
        index="Measurement date", columns="Station code", values=pollutant
    )

    rename_dict_pred = {station: "target_value"}
    for i, neighbor_code in enumerate(neighbors):
        if neighbor_code in meas_pivot_pred.columns:
            rename_dict_pred[neighbor_code] = f"neighbor_{i + 1}_value"

    meas_pivot_pred_renamed = meas_pivot_pred.rename(columns=rename_dict_pred)

    # Add time features and select required features
    pred_data = create_features(meas_pivot_pred_renamed)

    # Handle potentially missing features
    for feature in model_features:
        if feature not in pred_data.columns:
            pred_data[feature] = 0

    # Select only the features needed by the model
    X_pred = pred_data[model_features].ffill().fillna(0)

    return X_pred


def adjust_preds(
    predictions,
    probabilities,
    timestamps,
    historical_weekly_avg_status1,
    classes,
    status_code_to_adjust=1,
):
    """Adjusts predictions to match the historical weekly average of a specific status code."""

    pred_series = pd.Series(predictions, index=pd.to_datetime(timestamps))
    proba_df = pd.DataFrame(
        probabilities, index=pd.to_datetime(timestamps), columns=classes
    )

    if not pred_series.empty:
        num_weeks = (pred_series.index.max() - pred_series.index.min()).days / 7.0
        if num_weeks < 1:
            num_weeks = 1
    else:
        num_weeks = 0

    target_status1_count = int(round(historical_weekly_avg_status1 * num_weeks))
    current_status1_count = (pred_series == status_code_to_adjust).sum()
    diff = target_status1_count - current_status1_count

    logger.info(
        f"  Adjusting Status {status_code_to_adjust}: Target Count={target_status1_count}, Current Count={current_status1_count}, Diff={diff}"
    )

    adjusted_predictions = pred_series.copy()
    if diff > 0:
        candidates_indices = pred_series[pred_series != status_code_to_adjust].index
        if not candidates_indices.empty:
            candidates_proba = proba_df.loc[candidates_indices, status_code_to_adjust]
            indices_to_change = candidates_proba.nlargest(diff).index
            adjusted_predictions.loc[indices_to_change] = status_code_to_adjust
    elif diff < 0:
        candidates_indices = pred_series[pred_series == status_code_to_adjust].index
        if not candidates_indices.empty:
            candidates_proba = proba_df.loc[candidates_indices, status_code_to_adjust]
            indices_to_change = candidates_proba.nsmallest(abs(diff)).index

            for idx in indices_to_change:
                row_probs = proba_df.loc[idx].drop(status_code_to_adjust)
                if not row_probs.empty:
                    next_best_class = row_probs.idxmax()
                    adjusted_predictions.loc[idx] = next_best_class
                else:
                    adjusted_predictions.loc[idx] = 0
    return adjusted_predictions.values


def main():
    logger.info("Starting instrument anomaly detection...")

    # Load data
    logger.info("Loading data...")
    instrument_df, pollutant_df, measurement_df = load_data()
    instrument_df = pd.merge(
        instrument_df,
        pollutant_df[["Item code", "Item name"]],
        on="Item code",
        how="left",
    )

    logger.info("Finding neighboring stations...")
    station_coords = get_station_coordinates(measurement_df)

    closest_neighbors_map = {}
    target_stations_list = list(set(t["station"] for t in TARGETS))

    for station in target_stations_list:
        closest_neighbors_map[station] = find_closest_neighbors(
            station, station_coords, n=N_NEIGHBORS
        )
        logger.info(
            f"  Closest neighbors for Station {station}: {closest_neighbors_map[station]}"
        )

    all_predictions = {}

    logger.info("\nProcessing targets (training and predicting)...")

    for target_info in TARGETS:
        station = target_info["station"]
        pollutant = target_info["pollutant"]
        period_start = pd.to_datetime(target_info["period_start"])
        period_end = pd.to_datetime(target_info["period_end"])
        target_key = f"{station}-{pollutant}"

        logger.info(f"\nProcessing {target_key}...")
        logger.info(f"  Training period: before {period_start}")
        logger.info(f"  Prediction period: {period_start} - {period_end}")

        neighbors = closest_neighbors_map.get(station, [])

        X_train, y_train, hist_avg_stat1 = prepare_training_data(
            instrument_df, measurement_df, station, pollutant, period_start, neighbors
        )

        if X_train is None or y_train is None or X_train.empty or y_train.empty:
            logger.warning(f"  No valid training data for {target_key}. Skipping.")
            continue

        logger.info(
            f"  Training with {len(X_train)} samples and {X_train.shape[1]} features"
        )
        logger.info(f"  Features: {X_train.columns.tolist()}")

        # Train model
        lgbm = lgb.LGBMClassifier(
            objective="multiclass",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            n_estimators=200,
            learning_rate=0.05,
            verbose=-1,
        )

        lgbm.fit(X_train, y_train)
        model_features = X_train.columns.tolist()

        logger.info(f"  Model training complete.")

        logger.info(f"  Generating predictions...")

        X_pred = prepare_prediction_data(
            measurement_df,
            station,
            pollutant,
            period_start,
            period_end,
            neighbors,
            model_features,
        )

        probabilities = lgbm.predict_proba(X_pred)
        initial_predictions = lgbm.classes_[np.argmax(probabilities, axis=1)]
        adjusted_predictions = adjust_preds(
            initial_predictions,
            probabilities,
            X_pred.index,
            hist_avg_stat1,
            lgbm.classes_,
            status_code_to_adjust=1,
        )

        station_preds = {}
        for timestamp, prediction in zip(X_pred.index, adjusted_predictions):
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            station_preds[timestamp_str] = int(prediction)

        if str(station) not in all_predictions:
            all_predictions[str(station)] = {}
        all_predictions[str(station)].update(station_preds)

        logger.info(f"  Completed predictions for {target_key}.")

    output_json = {"target": all_predictions}
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nSaving predictions to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output_json, f, indent=2)

    logger.info("Prediction complete!")


if __name__ == "__main__":
    main()
