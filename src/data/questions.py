import polars as pl
import json
import sys
import os
from loguru import logger

MEASUREMENT_DATA_PATH = "data/raw/measurement_data.csv"
INSTRUMENT_DATA_PATH = "data/raw/instrument_data.csv"
POLLUTANT_DATA_PATH = "data/raw/pollutant_data.csv"
OUTPUT_PATH = "predictions/questions.json"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{module}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)

logger.info("Loading datasets...")
instrument_df = pl.scan_csv(INSTRUMENT_DATA_PATH)
pollutant_df = pl.scan_csv(POLLUTANT_DATA_PATH)

logger.info("Preprocessing data...")
instrument_df = instrument_df.with_columns(
    pl.col("Measurement date").str.to_datetime()
).with_columns(
    pl.col("Average value").cast(pl.Float64, strict=False)
)


def solve_q1(instrument_lf: pl.LazyFrame, pollutant_lf: pl.LazyFrame) -> float:
    """
    Calculates the average SO2 concentration across all stations,
    returning the average of station averages. Considers only 'Normal' measurements.

    Args:
        instrument_lf: LazyFrame with instrument measurement data including status.
        pollutant_lf: LazyFrame mapping pollutant names to item codes.
    """
    logger.info("Solving Q1...")

    so2_item_code = pollutant_lf.filter(pl.col("Item name") == "SO2").select("Item code").collect().item(0, 0)

    normal_so2_measurements = instrument_lf.filter(
        (pl.col("Instrument status") == 0) & (pl.col("Item code") == so2_item_code)
    ).filter( # Filter nulls here
        pl.col("Average value").is_not_null()
    )

    station_averages = normal_so2_measurements.group_by("Station code").agg(
        pl.col("Average value").mean()
    )
    
    overall_average = station_averages.select("Average value").mean().collect().item(0, 0)

    result_q1 = round(overall_average, 5)
    logger.success(f"Q1 Result: {result_q1}")
    return result_q1


def solve_q2(instrument_lf: pl.LazyFrame, pollutant_lf: pl.LazyFrame) -> dict | str:
    """
    Calculates the average CO levels per season for station 209.
    Considers only 'Normal' measurements.

    Args:
        instrument_lf: LazyFrame with instrument measurement data including status and datetime dates.
        pollutant_lf: LazyFrame mapping pollutant names to item codes.
    """
    logger.info("Solving Q2...")
    STATION_CODE = 209

    co_item_code = pollutant_lf.filter(pl.col("Item name") == "CO").select("Item code").collect().item(0, 0)

    filtered_lf = instrument_lf.filter(
        (pl.col("Instrument status") == 0)
        & (pl.col("Item code") == co_item_code)
        & (pl.col("Station code") == STATION_CODE)
    )
    
    filtered_df = filtered_lf.filter(pl.col("Average value").is_not_null()).collect()
    
    season_map = {
        12: 1, 1: 1, 2: 1,  # Winter
        3: 2, 4: 2, 5: 2,  # Spring
        6: 3, 7: 3, 8: 3,  # Summer
        9: 4, 10: 4, 11: 4, # Fall
    }

    filtered_df = filtered_df.with_columns(
        pl.col("Measurement date").dt.month().alias("month")
    )
    
    filtered_df = filtered_df.with_columns(
        pl.col("month").map_elements(lambda m: season_map.get(m), return_dtype=pl.Int64).alias("season")
    )

    seasonal_averages = filtered_df.group_by("season").agg(
        pl.col("Average value").mean()
    )

    result_q2 = {str(i): 0.0 for i in range(1, 5)}
    for row in seasonal_averages.iter_rows(): 
        season, avg = row
        if season is not None: 
            result_q2[str(int(season))] = round(avg, 5)

    logger.success(f"Q2 Result: {result_q2}")
    return result_q2


def solve_q3(instrument_lf: pl.LazyFrame, pollutant_lf: pl.LazyFrame) -> int | str:
    """
    Finds the hour with the highest variability (standard deviation) for O3.
    Considers only 'Normal' measurements across all stations.

    Args:
        instrument_lf: LazyFrame with instrument measurement data including status and datetime dates.
        pollutant_lf: LazyFrame mapping pollutant names to item codes.
    """
    logger.info("Solving Q3...")

    o3_item_code = pollutant_lf.filter(pl.col("Item name") == "O3").select("Item code").collect().item(0, 0)

    normal_o3_measurements_lf = instrument_lf.filter(
        (pl.col("Instrument status") == 0) & (pl.col("Item code") == o3_item_code)
    )

    normal_o3_measurements_lf = normal_o3_measurements_lf.with_columns(
        pl.col("Measurement date").dt.hour().alias("hour")
    )

    hourly_std_dev_df = normal_o3_measurements_lf.group_by("hour").agg(
        pl.col("Average value").std().alias("std_dev")
    ).collect()

    max_std_row_df = hourly_std_dev_df.filter(
        pl.col("std_dev") == pl.col("std_dev").max()
    )

    hour_max_std_dev = max_std_row_df.select("hour").item(0, 0)
    result_q3 = int(hour_max_std_dev)

    max_std_dev = max_std_row_df.select("std_dev").item(0, 0)
    logger.success(
        f"Q3 Result: Hour {result_q3} has the highest O3 variability (Std Dev: {max_std_dev:.5f})")
    return result_q3


def solve_q4(instrument_lf: pl.LazyFrame) -> int | str:
    """
    Finds the station code with the most measurements labeled as "Abnormal data".

    Args:
        instrument_lf: LazyFrame with instrument measurement data including status.

    """
    logger.info("Solving Q4...")
    ABNORMAL_DATA_STATUS_CODE = 9

    abnormal_data_measurements_lf = instrument_lf.filter(
        pl.col("Instrument status") == ABNORMAL_DATA_STATUS_CODE
    )

    station_counts_df = abnormal_data_measurements_lf.group_by("Station code").len().sort("len", descending=True).head(1).collect()

    station_max_abnormal = station_counts_df.select("Station code").item(0, 0)
    max_count = station_counts_df.select("len").item(0, 0)

    result_q4 = int(station_max_abnormal)

    logger.success(
        f"Q4 Result: Station {result_q4} has the most 'Abnormal data' measurements (Count: {max_count})")
    return result_q4


def solve_q5(instrument_lf: pl.LazyFrame) -> int | str:
    """
    Finds the station code with the most 'not normal' measurements.

    Args:
        instrument_lf: LazyFrame with instrument measurement data including status.
    """
    logger.info("Solving Q5...")
    NORMAL_STATUS_CODE = 0

    not_normal_measurements_lf = instrument_lf.filter(
        pl.col("Instrument status") != NORMAL_STATUS_CODE
    )

    station_counts_df = not_normal_measurements_lf.group_by("Station code").len().sort("len", descending=True).head(1).collect()

    station_max_not_normal = station_counts_df.select("Station code").item(0, 0)
    max_count = station_counts_df.select("len").item(0, 0)
    
    result_q5 = int(station_max_not_normal)

    logger.success(
        f"Q5 Result: Station {result_q5} has the most 'not normal' measurements (Count: {max_count})")
    return result_q5


def solve_q6(instrument_lf: pl.LazyFrame, pollutant_lf: pl.LazyFrame) -> dict | str:
    """
    Counts Good, Normal, Bad, and Very bad records for PM2.5 pollutant.
    Considers only 'Normal' measurements (status code 0).

    Args:
        instrument_lf: LazyFrame with instrument measurement data including status.
        pollutant_lf: LazyFrame with pollutant info including thresholds.
    """
    logger.info("Solving Q6...")
    POLLUTANT_NAME = "PM2.5"
    NORMAL_STATUS_CODE = 0

    pollutant_df_collected = pollutant_lf.collect()
    pm25_row_tuple = pollutant_df_collected.filter(pl.col("Item name") == POLLUTANT_NAME).row(0, named=True)

    pm25_item_code = pm25_row_tuple["Item code"]
    
    thresholds = {
        "Good": pm25_row_tuple["Good"],
        "Normal": pm25_row_tuple["Normal"],
        "Bad": pm25_row_tuple["Bad"],
    }
    
    good_threshold = float(thresholds["Good"])
    normal_threshold = float(thresholds["Normal"])
    bad_threshold = float(thresholds["Bad"])

    normal_pm25_measurements_lf = instrument_lf.filter(
        (pl.col("Instrument status") == NORMAL_STATUS_CODE)
        & (pl.col("Item code") == pm25_item_code)
        & pl.col("Average value").is_not_null()
    ).with_columns(
        pl.col("Average value").cast(pl.Float64, strict=False) 
    )

    classified_lf = normal_pm25_measurements_lf.with_columns(
        pl.when(pl.col("Average value") <= good_threshold).then(pl.lit("Good"))
          .when(pl.col("Average value") <= normal_threshold).then(pl.lit("Normal"))
          .when(pl.col("Average value") <= bad_threshold).then(pl.lit("Bad"))
          .otherwise(pl.lit("Very bad"))
          .alias("Category")
    )

    category_counts_df = classified_lf.group_by("Category").len().collect()
    
    result_q6 = {
        "Good": 0, "Normal": 0, "Bad": 0, "Very bad": 0,
    }
        
    # Iterate over collected results
    for row in category_counts_df.iter_rows(named=True):
        category = row['Category']
        count = row['len']
        if category in result_q6: 
            result_q6[category] = int(count)

    logger.success(f"Q6 Result: {result_q6}")
    return result_q6


def main():
    results = {}
    
    results["Q1"] = solve_q1(instrument_df, pollutant_df) 
    results["Q2"] = solve_q2(instrument_df, pollutant_df)
    results["Q3"] = solve_q3(instrument_df, pollutant_df)
    results["Q4"] = solve_q4(instrument_df)
    results["Q5"] = solve_q5(instrument_df)
    results["Q6"] = solve_q6(instrument_df, pollutant_df)

    output_json = {"target": results}

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output_json, f, indent=4)

    logger.info(f"Results written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
