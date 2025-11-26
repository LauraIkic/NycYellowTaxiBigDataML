"""
Entry point for the NYC Yellow Taxi ETL pipeline.

This module orchestrates a complete extract–transform–load workflow
for New York City yellow taxi trips.  It reads trip data from
Parquet files in the ``assets`` directory, cleans the data using a
dedicated cleaner class, transforms the cleaned records into a star
schema (consisting of a date–time dimension, a fare dimension and a
fact table), validates the results, and finally loads the
dimensions and fact into a PostgreSQL database.  Separate classes
handle extraction, cleaning, transformation, loading and
validation to adhere to the single‑responsibility principle.

Before running this script, verify that your database credentials are
correct and that the required Python dependencies listed in
``requirements.txt`` (such as ``pandas`` and ``pyarrow``) are
installed in your environment.  Use ``pip install -r requirements.txt``
to install them if necessary.
"""

from taxi_data_extractor import TaxiDataExtractor
from taxi_data_transformer import TaxiDataTransformer
from taxi_data_sanity_checker import TaxiDataSanityChecker
from data_loader import DataLoader
from db_connection import DBConnection
from weather_data_transformer import WeatherDataTransformer

import pandas as pd  # needed for datetime parsing and filtering
from taxi_data_cleaner import TaxiDataCleaner
from dashboard import create_app


def main() -> None:
    # Database connection settings. Update these to match your local
    # PostgreSQL configuration. The database will be created if it
    # doesn't already exist.
    user = "postgres"
    password = "password123"
    host = "localhost"
    port = "5433"
    dbname = "ny_taxi_dwh"

    # Initialize the database connection and create the database if needed.
    db = DBConnection(user, password, host, port, dbname)
    db.create_database_if_not_exists()
    engine = db.connect()

    # Directory containing the Parquet files. Adjust the path as necessary.
    data_dir = "assets"

    # Step 1: Extract data from Parquet files.
    extractor = TaxiDataExtractor(data_dir=data_dir, engine="pyarrow")
    raw_df = extractor.extract()
    print(f"[Extraction] Loaded {len(raw_df)} rows from Parquet files.")

    # Instantiate the data cleaner and apply all configured quality filters.
    # The cleaner handles timestamp conversion, removal of negative values,
    # checks passenger counts, filters distance/duration/speed outliers,
    # enforces fare/charge consistency, validates enumeration codes and
    # ensures location identifiers are within the known TLC zone range.
    # For a detailed breakdown of the rules, refer to ``taxi_data_cleaner.py``.
    cleaner = TaxiDataCleaner()
    raw_df = cleaner.clean(raw_df)

    # Step 2: Transform the cleaned data into a date‑time dimension, fare dimension and fact table.
    transformer = TaxiDataTransformer()
    # The transformer returns three DataFrames: the date‑time dimension,
    # the fare dimension (capturing unique combinations of fare‑related values),
    # and the fact table referencing both dimensions.
    dim_datetime, dim_fare, fact_trips = transformer.transform(raw_df)
    print(
        f"[Transformation] Produced dim_datetime with {len(dim_datetime)} rows, "
        f"dim_fare with {len(dim_fare)} rows and fact_trips with {len(fact_trips)} rows."
    )

    # Step 3: Validate the transformed data.  We validate using the raw
    # data and the date‑time dimension and fact table.  The fare
    # dimension is derived directly from the raw data and does not
    # affect row counts, so it is not passed into the sanity checker.
    checker = TaxiDataSanityChecker(
        raw_df=raw_df,
        dim_datetime=dim_datetime,
        fact_df=fact_trips,
        # optional custom date range can be provided here if needed
    )
    checker.run_all_checks()

    # Step 4: Load the dimension and fact data into the PostgreSQL database.
    loader = DataLoader(engine)
    loader.load_dimension(dim_datetime, "dim_datetime", chunk_size=50000)
    loader.load_dimension(dim_fare, "dim_fare", chunk_size=50000)
    loader.load_fact_using_copy(fact_trips, "fact_trips")

    # Step 5: Load hourly weather data and build the weather dimension.
    # Weather CSVs live in the ``assets`` directory alongside the Parquet files.
    # Construct full paths to each file.  Note that the February file name in the
    # provided dataset contained a trailing space before ``.csv``; we preserve
    # the filename exactly as given but prefix it with the assets directory.
    from pathlib import Path

    weather_dir = Path(data_dir)
    weather_files = [
        weather_dir / "New York City 2025-01-01 to 2025-01-31.csv",
        weather_dir / "New York City 2025-02-01 to 2025-02-28 .csv",
        weather_dir / "New York City 2025-03-01 to 2025-03-31.csv",
    ]
    try:
        weather_dfs = []
        for wf in weather_files:
            try:
                # ``wf`` is a Path object; convert to string for pandas
                df_weather = pd.read_csv(str(wf))
                weather_dfs.append(df_weather)
            except FileNotFoundError:
                print(f"[Weather] Weather file not found: {wf}. Skipping this file.")
        if weather_dfs:
            weather_df = pd.concat(weather_dfs, ignore_index=True)
            weather_transformer = WeatherDataTransformer()
            dim_weather = weather_transformer.transform(weather_df, dim_datetime)
            loader.load_dimension(dim_weather, "dim_weather", chunk_size=50000)
            print(f"[Weather] Loaded dim_weather with {len(dim_weather)} rows.")
        else:
            print("[Weather] No weather files found. Skipping weather dimension.")
    except Exception as e:
        print(f"[Weather] An error occurred while processing weather data: {e}")

    print("[ETL] NYC Yellow Taxi pipeline completed successfully!")

if __name__ == "__main__":
    main()