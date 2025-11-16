"""
Entry point for the NYC Yellow Taxi ETL pipeline.

This script reads Parquet trip data from the ``assets`` directory,
transforms it into a star schema with a date dimension and a fact
table, performs basic sanity checks, and loads the results into a
PostgreSQL database. The ``DataLoader`` and ``DBConnection`` classes
from your previous project are reused here, while taxi-specific
extraction, transformation and validation are provided by dedicated
modules.

Before running this script, ensure your database credentials are
correct and the required Python dependencies listed in
``requirements.txt`` are installed (notably ``pandas`` and
``pyarrow``). You can install them via ``pip install -r
requirements.txt``.
"""

from taxi_data_extractor import TaxiDataExtractor
from taxi_data_transformer import TaxiDataTransformer
from taxi_data_sanity_checker import TaxiDataSanityChecker
from data_loader import DataLoader
from db_connection import DBConnection

import pandas as pd  # needed for datetime parsing and filtering


def main() -> None:
    # Database connection settings. Update these to match your local
    # PostgreSQL configuration. The database will be created if it
    # doesn't already exist.
    user = "postgres"
    password = "your_password_here"
    host = "localhost"
    port = "5432"
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

    # Filter out records with negative values in critical numeric fields. Some
    # taxi records may contain negative fare amounts or other metrics (e.g.,
    # cancelled or adjusted rides). To maintain data quality, drop these rows
    # before transformation. You can extend this list to include additional
    # numeric columns as needed.
    numeric_fields_to_filter = [
        "fare_amount",
        "tip_amount",
        "total_amount",
        "trip_distance",
        "extra",
        "tolls_amount",
        "congestion_surcharge",
        "mta_tax",
        "passenger_count",
    ]
    for col in numeric_fields_to_filter:
        if col in raw_df.columns:
            raw_df = raw_df[raw_df[col] >= 0]
    print(f"[Cleaning] Retained {len(raw_df)} rows after filtering negative values.")

    # Filter out records with negative trip durations (dropoff before pickup).
    # Compute trip duration in minutes and drop rows where duration is negative.
    # First convert pickup and dropoff columns to datetime to compute duration.
    if "tpep_pickup_datetime" in raw_df.columns and "tpep_dropoff_datetime" in raw_df.columns:
        raw_df["tpep_pickup_datetime"] = pd.to_datetime(raw_df["tpep_pickup_datetime"], errors="coerce")
        raw_df["tpep_dropoff_datetime"] = pd.to_datetime(raw_df["tpep_dropoff_datetime"], errors="coerce")
        duration = (raw_df["tpep_dropoff_datetime"] - raw_df["tpep_pickup_datetime"]).dt.total_seconds() / 60.0
        raw_df = raw_df[duration >= 0]
        print(f"[Cleaning] Retained {len(raw_df)} rows after filtering negative durations.")

    # Step 2: Transform the sanitized data into dimensions and fact tables.
    transformer = TaxiDataTransformer()
    dim_date, fact_trips = transformer.transform(raw_df)
    print(f"[Transformation] Produced dim_date with {len(dim_date)} rows and fact_trips with {len(fact_trips)} rows.")

    # Step 3: Validate the transformed data.
    checker = TaxiDataSanityChecker(raw_df=raw_df, dim_date=dim_date, fact_df=fact_trips)
    checker.run_all_checks()

    # Step 4: Load the data into the PostgreSQL database.
    loader = DataLoader(engine)
    loader.load_dimension(dim_date, "dim_date", chunk_size=50000)
    loader.load_fact_using_copy(fact_trips, "fact_trips")
    print("[ETL] NYC Yellow Taxi pipeline completed successfully!")


if __name__ == "__main__":
    main()