"""
Data transformation module for the NYC Yellow Taxi ETL project.

This module defines a `TaxiDataTransformer` class responsible for
cleaning and enriching raw taxi trip data extracted from Parquet
files. The transformer converts string columns to proper dtypes,
derives useful metrics (trip duration, tip percentage), and builds
a date‑time dimension. Each unique pickup hour in the raw data is
captured as a row in the dimension table, allowing fact records to
link to a specific hour instead of just a date. The transformer
returns both the dimension table and a fact table ready to be loaded
into a data warehouse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class TaxiDataTransformer:
    """Transforms raw NYC yellow taxi trip data into a star schema.

    The primary method, :meth:`transform`, accepts a pandas DataFrame
    containing raw trip records. It cleans and enriches the data and
    returns a date‑time dimension table (with one row per hour) and a
    fact table. The transformer attempts to gracefully handle missing
    columns by only computing metrics when the required columns are
    present.
    """

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Cleans and enriches raw taxi trip data.

        Parameters
        ----------
        df : pandas.DataFrame
            Raw taxi trip data. Expected to contain at least the
            following columns: ``tpep_pickup_datetime`` and
            ``tpep_dropoff_datetime``. Additional columns such as
            ``tip_amount`` and ``fare_amount`` will be used if present.

        Returns
        -------
        tuple of (pandas.DataFrame, pandas.DataFrame, pandas.DataFrame)
            ``(dim_datetime, dim_fare, fact_trips)`` where ``dim_datetime`` is a
            date‑time dimension table with surrogate keys (one row per
            hour), ``dim_fare`` is a fare dimension capturing unique
            combinations of fare‑related monetary values, and
            ``fact_trips`` is the fact table enriched with derived
            metrics and foreign keys to the date‑time and fare
            dimensions.
        """
        # Make a copy to avoid mutating the caller's DataFrame.
        df = df.copy()

        # Ensure required datetime columns exist.
        required_cols = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns for transformation: {missing_cols}"
            )

        # Convert pickup and dropoff datetimes to pandas datetime dtype.
        df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
        df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce")

        # Compute trip duration in minutes. Negative durations (if any)
        # will be retained at this stage; sanity checks can flag them.
        duration = (
            df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
        ).dt.total_seconds() / 60.0
        df["trip_duration_min"] = duration.round(2)

        # Compute tip percentage if tip_amount and fare_amount columns exist.
        if {"tip_amount", "fare_amount"}.issubset(df.columns):
            with np.errstate(divide="ignore", invalid="ignore"):
                tip_pct = df["tip_amount"] / df["fare_amount"] * 100.0
                tip_pct = tip_pct.replace([np.inf, -np.inf], np.nan).fillna(0)
            df["tip_percentage"] = tip_pct.round(2)
        else:
            # If the required columns are absent, create a NaN column so
            # the schema remains consistent.
            df["tip_percentage"] = np.nan

        # Build a date‑time dimension from the pickup timestamps rounded down to the hour.
        # Each unique pickup hour becomes a row in the dimension table.  This
        # allows facts to be linked to a specific hour rather than just the
        # date.
        # Use lowercase 'h' for hour to avoid FutureWarning (uppercase 'H' is
        # deprecated in pandas 2.x)
        df["pickup_hour"] = df["tpep_pickup_datetime"].dt.floor("h")
        unique_hours = (
            df["pickup_hour"].dropna().drop_duplicates().sort_values().reset_index(drop=True)
        )
        dim_datetime = pd.DataFrame({"full_datetime": unique_hours})
        dim_datetime["date"] = dim_datetime["full_datetime"].dt.date
        dim_datetime["day"] = dim_datetime["full_datetime"].dt.day
        dim_datetime["month"] = dim_datetime["full_datetime"].dt.month
        dim_datetime["year"] = dim_datetime["full_datetime"].dt.year
        dim_datetime["quarter"] = dim_datetime["full_datetime"].dt.quarter
        # ``week`` uses ISO week number
        dim_datetime["week"] = dim_datetime["full_datetime"].dt.isocalendar().week
        dim_datetime["hour"] = dim_datetime["full_datetime"].dt.hour

        def _get_season(month: int) -> str:
            if month in (12, 1, 2):
                return "Winter"
            elif month in (3, 4, 5):
                return "Spring"
            elif month in (6, 7, 8):
                return "Summer"
            else:
                return "Fall"

        dim_datetime["season"] = dim_datetime["month"].apply(_get_season)
        # Assign surrogate keys starting at 1.
        dim_datetime = dim_datetime.reset_index(drop=True)
        dim_datetime["datetime_id"] = dim_datetime.index + 1
        # Reorder columns for clarity.
        dim_datetime = dim_datetime[
            [
                "datetime_id",
                "full_datetime",
                "date",
                "day",
                "month",
                "quarter",
                "year",
                "week",
                "hour",
                "season",
            ]
        ]

        # Merge datetime_id onto the fact table using the rounded pickup hour.
        fact_trips = df.merge(
            dim_datetime[["datetime_id", "full_datetime"]],
            left_on="pickup_hour",
            right_on="full_datetime",
            how="left",
        )
        # Drop helper columns used for merging.
        fact_trips = fact_trips.drop(columns=["pickup_hour", "full_datetime"])

        # ------------------------------------------------------------------
        # Build fare dimension.
        # Determine which monetary columns are present. These fields may
        # include the base fare, tip amount, total amount and various
        # surcharges. Only columns present in the input will be used.
        all_fare_cols = [
            "fare_amount",
            "tip_amount",
            "total_amount",
            "extra",
            "tolls_amount",
            "congestion_surcharge",
            "mta_tax",
        ]
        fare_cols = [col for col in all_fare_cols if col in df.columns]
        if fare_cols:
            # Create a dimension table with unique combinations of the
            # fare‑related columns. Each unique combination gets a
            # surrogate key ``fare_id``.
            dim_fare = df[fare_cols].drop_duplicates().reset_index(drop=True)
            dim_fare = dim_fare.copy()
            dim_fare["fare_id"] = dim_fare.index + 1
            # Merge the ``fare_id`` onto the fact table based on all
            # fare‑related columns. After merging, drop the original
            # fare fields from the fact table.
            fact_trips = fact_trips.merge(
                dim_fare[["fare_id"] + fare_cols],
                on=fare_cols,
                how="left",
            )
            fact_trips = fact_trips.drop(columns=fare_cols)
            # Reorder columns: place ``fare_id`` after datetime_id for
            # readability. If other columns exist, they remain in place.
            cols = list(fact_trips.columns)
            # Move fare_id immediately after datetime_id if both exist
            if "fare_id" in cols and "datetime_id" in cols:
                cols.remove("fare_id")
                idx = cols.index("datetime_id") + 1
                cols.insert(idx, "fare_id")
            fact_trips = fact_trips[cols]
        else:
            # If no fare columns are present, create an empty dimension
            # with a single row and assign the same fare_id to all
            # facts.
            dim_fare = pd.DataFrame({"fare_id": [1]})
            fact_trips["fare_id"] = 1

        return dim_datetime, dim_fare, fact_trips