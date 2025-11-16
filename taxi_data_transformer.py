"""
Data transformation module for the NYC Yellow Taxi ETL project.

This module defines a `TaxiDataTransformer` class responsible for
cleaning and enriching raw taxi trip data extracted from Parquet
files. The transformer converts string columns to proper dtypes,
derives useful metrics (trip duration, tip percentage), and builds
dimension tables such as a date dimension. It outputs both the
dimension table(s) and a fact table ready to be loaded into a data
warehouse.
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
    returns a date dimension table and a fact table. The transformer
    attempts to gracefully handle missing columns by only computing
    metrics when the required columns are present.
    """

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        tuple of (pandas.DataFrame, pandas.DataFrame)
            ``(dim_date, fact_trips)`` where ``dim_date`` is a date
            dimension table with surrogate keys and ``fact_trips`` is
            the fact table enriched with derived metrics and a date key.
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

        # Build a date dimension from the normalized pickup dates (midnight).
        df["pickup_date"] = df["tpep_pickup_datetime"].dt.normalize()
        unique_dates = df["pickup_date"].dropna().drop_duplicates().sort_values().reset_index(drop=True)
        dim_date = pd.DataFrame({"full_date": unique_dates})
        dim_date["day"] = dim_date["full_date"].dt.day
        dim_date["month"] = dim_date["full_date"].dt.month
        dim_date["year"] = dim_date["full_date"].dt.year
        dim_date["quarter"] = dim_date["full_date"].dt.quarter
        dim_date["week"] = dim_date["full_date"].dt.isocalendar().week

        def _get_season(month: int) -> str:
            if month in (12, 1, 2):
                return "Winter"
            elif month in (3, 4, 5):
                return "Spring"
            elif month in (6, 7, 8):
                return "Summer"
            else:
                return "Fall"

        dim_date["season"] = dim_date["month"].apply(_get_season)
        # Assign surrogate keys starting at 1.
        dim_date = dim_date.reset_index(drop=True)
        dim_date["date_id"] = dim_date.index + 1
        # Reorder columns for clarity.
        dim_date = dim_date[
            [
                "date_id",
                "full_date",
                "day",
                "month",
                "quarter",
                "year",
                "week",
                "season",
            ]
        ]

        # Merge date_id onto the fact table using the pickup_date.
        fact_trips = df.merge(
            dim_date[["date_id", "full_date"]],
            left_on="pickup_date",
            right_on="full_date",
            how="left",
        )
        # Drop helper columns used for merging.
        fact_trips = fact_trips.drop(columns=["pickup_date", "full_date"])

        return dim_date, fact_trips