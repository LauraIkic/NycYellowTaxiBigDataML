"""
Data cleaning module for the NYC Yellow Taxi ETL project.

This module defines a :class:`TaxiDataCleaner` responsible for
applying a series of data quality filters to raw taxi trip records
before they are passed to the transformer.  Cleaning removes rows
that contain values inconsistent with typical yellow cab operations or
business rules.  Examples include negative numeric values (surcharges
are always additive【320682609278729†L151-L159】, so negative values indicate
errors), passenger counts outside a reasonable range, extreme
distances or durations, inconsistent fare totals beyond a configurable
tolerance, invalid enumeration codes, and pickup or
drop‑off zones outside the known TLC zone IDs.

The default thresholds and date range reflect the expected first
quarter of 2025.  These parameters can be overridden when
instantiating the cleaner to tailor it to different data sets or
quality requirements.  The cleaner operates on a copy of the input
DataFrame and emits log messages indicating how many records were
removed by each filter to aid debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

import numpy as np
import pandas as pd


@dataclass
class TaxiDataCleaner:
    """Cleans raw taxi data by applying a sequence of quality filters.

    Parameters
    ----------
    start_datetime : pandas.Timestamp, optional
        The earliest allowed pickup timestamp. Records with pickup times
        earlier than this value will be dropped. Default is January 1,
        2025 at midnight.
    end_datetime : pandas.Timestamp, optional
        The latest allowed pickup timestamp. Records with pickup times
        later than this value will be dropped. Default is March 31,
        2025 at 23:59:59.
    max_passenger_count : int, optional
        Maximum number of passengers considered valid for a single taxi
        ride. Records with passenger counts above this value or below
        one will be removed. Default is 6.
    min_trip_distance : float, optional
        Minimum plausible trip distance in miles. Trips shorter than
        this distance are dropped. Default is 0.01 miles.
    max_trip_distance : float, optional
        Maximum plausible trip distance in miles. Trips longer than
        this distance are dropped. Default is 50 miles.
    max_trip_duration_hours : float, optional
        Maximum plausible trip duration in hours. Trips with durations
        exceeding this limit are dropped. Default is 4 hours.
    max_speed_mph : float, optional
        Maximum plausible average speed in miles per hour. Trips with
        calculated speeds above this limit are dropped. Default is 100 mph.
    allowed_payment_types : Iterable[int], optional
        List of valid payment type codes. Records with other values
        will be removed. Default accepts codes 1–6.
    allowed_rate_codes : Iterable[int], optional
        List of valid rate code IDs. Records with other values will be
        removed. Default accepts codes 1–6.
    location_columns : Iterable[str], optional
        Column names that contain location identifiers. These will be
        checked to ensure their values fall between 1 and 265. Default
        includes both trip record naming conventions for pickup and
        dropoff zones.
    """

    start_datetime: pd.Timestamp = pd.Timestamp("2025-01-01T00:00:00")
    end_datetime: pd.Timestamp = pd.Timestamp("2025-03-31T23:59:59")
    max_passenger_count: int = 6
    min_trip_distance: float = 0.01
    max_trip_distance: float = 50.0
    max_trip_duration_hours: float = 4.0
    max_speed_mph: float = 100.0
    allowed_payment_types: Iterable[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    allowed_rate_codes: Iterable[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    location_columns: Iterable[str] = field(
        default_factory=lambda: [
            "PULocationID",
            "DOLocationID",
            "pickup_location_id",
            "dropoff_location_id",
        ]
    )

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all configured filters to the provided DataFrame.

        The cleaning process operates on a copy of the input to avoid
        modifying the caller's DataFrame.  It prints progress messages
        indicating how many rows were removed by each filter.

        Parameters
        ----------
        df : pandas.DataFrame
            Raw taxi trip data as extracted from source files.

        Returns
        -------
        pandas.DataFrame
            A cleaned DataFrame with invalid or inconsistent records
            removed.
        """
        cleaned = df.copy()

        # Convert pickup and dropoff timestamps to pandas datetime.  This
        # conversion is required for subsequent duration calculations.
        if "tpep_pickup_datetime" in cleaned.columns:
            cleaned["tpep_pickup_datetime"] = pd.to_datetime(
                cleaned["tpep_pickup_datetime"], errors="coerce"
            )
        if "tpep_dropoff_datetime" in cleaned.columns:
            cleaned["tpep_dropoff_datetime"] = pd.to_datetime(
                cleaned["tpep_dropoff_datetime"], errors="coerce"
            )

        # 0. Remove rows with negative values in standard numeric columns.
        #    Negative surcharges or amounts are generally invalid in TLC data
        #    (surcharges are always additive【320682609278729†L151-L159】).  Rows containing
        #    negative values in any of these fields are therefore dropped.  If
        #    additional numeric columns should be enforced, extend this list.
        numeric_fields = [
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
        for col in numeric_fields:
            if col in cleaned.columns:
                before = len(cleaned)
                cleaned = cleaned[cleaned[col] >= 0]
                removed = before - len(cleaned)
                if removed > 0:
                    print(f"[Cleaner] Removed {removed} rows with negative {col} values.")

        # 1. Filter by passenger count.  Accept counts between 1 and
        #    ``max_passenger_count`` inclusive.  The TLC collects driver‑reported
        #    passenger counts【320682609278729†L151-L159】; zero or negative counts are
        #    considered invalid and dropped.
        if "passenger_count" in cleaned.columns:
            before = len(cleaned)
            cleaned = cleaned[
                (cleaned["passenger_count"] > 0)
                & (cleaned["passenger_count"] <= self.max_passenger_count)
            ]
            removed = before - len(cleaned)
            if removed > 0:
                print(
                    f"[Cleaner] Removed {removed} rows with passenger counts ≤0 or >{self.max_passenger_count}."
                )

        # 2. Filter by pickup datetime range.
        if "tpep_pickup_datetime" in cleaned.columns:
            before = len(cleaned)
            cleaned = cleaned[
                (cleaned["tpep_pickup_datetime"] >= self.start_datetime)
                & (cleaned["tpep_pickup_datetime"] <= self.end_datetime)
            ]
            removed = before - len(cleaned)
            if removed > 0:
                print(
                    f"[Cleaner] Removed {removed} rows outside the pickup date range {self.start_datetime} to {self.end_datetime}."
                )

        # 3. Filter distance and duration outliers and unrealistic speeds.
        required_cols = {
            "trip_distance",
            "tpep_dropoff_datetime",
            "tpep_pickup_datetime",
        }
        if required_cols.issubset(cleaned.columns):
            # Compute trip duration in hours for each row.
            duration_hours = (
                cleaned["tpep_dropoff_datetime"] - cleaned["tpep_pickup_datetime"]
            ).dt.total_seconds() / 3600.0
            # Avoid division by zero; speeds with zero duration are set to NaN.
            duration_hours_nonzero = duration_hours.replace(0, np.nan)
            speeds = cleaned["trip_distance"] / duration_hours_nonzero
            # Build a boolean mask for valid trips.  Require non-negative
            # durations, distance within configured bounds and a reasonable
            # average speed.  Extremely short or long trips, or negative
            # durations, are removed.
            mask = (
                (cleaned["trip_distance"] >= self.min_trip_distance)
                & (cleaned["trip_distance"] <= self.max_trip_distance)
                & (duration_hours >= 0)
                & (duration_hours <= self.max_trip_duration_hours)
                & (speeds <= self.max_speed_mph)
            )
            before = len(cleaned)
            cleaned = cleaned[mask]
            removed = before - len(cleaned)
            if removed > 0:
                print(
                    f"[Cleaner] Removed {removed} rows with implausible distance, duration or speed values."
                )

        # 4. Check charge consistency.  Trips where the total_amount is
        #    significantly less than the sum of component charges or
        #    where the fare_amount is zero while surcharges/tips are
        #    positive are likely erroneous.  A tolerance is applied to
        #    allow for small rounding differences in the data.  Rows
        #    failing these conditions are dropped.
        if "total_amount" in cleaned.columns:
            charge_fields = [
                "fare_amount",
                "extra",
                "mta_tax",
                "tolls_amount",
                "congestion_surcharge",
                "tip_amount",
            ]
            present_fields = [c for c in charge_fields if c in cleaned.columns]
            if present_fields:
                charges_sum = cleaned[present_fields].fillna(0).sum(axis=1)
                tolerance = 1.0  # allow up to $1.00 discrepancy in totals
                # Build a mask of rows to keep: total_amount should be at least charges_sum - tolerance
                mask = (cleaned["total_amount"] + tolerance >= charges_sum)
                # Additionally, if fare_amount exists, require it to be positive when other charges are positive
                if "fare_amount" in cleaned.columns:
                    mask = mask & ~(
                        (cleaned["fare_amount"] == 0) & (charges_sum > 0)
                    )
                before = len(cleaned)
                cleaned = cleaned[mask]
                removed = before - len(cleaned)
                if removed > 0:
                    print(
                        f"[Cleaner] Removed {removed} rows with inconsistent fare or charge totals beyond ${tolerance:.2f} tolerance."
                    )

        # 5. Validate payment type codes.
        if "payment_type" in cleaned.columns:
            # Convert payment type to numeric; non-numeric values become NaN.
            cleaned["payment_type"] = pd.to_numeric(
                cleaned["payment_type"], errors="coerce"
            )
            before = len(cleaned)
            cleaned = cleaned[cleaned["payment_type"].isin(self.allowed_payment_types)]
            removed = before - len(cleaned)
            if removed > 0:
                print(
                    f"[Cleaner] Removed {removed} rows with invalid payment_type codes (allowed: {self.allowed_payment_types})."
                )

        # 6. Validate rate code IDs for columns named 'RatecodeID' or 'rate_code'.
        for rate_col in ["RatecodeID", "rate_code"]:
            if rate_col in cleaned.columns:
                cleaned[rate_col] = pd.to_numeric(
                    cleaned[rate_col], errors="coerce"
                )
                before = len(cleaned)
                cleaned = cleaned[cleaned[rate_col].isin(self.allowed_rate_codes)]
                removed = before - len(cleaned)
                if removed > 0:
                    print(
                        f"[Cleaner] Removed {removed} rows with invalid rate codes in column '{rate_col}'."
                    )

        # 7. Validate pickup and dropoff location identifiers.  Use a
        #    numeric range of 1–265 inclusive (TLC zone IDs).
        for loc_col in self.location_columns:
            if loc_col in cleaned.columns:
                cleaned[loc_col] = pd.to_numeric(
                    cleaned[loc_col], errors="coerce"
                )
                before = len(cleaned)
                cleaned = cleaned[
                    (cleaned[loc_col] >= 1) & (cleaned[loc_col] <= 265)
                ]
                removed = before - len(cleaned)
                if removed > 0:
                    print(
                        f"[Cleaner] Removed {removed} rows with out-of-range IDs in '{loc_col}'."
                    )

        return cleaned