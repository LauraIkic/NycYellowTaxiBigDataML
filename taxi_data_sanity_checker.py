"""
Sanity checks for the NYC Yellow Taxi ETL project.

This module defines :class:`TaxiDataSanityChecker`, a helper used
after the data has been cleaned and transformed.  It performs a
series of assertions to confirm that the fact table and its
dimensions satisfy basic integrity constraints: row counts match
between raw and fact data, numeric columns in the fact table are
non‑negative, every pickup hour present in the raw data exists in the
date–time dimension, and all timestamps fall within the expected
quarter‑year range.  These checks are separate from the cleaning
operations performed by :class:`TaxiDataCleaner`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass
class TaxiDataSanityChecker:
    """Runs a set of sanity checks on transformed taxi data.

    Parameters
    ----------
    raw_df : pandas.DataFrame
        The raw DataFrame produced by the extractor.
    dim_datetime : pandas.DataFrame
        The date‑time dimension table produced by the transformer.
    fact_df : pandas.DataFrame
        The fact table produced by the transformer.
    numeric_columns : Sequence[str], optional
        A list of columns expected to contain non‑negative numeric
        values. If not provided, a default set of typical taxi
        metrics will be used.
    expected_start_datetime : pandas.Timestamp or None, optional
        The minimum datetime (inclusive) that should appear in the
        dataset. Defaults to January 1 2025 at 00:00.
    expected_end_datetime : pandas.Timestamp or None, optional
        The maximum datetime (inclusive) that should appear in the
        dataset. Defaults to March 31 2025 at 23:59:59.
    """

    raw_df: pd.DataFrame
    dim_datetime: pd.DataFrame
    fact_df: pd.DataFrame
    numeric_columns: Sequence[str] | None = None
    expected_start_datetime: pd.Timestamp | None = None
    expected_end_datetime: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        # Define a default set of numeric columns if none are provided.
        if self.numeric_columns is None:
            self.numeric_columns = [
                "passenger_count",
                "trip_distance",
                "fare_amount",
                "tip_amount",
                "total_amount",
                "extra",
                "tolls_amount",
                "congestion_surcharge",
                "mta_tax",
                "trip_duration_min",
            ]

        # Set default expected datetime range if not provided. The default
        # covers the first quarter of 2025. Note that the end datetime
        # includes the entire last hour of March 31 2025.
        if self.expected_start_datetime is None:
            self.expected_start_datetime = pd.Timestamp("2025-01-01T00:00:00")
        if self.expected_end_datetime is None:
            # Use 23:59:59 on March 31 so hours up to 23:00 are included
            self.expected_end_datetime = pd.Timestamp("2025-03-31T23:59:59")

    def check_row_count(self) -> None:
        """Assert that the fact table has the same number of rows as the raw data."""
        expected = len(self.raw_df)
        actual = len(self.fact_df)
        assert expected == actual, (
            f"Row count mismatch: raw_df has {expected} rows but fact_df has {actual} rows"
        )
        print("[Sanity] Row count check passed.")

    def check_no_negative_values(self) -> None:
        """Assert that specified numeric columns contain no negative values."""
        for col in self.numeric_columns:
            if col not in self.fact_df.columns:
                # Column missing from fact table; skip.
                continue
            invalid = self.fact_df[self.fact_df[col] < 0]
            assert invalid.empty, f"Negative values found in column '{col}':\n{invalid[[col]].head()}"
        print("[Sanity] Non-negative numeric values check passed.")

    def check_datetime_dimension(self) -> None:
        """Assert that every pickup hour in the raw data exists in the date‑time dimension."""
        # Floor pickup datetimes to the hour for comparison
        raw_hours = (
            # Use lowercase 'h' for hour to avoid FutureWarning (uppercase 'H' is deprecated)
            self.raw_df["tpep_pickup_datetime"].dt.floor("h").dropna().unique()
        )
        raw_hours_set = set(raw_hours)
        dim_hours_set = set(self.dim_datetime["full_datetime"].unique())
        missing = raw_hours_set - dim_hours_set
        assert not missing, (
            f"Date‑time dimension is missing the following hours: {sorted(missing)[:10]}"
        )
        print("[Sanity] Date‑time dimension coverage check passed.")

    def check_date_range(self) -> None:
        """Assert that all datetime values fall within the expected range.

        Checks the raw pickup datetimes, the date‑time dimension, and
        the fact table (via the dimension) to ensure that no records
        fall outside the configured start and end datetimes. If any
        records are found outside this window, an assertion error is
        raised with a preview of the offending values.
        """
        start = self.expected_start_datetime
        end = self.expected_end_datetime
        # Check raw pickup datetimes
        invalid_raw = self.raw_df[
            (self.raw_df["tpep_pickup_datetime"] < start)
            | (self.raw_df["tpep_pickup_datetime"] > end)
        ]
        assert invalid_raw.empty, (
            f"Raw data contains pickup datetimes outside expected range "
            f"{start} to {end}:\n"
            f"{invalid_raw[['tpep_pickup_datetime']].head()}"
        )
        # Check date‑time dimension
        invalid_dim = self.dim_datetime[
            (self.dim_datetime["full_datetime"] < start)
            | (self.dim_datetime["full_datetime"] > end)
        ]
        assert invalid_dim.empty, (
            f"Date‑time dimension contains datetimes outside expected range "
            f"{start} to {end}:\n"
            f"{invalid_dim[['full_datetime']].head()}"
        )
        # Check fact table by joining to the dimension on datetime_id
        if "datetime_id" in self.fact_df.columns:
            merged = self.fact_df.merge(
                self.dim_datetime[["datetime_id", "full_datetime"]],
                on="datetime_id",
                how="left",
            )
            invalid_fact = merged[
                (merged["full_datetime"] < start)
                | (merged["full_datetime"] > end)
            ]
            assert invalid_fact.empty, (
                f"Fact table contains datetimes outside expected range "
                f"{start} to {end}:\n"
                f"{invalid_fact[['full_datetime']].head()}"
            )
        print(
            f"[Sanity] Date‑time range check passed (from {start} to {end})."
        )

    def run_all_checks(self) -> None:
        """Run all defined sanity checks."""
        self.check_row_count()
        self.check_no_negative_values()
        self.check_datetime_dimension()
        self.check_date_range()
        print("All taxi data sanity checks passed successfully.")