"""
Sanity checks for the NYC Yellow Taxi ETL project.

This module defines a `TaxiDataSanityChecker` class that performs
basic validation on the raw and transformed taxi trip data. The
checks are intended to catch common issues such as mismatched row
counts, negative numeric values, and incomplete date dimensions.
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
    dim_date : pandas.DataFrame
        The date dimension table produced by the transformer.
    fact_df : pandas.DataFrame
        The fact table produced by the transformer.
    numeric_columns : Sequence[str], optional
        A list of columns expected to contain non-negative numeric
        values. If not provided, a default set of typical taxi
        metrics will be used.
    """

    raw_df: pd.DataFrame
    dim_date: pd.DataFrame
    fact_df: pd.DataFrame
    numeric_columns: Sequence[str] | None = None

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

    def check_date_dimension(self) -> None:
        """Assert that every pickup date in the raw data exists in the date dimension."""
        raw_dates = set(self.raw_df["tpep_pickup_datetime"].dt.normalize().dropna().unique())
        dim_dates = set(self.dim_date["full_date"].unique())
        missing = raw_dates - dim_dates
        assert not missing, f"Dates missing from date dimension: {sorted(missing)[:10]}"
        print("[Sanity] Date dimension coverage check passed.")

    def run_all_checks(self) -> None:
        """Run all defined sanity checks."""
        self.check_row_count()
        self.check_no_negative_values()
        self.check_date_dimension()
        print("All taxi data sanity checks passed successfully.")