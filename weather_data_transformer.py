"""
Weather data transformation module for the NYC Yellow Taxi ETL project.

This module defines a ``WeatherDataTransformer`` class responsible for
    preparing New York City hourly weather data for loading into a data
    warehouse. The transformer parses the timestamp information from the
    raw weather data, aligns it with a date‑time dimension table, and
    selects a subset of useful weather attributes for analysis.

Example usage::

    from weather_data_transformer import WeatherDataTransformer
    import pandas as pd

    # For hourly weather, you can combine multiple CSVs into a single
    # DataFrame before transforming:
    # weather_df = pd.concat([pd.read_csv(f) for f in weather_files])
    # dim_weather = WeatherDataTransformer().transform(weather_df, dim_datetime)

You can then load ``dim_weather`` into your data warehouse using the
``DataLoader`` class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class WeatherDataTransformer:
    """Transforms raw weather data into a dimension table.

    The :meth:`transform` method accepts a pandas DataFrame containing
    weather records and a date dimension table. It parses the weather
    observation timestamp, aligns the data with the date dimension, and
    selects a subset of weather attributes for downstream analysis.

    Note
    ----
    The weather CSVs are expected to contain a ``datetime`` column in
    ISO 8601 format (e.g. ``'2025-01-01T13:00:00'``) along with hourly
    weather statistics such as ``temp``, ``humidity``, ``precip``, etc.
    The transformer is resilient to missing columns and will include
    only the fields present in both the input and the predefined list of
    weather attributes.
    """

    def transform(self, weather_df: pd.DataFrame, dim_datetime: pd.DataFrame) -> pd.DataFrame:
        """Transform raw hourly weather data into a ``dim_weather`` DataFrame.

        Parameters
        ----------
        weather_df : pandas.DataFrame
            Raw weather data loaded from one or more CSVs. Must contain
            a ``datetime`` column representing the time of the weather
            observation.
        dim_datetime : pandas.DataFrame
            The date‑time dimension table with a ``datetime_id`` and
            ``full_datetime`` columns. The ``full_datetime`` column will
            be used to match weather observations to their
            corresponding hour keys.

        Returns
        -------
        pandas.DataFrame
            A dimension table containing a ``datetime_id`` and a subset
            of weather attributes. Duplicate hours are dropped so that
            each ``datetime_id`` appears at most once.
        """
        # Work on copies to avoid modifying caller's data.
        weather_df = weather_df.copy()
        dim_dt = dim_datetime.copy()

        # Parse the full datetime of the weather observation. The
        # weather dataset provides the exact timestamp at the start of
        # each hour.
        weather_df["full_datetime"] = pd.to_datetime(weather_df["datetime"], errors="coerce")

        # Join weather observations to the date‑time dimension on the
        # full datetime. Only observations matching the dimension
        # (i.e., within the expected date range) will be included.
        mapping = dim_dt[["datetime_id", "full_datetime"]]
        merged = pd.merge(mapping, weather_df, on="full_datetime", how="inner")

        # Define the list of weather columns we want to keep. If some
        # columns are missing from the CSV, they will simply be
        # skipped. Always include the ``datetime_id`` so the dimension
        # integrates with the star schema.
        target_cols: List[str] = [
            "datetime_id",
            # Temperature and humidity
            "temp",
            "feelslike",
            "humidity",
            # Precipitation
            "precip",
            "precipprob",
            "snow",
            # Wind conditions
            "windgust",
            "windspeed",
            # Other weather characteristics
            "cloudcover",
            "uvindex",
            "visibility",
            "conditions",
        ]
        existing_cols = [col for col in target_cols if col in merged.columns]
        dim_weather = (
            merged[existing_cols].drop_duplicates(subset=["datetime_id"]).reset_index(drop=True)
        )

        return dim_weather