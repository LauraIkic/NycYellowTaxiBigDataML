"""
Data extraction module for the NYC Yellow Taxi ETL project.

This module defines a `TaxiDataExtractor` class responsible for loading
and merging multiple Parquet files containing New York City yellow taxi
trip data. The extractor reads all Parquet files from a given
directory using the specified engine (defaulting to ``pyarrow``) and
concatenates them into a single DataFrame.

Example usage::

    from taxi_data_extractor import TaxiDataExtractor

    extractor = TaxiDataExtractor(data_dir="assets", engine="pyarrow")
    taxi_df = extractor.extract()

You can then pass ``taxi_df`` to the transformer for further
processing.
"""

from pathlib import Path
from typing import Optional, List

import pandas as pd


class TaxiDataExtractor:
    """Extracts and merges NYC yellow taxi Parquet files.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing one or more Parquet files to be merged.
    engine : str, optional
        Parquet engine to use when reading files. Must be supported by
        pandas (e.g. ``'pyarrow'`` or ``'fastparquet'``). Defaults to
        ``'pyarrow'``, which is generally the most performant and
        actively maintained.

    Notes
    -----
    If no Parquet files are found in ``data_dir``, a
    ``FileNotFoundError`` will be raised. The extractor returns a
    single pandas DataFrame containing all rows from all Parquet files,
    preserving the original column structure.
    """

    def __init__(self, data_dir: str | Path, engine: str = "pyarrow"):
        self.data_dir = Path(data_dir)
        self.engine = engine

    def _get_parquet_files(self) -> List[Path]:
        """Returns a sorted list of Parquet files in the data directory."""
        return sorted(self.data_dir.glob("*.parquet"))

    def extract(self) -> pd.DataFrame:
        """Reads and concatenates all Parquet files into one DataFrame.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the concatenated contents of all
            Parquet files found in ``data_dir``.

        Raises
        ------
        FileNotFoundError
            If no Parquet files are found in ``data_dir``.
        ImportError
            If the specified Parquet engine is unavailable.
        """
        parquet_files = self._get_parquet_files()
        if not parquet_files:
            raise FileNotFoundError(
                f"No .parquet files found in {self.data_dir.resolve()}"
            )

        # Read each Parquet file individually. If the requested engine
        # isn't available, pandas will raise an ImportError which
        # surfaces up to the caller.
        dataframes: List[pd.DataFrame] = []
        for pf in parquet_files:
            df = pd.read_parquet(pf, engine=self.engine)
            dataframes.append(df)

        # Concatenate all DataFrames. Using ignore_index ensures the
        # resulting DataFrame has a continuous integer index.
        combined_df = pd.concat(dataframes, ignore_index=True)
        return combined_df