import math
from tqdm import tqdm
import pandas as pd
import tempfile
import os


class DataLoader:
    def __init__(self, engine):
        self.engine = engine

    def load_dimension(self, df, table_name, chunk_size=50000):
        """
        Loads dimension data into the database using chunked inserts with a progress bar.
        Uses 'replace' on the first chunk to avoid duplicates.
        """
        self._load_in_chunks(df, table_name, chunk_size)
        print(f"[DataLoader] Loaded dimension '{table_name}' with {len(df)} rows.")

    def load_fact(self, df, table_name, chunk_size=50000):
        """
        Loads fact data into the database using chunked inserts with a progress bar.
        Uses 'replace' on the first chunk to avoid duplicates.
        """
        self._load_in_chunks(df, table_name, chunk_size)
        print(f"[DataLoader] Loaded fact '{table_name}' with {len(df)} rows.")

    def _load_in_chunks(self, df: pd.DataFrame, table_name: str, chunk_size: int):
        """
        Helper method to split the DataFrame into chunks and load each chunk into the database.
        If the entire DataFrame fits in one chunk, it uses 'replace' mode.
        For multi-chunk loads, the first chunk is loaded in 'replace' mode and the remaining are appended.
        """
        total_rows = len(df)
        total_chunks = math.ceil(total_rows / chunk_size)
        print(f"Loading {total_rows} rows into '{table_name}' in {total_chunks} chunk(s).")

        if total_chunks == 1:
            df.to_sql(table_name, self.engine, if_exists='replace', index=False)
            return

        start_index = 0
        with tqdm(total=total_rows, desc=f"Loading {table_name}", colour='green', leave=True) as pbar:
            while start_index < total_rows:
                end_index = start_index + chunk_size
                chunk_df = df.iloc[start_index:end_index]
                if start_index == 0:
                    chunk_df.to_sql(table_name, self.engine, if_exists='replace', index=False)
                else:
                    chunk_df.to_sql(table_name, self.engine, if_exists='append', index=False)
                start_index = end_index
                pbar.update(len(chunk_df))

    def load_fact_using_copy(self, df, table_name):
        """
        Alternative method for loading fact data using PostgreSQL's COPY command.
        This method writes the DataFrame to a temporary CSV file and executes a COPY command,
        which is much faster for very large tables.
        The table is created (or replaced) using df.head(0) to ensure a fresh start.
        """
        # Create or replace the table with the correct schema.
        df.head(0).to_sql(table_name, self.engine, if_exists='replace', index=False)

        # Write the DataFrame to a temporary CSV file (without header and index).
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv') as tmp:
            df.to_csv(tmp.name, index=False, header=False)
            tmp_file = tmp.name

        # Use a raw database connection to execute the COPY command.
        conn = self.engine.raw_connection()
        try:
            cur = conn.cursor()
            with open(tmp_file, 'r') as f:
                cur.copy_expert(f"COPY {table_name} FROM STDIN WITH CSV", f)
            conn.commit()
            print(f"[DataLoader] Loaded fact '{table_name}' with {len(df)} rows using COPY.")
        except Exception as e:
            print(f"Error during COPY: {e}")
            conn.rollback()
        finally:
            cur.close()
            conn.close()
            os.remove(tmp_file)
