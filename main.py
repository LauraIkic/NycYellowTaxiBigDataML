import pandas as pd
from pathlib import Path

data_dir = Path("assets")
parquet_files = sorted(data_dir.glob("*.parquet"))

if not parquet_files:
    raise FileNotFoundError(f"No .parquet files found in {data_dir.resolve()}")

print(f"Found {len(parquet_files)} parquet files:")
for f in parquet_files:
    print(" -", f.name)

combined_df = pd.concat((pd.read_parquet(f) for f in parquet_files), ignore_index=True)

print("\nCombined DataFrame info:")
print(combined_df.info())
print("\nSample rows:")
print(combined_df.head())

# Optionally save to a new parquet file
#output_path = data_dir / "combined_yellow_tripdata.parquet"
#combined_df.to_parquet(output_path, index=False)