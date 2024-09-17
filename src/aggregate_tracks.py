"""
    This stage aggregates the tracks over their duration, generating a
    dataset with the following columns:
    - track_id
    - maximum_altitude
    - average_altitude
    - average_ground_speed
"""

import os
import glob

from rich.console import Console
import polars as pl
from rich.progress import track

from utils.dvc.params import get_params


console = Console()
params = get_params("aggregate_tracks")

OUTPUT_PATH = params["output_path"]
INPUT_PATH = params["input_path"]
RESTART = params["restart"]

if not os.path.exists(params["output_path"]):
    os.makedirs(params["output_path"])

PARQUET_FILES = [
    file for file in os.listdir(INPUT_PATH) if file.endswith(".parquet")
]

PARQUET_FILES.sort()
n_files = len(PARQUET_FILES)

console.log(f"Found {len(PARQUET_FILES)} files in {INPUT_PATH}")
console.rule("Starting track aggreagation process...")

OUTPUT_FILES = [
    file for file in os.listdir(OUTPUT_PATH) if file.endswith(".parquet")
]

if RESTART and len(OUTPUT_FILES) > 0:
    console.log(
        "Restart flag is set to True. Cleaning all files in the output"
        + "folder."
    )
    files_to_rm = glob.glob(os.path.join(OUTPUT_PATH, "*.parquet"))
    for file in files_to_rm:
        os.remove(file)
else:
    # Only clean files that are not already present in the output folder.
    PARQUET_FILES = [
        file for file in PARQUET_FILES if file not in OUTPUT_FILES]
    previously_cleaned = n_files - len(PARQUET_FILES)
    console.log(
        f"Skipped {previously_cleaned} files that were already in the"
        + "output folder."
    )

for file in track(PARQUET_FILES, description="Aggregating tracks..."):
    console.log(f"Processing file {file}...")
    daily_data = pl.read_parquet(os.path.join(INPUT_PATH, file))

    # Group the data by track_id and calculate the maximum altitude,
    # average altitude, and average ground speed.
    grouped_data = (
        daily_data.groupby("track_id")
        .agg(
            pl.max("altitude").alias("maximum_altitude"),
            pl.avg("altitude").alias("average_altitude"),
            pl.avg("ground_speed").alias("average_ground_speed"),
        )
        .sort("track_id")
    )

    # Save the aggregated data to a new parquet file.
    output_file = os.path.join(OUTPUT_PATH, file)
    grouped_data = grouped_data.collect()

    grouped_data.write_parquet(output_file)
