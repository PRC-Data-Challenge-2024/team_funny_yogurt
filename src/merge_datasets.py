"""
Stage to merge aggregated tracks dataset with flight list information.
"""

import os

from rich.console import Console
import polars as pl
from rich.progress import track

from utils.dvc.params import get_params


console = Console()
params = get_params("merge_datasets")

OUTPUT_PATH = params["output_path"]
INPUT_PATH = params["input_path"]
COLS_TO_DROP = params["cols_to_drop"]
TRACKS_PATH = INPUT_PATH["tracks"]
TRAIN_FLIGHT_LIST_PATH = INPUT_PATH["train_flight_list"]
TEST_FLIGHT_LIST_PATH = INPUT_PATH["test_flight_list"]

if not os.path.exists(os.path.dirname(params["output_path"]["train"])):
    os.makedirs(os.path.dirname(params["output_path"]["train"]))
if not os.path.exists(os.path.dirname(params["output_path"]["test"])):
    os.makedirs(os.path.dirname(params["output_path"]["test"]))

PARQUET_FILES = [
    file for file in os.listdir(TRACKS_PATH) if file.endswith(".parquet")
]
PARQUET_FILES.sort()

n_files = len(PARQUET_FILES)

console.log(f"Found {len(PARQUET_FILES)} files in {TRACKS_PATH}")
console.rule("Starting dataset merging process...")

for split, path in zip(
    ["train", "test"], [TRAIN_FLIGHT_LIST_PATH, TEST_FLIGHT_LIST_PATH]
):
    flight_list_data = pl.read_csv(path)
    flight_list_data = flight_list_data.drop(COLS_TO_DROP)
    merged_data = pl.DataFrame()

    for file in track(
        PARQUET_FILES, description=f"Merging {split} datasets..."
    ):
        tracks = pl.read_parquet(os.path.join(TRACKS_PATH, file))
        tracks = tracks.with_columns(tracks["flight_id"].cast(pl.Int64))
        # Left join from tracks to flight list on flight_id
        merged = tracks.join(flight_list_data, on="flight_id", how="left")
        merged_data = pl.concat([merged_data, merged])
    if split == "train":
        merged_data = merged_data.drop_nulls(subset=["tow"])
    else:
        merged_data = merged_data.drop_nulls(subset=["flown_distance"])
    print(merged_data.dtypes)
    merged_data.write_csv(OUTPUT_PATH[split])
    console.log(f"Dataset saved to {OUTPUT_PATH[split]}")
