"""
Load models and calculate predictions on the final submission set.
"""

import os
import json

from rich.console import Console
import pandas as pd
from catboost import CatBoostRegressor, Pool

from utils.dvc.params import get_params


console = Console()
params = get_params("predict_submission")
OUTPUT_PATH = params["output_path"]
INPUT_PATH = params["input_path"]
CATEGORICAL_FEATURES = params["categorical_features"]


data = pd.read_csv(INPUT_PATH['data'])
flight_id = data['flight_id']

submission_data = pd.read_csv(INPUT_PATH['submission'])
submission_ids = submission_data['flight_id']

with open(INPUT_PATH['tow_mean'], 'r') as file:
    mean_tow = float(file.read())

with open(INPUT_PATH['selected_features'], 'r') as f:
    selected_features = json.load(f)

data = data[selected_features['selected_columns_names']]

model = CatBoostRegressor().load_model(INPUT_PATH['model'])

console.log(f"Loaded model from {INPUT_PATH['model']}")
console.log(f"Loaded data from {INPUT_PATH['data']}")

predictions = model.predict(data)

submission = pd.DataFrame(
    {"flight_id": flight_id, "tow": predictions})

submission_ids = submission_ids[~submission_ids.isin(submission['flight_id'])]
submission = pd.concat(
    [submission, pd.DataFrame({"flight_id": submission_ids, "tow": mean_tow})])

submission.to_csv(OUTPUT_PATH, index=False)

console.log(
    f"Saved submission to {OUTPUT_PATH}")