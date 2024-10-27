"""
Train a CatBoostRegressor
"""

import json

from rich.console import Console
import pandas as pd
from catboost import CatBoostRegressor, Pool

from utils.dvc.params import get_params


console = Console()
params = get_params("train_catboost")

OUTPUT_PATH = params["output_path"]
INPUT_PATH = params["input_path"]
CATEGORICAL_FEATURES = params["categorical_features"]
TARGET = params["target"]
RANDOM_STATE = params["random_state"]
DTYPES = params["dtypes"]
PARAMS_GRID = params["params_grid"]
RANDOM_SEARCH_ITERS = params["random_search_iters"]
CV_FOLDS = params["cv_folds"]
EARLY_STOPPING_ROUNDS = params["early_stopping_rounds"]


console.log('Loading datasets')
training_data = pd.read_csv(INPUT_PATH['train_data'], dtype=DTYPES)
validation_data = pd.read_csv(INPUT_PATH['validation_data'], dtype=DTYPES)

x_train = training_data.drop(TARGET, axis=1)
y_train = training_data[TARGET]

x_val = validation_data.drop(TARGET, axis=1)
y_val = validation_data[TARGET]

with open(INPUT_PATH['selected_features'], 'r') as f:
    selected_features = json.load(f)

x_train = x_train[selected_features['selected_columns_names']]
x_val = x_val[selected_features['selected_columns_names']]

cat_features = [
    x for x in selected_features[
        'selected_columns_names'] if x in CATEGORICAL_FEATURES]
cat_features_indices = [x_train.columns.get_loc(x) for x in cat_features]

train_pool = Pool(x_train, y_train, cat_features=cat_features_indices)
val_pool = Pool(x_val, y_val, cat_features=cat_features_indices)

# * Use Random Search to find the best hyperparameters for the CatBoost model
console.log('Running Random Search to find the best hyperparameters')

search = CatBoostRegressor(
    random_seed=RANDOM_STATE,
    loss_function='RMSE'
)

search = search.randomized_search(
    param_distributions=PARAMS_GRID,
    X=train_pool,
    search_by_train_test_split=True,
    n_iter=RANDOM_SEARCH_ITERS,
    cv=CV_FOLDS
)

# * Train a model with best hyperparameters
best_params = search['params']

console.log('Training the model with the best hyperparameters')
catboost = CatBoostRegressor(
    random_seed=RANDOM_STATE,
    loss_function='RMSE',
    **best_params
)

catboost = catboost.fit(
    train_pool,
    eval_set=val_pool,
    use_best_model=True,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS)

catboost.save_model(OUTPUT_PATH)

console.log(f'Model saved to {OUTPUT_PATH}')
