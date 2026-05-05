from pathlib import Path

import pandas as pd

STOCKS = ["NVDA", "AMD", "TSLA", "ASML", "ASM.AS"]

TRAIN_START_DATE = "2015-01-01"
TEST_START_DATE = "2023-01-01"
FORECAST_DATE = "2024-01-01"
FORECAST_HORIZON_MONTHS = 12

BASELINE_WINDOWS = [12, 24, 36, 60]

FEATURE_WINDOWS = [3, 6, 12]
FEATURE_AGGREGATIONS = ["mean", "min", "max", "std"]
FEATURE_LAGS = [1, 3]
HYPERPARAMETER_TUNING_RUNTIME = 10  # seconds

RESULTS_DIR = Path(__file__).resolve().parent / "output" / "results"
SAVE_TRAINING_DATA = True

# The earliest date needed to compute features for the training data, given the largest
# feature window and lag.
DATA_START_DATE = pd.to_datetime(TRAIN_START_DATE) - pd.DateOffset(months=max(FEATURE_WINDOWS))

# Train end determined by beginning of test period, minus a gap to ensure no overlap between
# training and test target windows.
TRAIN_END_DATE = pd.to_datetime(TEST_START_DATE) - pd.DateOffset(months=FORECAST_HORIZON_MONTHS)

# Gap needed between end of test and forecast date: 12m
TEST_END_DATE = pd.to_datetime(FORECAST_DATE) - pd.DateOffset(months=1)

# all start/end dates are inclusive
