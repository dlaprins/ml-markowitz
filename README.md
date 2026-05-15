# ml-markowitz

Equity return forecasting and mean-variance portfolio optimisation.

Work in progress.


## Overview

Given a universe of stocks, the pipeline:

1. Fetches monthly total-return data (price + dividends) via `yfinance`.
2. Engineers features from the monthly return series.
3. Trains forecasting models (LightGBM, ARIMA, historical benchmarks) to
   predict the forward 12-month compounded return per stock.
4. Backtest models out-of-time and selects the best one.
5. Constructs a tangent portfolio using Markowitz mean-variance optimisation
   with the model's expected-return forecasts and a historical covariance
   estimate.

Temporal setup is slightly convoluted:
- Gap of length observation window; used for feature generation
- Train; time period with both features and target, used for model construction
- Gap of length forecast horizon; since the target is 12m forward compounded, there needs to be a gap between train and test to avoid data leakage
- Test; time period with both features and target, used for model selection
- Gap of length forecast horizon; since the target is 12m forward compounded, there are 12 months with feature data but no target data
- Forecast date; most recent data, used for porfolio selection

## Installation

```bash
pip install -e ".[dev]"
```

Install with notebook support:

```bash
pip install -e ".[dev,notebooks]"
```

## Usage

```bash
python main.py
```

## Development

Install pre-commit hooks after cloning:

```bash
pre-commit install
```

Run linter:

```bash
ruff check .
ruff format .
```

Run tests:

```bash
pytest
```

## Repository layout

```
src/
  ml_markowitz/     # package source
tests/              # pytest test suite
notebooks/          # exploratory notebooks
output/             # generated artefacts — not committed
pyproject.toml      # build config, ruff config, pytest config
```


## To-do:
- add historical benchmark computation
- add arima pipeline
- add superior feature generation
- add feature selection to lgbm pipeline
- add backtesting pipeline for model selection
- add efficient frontier computation
