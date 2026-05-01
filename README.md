# ml-markowitz

Equity return forecasting and mean-variance portfolio optimisation.

## Overview

Given a universe of stocks, the pipeline:

1. Fetches monthly total-return data (price + dividends) via `yfinance`.
2. Engineers lag and rolling-window features from the monthly return series.
3. Trains forecasting models (LightGBM, ARIMA, historical benchmarks) to
   predict the forward 12-month compounded return per stock.
4. Evaluates models out-of-time and selects the best one.
5. Constructs a tangent portfolio using Markowitz mean-variance optimisation
   with the model's expected-return forecasts and a historical covariance
   estimate.

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
