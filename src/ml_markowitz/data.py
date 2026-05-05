import numpy as np
import pandas as pd
import yfinance as yf

from ml_markowitz.config import (
    DATA_START_DATE,
    FORECAST_HORIZON_MONTHS,
    RESULTS_DIR,
    SAVE_TRAINING_DATA,
)


def get_monthly_returns_dividends(stocks: list[str], save_data=SAVE_TRAINING_DATA) -> pd.DataFrame:
    """Fetches monthly returns for a list of stocks, including dividends, and saves
    the data to a CSV file.
    Args:
        stocks: List of stock ticker symbols.
    Returns:
       returns_df: A DataFrame containing monthly returns for each stock, including dividends.
    """
    returns_data = {}

    for stock in stocks:
        ticker = yf.Ticker(stock)
        monthly_data = ticker.history(start=DATA_START_DATE, interval="1mo", auto_adjust=True)
        close_prices = monthly_data["Close"].ffill()
        monthly_returns = close_prices.pct_change()
        returns_data[stock] = monthly_returns

    # Convert to DataFrame with an outer join
    returns_df = pd.DataFrame(returns_data).sort_index()

    # Keep only the first available trading day of each month
    returns_df = returns_df.resample("MS").first()

    if save_data:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        returns_df.to_csv(RESULTS_DIR / "monthly_returns_div.csv")

    if returns_df.index.tz is not None:
        returns_df.index = returns_df.index.tz_localize(None)

    return returns_df


def get_forward_compounded_returns(
    returns_df: pd.DataFrame, window: int = FORECAST_HORIZON_MONTHS
) -> pd.DataFrame:
    """Compute forward rolling compounded returns for each date.

    For each date t, compounds the simple returns over the strictly future
    `window` months: (1 + r[t+1]) * (1 + r[t+2]) * ... * (1 + r[t+window]) - 1.
    The target at t therefore excludes r[t], so features computed from returns
    up to and including t (e.g. target_today = r[t]) are strictly past relative
    to the target. The final `window` dates are dropped because their forward
    window is incomplete.

    Args:
        returns_df: DataFrame of simple monthly returns, indexed by date,
            with one column per stock.
        window: Number of months to compound forward. Defaults to 12.

    Returns:
        DataFrame of forward compounded returns with the same column
        structure as returns_df, with the last `window` rows dropped.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    rolling_prod = (1 + returns_df).rolling(window=window).apply(np.prod, raw=True)
    forward_compounded = rolling_prod.shift(-window) - 1
    forward_compounded = forward_compounded.dropna()
    return forward_compounded
