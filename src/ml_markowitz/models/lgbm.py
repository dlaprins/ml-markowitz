import pandas as pd
from flaml import AutoML
from lightgbm import LGBMRegressor
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit

from ml_markowitz.config import (
    FEATURE_AGGREGATIONS,
    FEATURE_LAGS,
    FEATURE_WINDOWS,
    FORECAST_HORIZON_MONTHS,
    HYPERPARAMETER_TUNING_RUNTIME,
    TEST_END_DATE,
    TEST_START_DATE,
    TRAIN_END_DATE,
    TRAIN_START_DATE,
)
from ml_markowitz.data import get_forward_compounded_returns


class FeatureEngineering:
    """Class for feature engineering."""

    def __init__(self, ts_raw):
        self.ts_raw = ts_raw

    def generate_features(self, windows, aggregations, lags=None):
        """Generates all features.

        Currently implemented:
        - aggregations over windows
        - autocorrelation w.r.t. lags over windows
        - delta per window as a trend proxy

        Args:
            windows (list of ints): windows over which to aggregate
            aggregations (list of strings or functions): aggregations to apply to the windowed ts
            lags (list of ints):  lags to use for autocorrelation
        Returns
            feats (pd.DataFrame): dataframe of features

        """

        if lags is None:
            lags = []

        feats = pd.DataFrame(index=self.ts_raw.index)

        feats["target_today"] = self.ts_raw
        # feats['target_last_step'] = self.ts_raw.shift(1)

        shifted_ts = self.ts_raw.shift(1)
        for window in windows:
            logger.info(f"Generating features for rolling window: {window}")
            rolled_ts = shifted_ts.rolling(window)

            feats[f"trend_{window}"] = feats["target_today"] - shifted_ts.shift(window)
            for agg in aggregations:
                feat_name = (
                    f"{agg}_{window}" if isinstance(agg, str) else f"{agg.__name__}_{window}"
                )
                feats[feat_name] = rolled_ts.agg(agg)

            for lag in lags:
                if lag <= window:
                    feat_name = f"autocorr_{window}_lag_{lag}"
                    feats[feat_name] = shifted_ts.rolling(window).corr(shifted_ts.shift(lag))

        return feats


class HyperparameterSelector:
    """Class for hyperparameter selection."""

    def __init__(self, runtime, random_seed=0):
        self.runtime = runtime
        self.random_seed = random_seed

    def generate_hyperparams(self, X_it, y_it):
        """Run hyperparameter tuning and saves the results.

        The algorithm used is the internal hyperparameter tuning algorithm used by FLAML,
        'FLOW^2', a so-called cost-frugal optimization. See
        https://arxiv.org/pdf/1911.04706
        https://arxiv.org/pdf/2005.01571
        for details.

        Currently, only LGBMRegressor() is implemented as a model.

        Args:
            X_it (pd.DataFrame): in-time feature set
            y_it (np.array): in-time target
        """
        automl = AutoML()
        logger.info(
            f"Starting LGBM hyperparameter tuning with FLAML. Runtime: {self.runtime} seconds."
        )
        # TODO: fix the CV leakage by using gapped TSCV
        automl.fit(
            X_it,
            y_it,
            task="regression",
            time_budget=self.runtime,
            estimator_list=["lgbm"],
            metric="mse",
            eval_method="cv",
            split_type=TimeSeriesSplit(n_splits=3),
            seed=self.random_seed,
            verbose=-1,
        )

        self.hyperparams = automl.best_config
        return self.hyperparams


def generate_train_test_forecast(
    monthly_returns: pd.DataFrame,
    window: int = FORECAST_HORIZON_MONTHS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Builds train, test, and production forecast feature datasets.

    For each stock, engineers lag and rolling-window features from its monthly
    return series. The target is the forward compounded return over `window`
    months. Rows are split into three sets using the config dates:

    - train: TRAIN_START_DATE..TRAIN_END_DATE — used for model fitting.
    - test:  TEST_START_DATE..TEST_END_DATE   — used for out-of-time evaluation.
    - forecast: the single most-recent row per stock where the forward target
      is not yet available (production rows for portfolio construction).

    The leakage gap (TRAIN_END_DATE..TEST_START_DATE) is excluded: its 12-month
    forward targets overlap with those in the test set.

    Args:
        monthly_returns: DataFrame of simple monthly returns with dates as the
            index and stock tickers as columns.
        window: Number of months to compound forward for the target.
            Defaults to FORECAST_HORIZON_MONTHS.

    Returns:
        A tuple of (train_data, test_data, forecast_data):
            - train_data: Features and target for the in-time training period.
            - test_data: Features and target for the out-of-time test period.
            - forecast_data: Features only for the most recent row per stock.
    """
    forward_compounded_returns = get_forward_compounded_returns(monthly_returns, window=window)
    all_features: list[pd.DataFrame] = []

    for stock in monthly_returns.columns:
        ts = monthly_returns[stock]
        target_ts = forward_compounded_returns[stock].reindex(ts.index)

        features = FeatureEngineering(ts).generate_features(
            FEATURE_WINDOWS, FEATURE_AGGREGATIONS, FEATURE_LAGS
        )
        features["target"] = target_ts
        features["stock"] = stock
        all_features.append(features)

    data = pd.concat(all_features, axis=0)
    data["stock"] = data["stock"].astype("category")

    train_data = data.loc[(data.index >= TRAIN_START_DATE) & (data.index <= TRAIN_END_DATE)]
    test_data = data.loc[(data.index >= TEST_START_DATE) & (data.index <= TEST_END_DATE)]
    # Production rows: target is NaN because the forward window is incomplete.
    # Take the single most-recent row per stock for portfolio construction.
    forecast_data = (
        data.loc[data["target"].isna()]
        .groupby("stock", observed=True)
        .tail(1)
        .drop(columns=["target"])
    )

    return train_data, test_data, forecast_data


def train_lgbm_model(
    train_data: pd.DataFrame,
    hyperparameter_tuning_runtime: int = HYPERPARAMETER_TUNING_RUNTIME,
    hyperparams: dict = None,
) -> tuple[LGBMRegressor, dict]:
    """Tunes and fits a LightGBM model on the training data.

    Args:
        train_data: DataFrame containing engineered features and a "target"
            column for the in-time training period.
        hyperparameter_tuning_runtime: Time budget in seconds for FLAML tuning.
        hyperparams: Dictionary of hyperparameters to use for training.
            If None, hyperparameter tuning is performed.
    Returns:
        A tuple of (model, hyperparams):
            - model: Fitted LGBMRegressor ready for inference.
            - hyperparams: Dictionary of hyperparameters used during training.
    """
    feature_cols = [col for col in train_data.columns if col != "target"]
    x_train = train_data[feature_cols]
    y_train = train_data["target"].to_numpy()

    if hyperparams is None:
        selector = HyperparameterSelector(runtime=hyperparameter_tuning_runtime, random_seed=6)
        hyperparams = selector.generate_hyperparams(x_train, y_train)

    model = LGBMRegressor(**hyperparams)
    model.fit(x_train, y_train)
    return model, hyperparams


def lgbm_pipeline(
    monthly_returns: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, dict]:
    """Runs the full LightGBM pipeline: feature engineering, training, and inference.

    Builds train/test/forecast datasets from monthly returns, tunes and fits a
    LGBMRegressor on the training data, then generates predictions for both
    the out-of-time test window and the production forecast row.

    Args:
        monthly_returns: DataFrame of simple monthly returns with dates as the
            index and stock tickers as columns.

    Returns:
        A tuple of (y_test, mu_forecast, hyperparams):
            - y_test: Series of predicted returns over the out-of-time test
              window, indexed by (stock, date).
            - mu_forecast: Series of predicted 12-month compounded returns per
              stock for portfolio construction.
            - hyperparams: Dictionary of best hyperparameters found during tuning.
    """
    train_data, test_data, forecast_data = generate_train_test_forecast(monthly_returns)
    feature_cols = [col for col in train_data.columns if col != "target"]

    # Train on the in-time window
    model, hyperparams = train_lgbm_model(train_data)

    # Predict on the out-of-time test set
    x_test = test_data[feature_cols]
    y_test = pd.Series(
        model.predict(x_test),
        index=pd.MultiIndex.from_arrays(
            [x_test["stock"], x_test.index],
            names=["stock", "date"],
        ),
        name="y_test",
    )

    # Predict the forward 12-month return for each stock (most recent feature row)
    # TODO: Retrain model on train+test data using same hyperparameters before final forecast
    mu_forecast = pd.Series(
        model.predict(forecast_data[feature_cols]),
        index=forecast_data["stock"],
        name="mu_lgbm_forecast",
    )

    return y_test, mu_forecast, hyperparams
