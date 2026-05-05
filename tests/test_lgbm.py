"""Unit tests for generate_train_test_forecast.

Timeline (120 months, 2010-01 to 2019-12):

  Zone            Date range              Months  Label
  --------------  ----------------------  ------  --------------
  Redundant pre   2010-01 to 2010-12        12   REDUNDANT_PRE
  Obs warmup      2011-01 to 2011-12        12   GAP
  Train           2012-01 to 2015-12        48   TRAIN           (4 years)
  Leakage gap     2016-01 to 2016-12        12   GAP
  Test            2017-01 to 2017-12        12   TEST            (1 year)
  Forecast        2018-01 to 2018-12        12   FORECAST        (valid targets)
  Redundant post  2019-01 to 2019-12        12   REDUNDANT_POST  (NaN targets)
                                           ---
                                           120

The obs warmup is 12 months (the minimum required for the 12m feature window).

Because REDUNDANT_POST follows FORECAST, the FORECAST rows have computable
forward targets (the trailing data covers the full forward window). The NaN
target rows are therefore in REDUNDANT_POST, not FORECAST.

Key behaviours verified:
  - REDUNDANT_PRE rows (before obs warmup) are excluded from all outputs.
  - FORECAST zone rows (valid targets, outside all date masks) are excluded
    from all outputs — they are neither train, test, nor the most-recent
    NaN row per stock.
  - REDUNDANT_POST provides the single most-recent NaN row per stock that
    populates forecast_data; the other REDUNDANT_POST rows are excluded.
  - GAP rows (obs warmup + leakage) are excluded from all outputs.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ml_markowitz.models.lgbm import generate_train_test_forecast

# ---------------------------------------------------------------------------
# Test-local temporal configuration
# ---------------------------------------------------------------------------

_DATA_START = pd.Timestamp("2010-01-01")  # first row of mock data (REDUNDANT_PRE)
_TRAIN_START = pd.Timestamp("2012-01-01")  # after 12m REDUNDANT_PRE + 12m obs warmup
_TRAIN_END = pd.Timestamp("2015-12-01")  # 48m after TRAIN_START (inclusive)
_TEST_START = pd.Timestamp("2017-01-01")  # TRAIN_END + 13m (12m leakage gap)
_TEST_END = pd.Timestamp("2017-12-01")  # 12m after TEST_START (inclusive)
# 2018-01 to 2018-12: FORECAST zone — valid targets (REDUNDANT_POST supplies forward data)
# 2019-01 to 2019-12: REDUNDANT_POST — NaN target rows → source of forecast_data

_N_STOCKS = 2
_STOCK_NAMES = ["A", "B"]
_N_MONTHS = 120

_ZONE_LABELS = (
    ["REDUNDANT_PRE"] * 12  # 2010-01 to 2010-12: before obs warmup
    + ["GAP"] * 12  # 2011-01 to 2011-12: obs warmup
    + ["TRAIN"] * 48  # 2012-01 to 2015-12: 4-year training period
    + ["GAP"] * 12  # 2016-01 to 2016-12: leakage gap
    + ["TEST"] * 12  # 2017-01 to 2017-12: 1-year test period
    + ["FORECAST"] * 12  # 2018-01 to 2018-12: valid targets, outside date masks
    + ["REDUNDANT_POST"] * 12  # 2019-01 to 2019-12: NaN targets → forecast_data
)

assert len(_ZONE_LABELS) == _N_MONTHS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dates() -> pd.DatetimeIndex:
    return pd.date_range(_DATA_START, periods=_N_MONTHS, freq="MS")


@pytest.fixture(scope="module")
def zone_labels(dates) -> pd.Series:
    """Zone label for each month in the mock dataset."""
    return pd.Series(_ZONE_LABELS, index=dates, name="zone")


@pytest.fixture(scope="module")
def mock_returns(dates) -> pd.DataFrame:
    """Synthetic monthly returns for two stocks over 120 months."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.normal(0.01, 0.05, (_N_MONTHS, _N_STOCKS)),
        index=dates,
        columns=_STOCK_NAMES,
    )


@pytest.fixture(scope="module")
def split_outputs(mock_returns):
    """Run generate_train_test_forecast with the test-local config dates."""
    with (
        patch("ml_markowitz.models.lgbm.TRAIN_START_DATE", _TRAIN_START),
        patch("ml_markowitz.models.lgbm.TRAIN_END_DATE", _TRAIN_END),
        patch("ml_markowitz.models.lgbm.TEST_START_DATE", _TEST_START),
        patch("ml_markowitz.models.lgbm.TEST_END_DATE", _TEST_END),
    ):
        return generate_train_test_forecast(mock_returns)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _unique_dates(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Sorted unique index values from a (possibly repeated) DatetimeIndex."""
    return df.index.unique().sort_values()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGenerateTrainTestForecast:
    # --- Zone correctness ---

    def test_train_contains_only_train_rows(self, split_outputs, zone_labels):
        train_data, _, _ = split_outputs
        zones = zone_labels.loc[_unique_dates(train_data)]
        assert (zones == "TRAIN").all(), f"Unexpected zones in train: {zones.unique()}"

    def test_test_contains_only_test_rows(self, split_outputs, zone_labels):
        _, test_data, _ = split_outputs
        zones = zone_labels.loc[_unique_dates(test_data)]
        assert (zones == "TEST").all(), f"Unexpected zones in test: {zones.unique()}"

    def test_forecast_data_contains_only_redundant_post_rows(self, split_outputs, zone_labels):
        # The NaN target rows are in REDUNDANT_POST (not the FORECAST zone), because
        # REDUNDANT_POST supplies the data needed to compute FORECAST zone targets.
        _, _, forecast_data = split_outputs
        zones = zone_labels.loc[_unique_dates(forecast_data)]
        assert (
            zones == "REDUNDANT_POST"
        ).all(), f"forecast_data should come from REDUNDANT_POST, got: {zones.unique()}"

    def test_gap_rows_absent_from_all_outputs(self, split_outputs, zone_labels):
        train_data, test_data, forecast_data = split_outputs
        all_dates = pd.Index(
            list(train_data.index) + list(test_data.index) + list(forecast_data.index)
        )
        output_zones = zone_labels.loc[all_dates]
        assert (output_zones != "GAP").all(), "GAP rows leaked into an output split"

    def test_redundant_pre_rows_absent_from_all_outputs(self, split_outputs, zone_labels):
        # Rows before the obs warmup must not appear in any output.
        train_data, test_data, forecast_data = split_outputs
        all_dates = pd.Index(
            list(train_data.index) + list(test_data.index) + list(forecast_data.index)
        )
        output_zones = zone_labels.loc[all_dates]
        assert (
            output_zones != "REDUNDANT_PRE"
        ).all(), "REDUNDANT_PRE rows (before obs warmup) leaked into an output split"

    def test_forecast_zone_rows_absent_from_all_outputs(self, split_outputs, zone_labels):
        # The FORECAST zone has valid targets (REDUNDANT_POST covers its forward window)
        # but falls outside all date masks, so it must not appear in any output.
        train_data, test_data, forecast_data = split_outputs
        all_dates = pd.Index(
            list(train_data.index) + list(test_data.index) + list(forecast_data.index)
        )
        output_zones = zone_labels.loc[all_dates]
        assert (
            output_zones != "FORECAST"
        ).all(), (
            "FORECAST zone rows (valid targets, outside date masks) leaked into an output split"
        )

    # --- Date range completeness ---

    def test_train_covers_full_train_period(self, split_outputs):
        train_data, _, _ = split_outputs
        unique_dates = _unique_dates(train_data)
        assert unique_dates[0] == _TRAIN_START, "Train does not start at TRAIN_START"
        assert unique_dates[-1] == _TRAIN_END, "Train does not end at TRAIN_END"
        expected_months = int(
            (_TRAIN_END.year - _TRAIN_START.year) * 12 + (_TRAIN_END.month - _TRAIN_START.month) + 1
        )
        assert len(unique_dates) == expected_months

    def test_test_covers_full_test_period(self, split_outputs):
        _, test_data, _ = split_outputs
        unique_dates = _unique_dates(test_data)
        assert unique_dates[0] == _TEST_START, "Test does not start at TEST_START"
        assert unique_dates[-1] == _TEST_END, "Test does not end at TEST_END"
        expected_months = int(
            (_TEST_END.year - _TEST_START.year) * 12 + (_TEST_END.month - _TEST_START.month) + 1
        )
        assert len(unique_dates) == expected_months

    # --- No overlap between splits ---

    def test_no_date_overlap_between_train_and_test(self, split_outputs):
        train_data, test_data, _ = split_outputs
        assert set(train_data.index).isdisjoint(set(test_data.index))

    def test_no_date_overlap_between_train_and_forecast(self, split_outputs):
        train_data, _, forecast_data = split_outputs
        assert set(train_data.index).isdisjoint(set(forecast_data.index))

    def test_no_date_overlap_between_test_and_forecast(self, split_outputs):
        _, test_data, forecast_data = split_outputs
        assert set(test_data.index).isdisjoint(set(forecast_data.index))

    # --- Schema ---

    def test_train_has_target_column(self, split_outputs):
        train_data, _, _ = split_outputs
        assert "target" in train_data.columns

    def test_test_has_target_column(self, split_outputs):
        _, test_data, _ = split_outputs
        assert "target" in test_data.columns

    def test_forecast_has_no_target_column(self, split_outputs):
        _, _, forecast_data = split_outputs
        assert "target" not in forecast_data.columns

    def test_train_target_has_no_nans(self, split_outputs):
        train_data, _, _ = split_outputs
        assert train_data["target"].notna().all()

    def test_test_target_has_no_nans(self, split_outputs):
        _, test_data, _ = split_outputs
        assert test_data["target"].notna().all()

    # --- Stock coverage ---

    def test_forecast_has_exactly_one_row_per_stock(self, split_outputs):
        _, _, forecast_data = split_outputs
        assert len(forecast_data) == _N_STOCKS

    def test_both_stocks_present_in_train(self, split_outputs):
        train_data, _, _ = split_outputs
        assert set(train_data["stock"].unique()) == set(_STOCK_NAMES)

    def test_both_stocks_present_in_test(self, split_outputs):
        _, test_data, _ = split_outputs
        assert set(test_data["stock"].unique()) == set(_STOCK_NAMES)

    def test_both_stocks_present_in_forecast(self, split_outputs):
        _, _, forecast_data = split_outputs
        assert set(forecast_data["stock"].unique()) == set(_STOCK_NAMES)
