from ml_markowitz.config import STOCKS
from ml_markowitz.data import get_monthly_returns_dividends
from ml_markowitz.models.lgbm import lgbm_pipeline

# from ml_markowitz.backtesting import backtest_models
# from ml_markowitz.models.arima import arima_pipeline
# from ml_markowitz.models.benchmark import benchmark_pipeline


def main():
    """Main function to run the ML Markowitz application."""

    # Load data
    monthly_returns = get_monthly_returns_dividends(STOCKS)

    # Train models and get test predictions and forecasts
    y_lgbm_test, mu_lgbm_forecast, hyperparams = lgbm_pipeline(monthly_returns)

    # y_arima_test, mu_arima_forecast = arima_pipeline(monthly_returns)
    # y_bench12_test, mu_12m = benchmark_pipeline(monthly_returns, window=12)
    # y_bench60_test, mu_60m = benchmark_pipeline(monthly_returns, window=60)

    # backtest_dict = {
    #     "lgbm" : y_lgbm_test,
    #     "arima" : y_arima_test,
    #     "12m_avg" : y_bench12_test,
    #     "60m_avg" : y_bench60_test,
    #     "ensemble_lgbm_12m" : (y_lgbm_test + y_bench12_test) / 2,
    # }

    # # Backtest models and get optimal model and backtest results
    # optimal_model, backtest_results = backtest_models(
    #     monthly_returns,
    #     backtest_dict,
    # )

    # Select forecast returns. Generate covariance matrix using the training data.

    # Apply Markowitz optimization to get optimal portfolio weights using
    # the optimized forecast returns.

    return


if __name__ == "__main__":
    main()
