import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ts_benchmark.data.utils import read_data
from ts_benchmark.evaluation.evaluator import Evaluator
from ts_benchmark.evaluation.strategy.business_day_ahead import (
    BusinessDayAheadForecast,
    weighted_accuracy,
)


class FutureCovariateModel:
    def __init__(self):
        self.fit_end = None
        self.forecast_horizons = []

    def forecast_fit(self, train_valid_data, *, covariates, train_ratio_in_tv):
        self.fit_end = train_valid_data.index.max()
        return self

    def forecast(self, horizon, series, *, covariates):
        self.forecast_horizons.append(horizon)
        return covariates["exog"].iloc[-horizon:, :1].to_numpy()

    def batch_forecast(self, horizon, batch_maker, exog_future, i):
        raise AssertionError("The sample forecast path should be used")


FutureCovariateModel.batch_forecast.__annotations__["not_implemented_batch"] = True


class BatchFutureCovariateModel(FutureCovariateModel):
    def batch_forecast(self, horizon, batch_maker, exog_future, i):
        self.forecast_horizons.append(horizon)
        batch = batch_maker.make_batch(batch_size=64, win_size=96)
        assert batch["input"].shape == (1, 96, 1)
        assert batch["covariates"]["exog"].shape == (1, 96, 1)
        return exog_future[:, -horizon:, :1]


BatchFutureCovariateModel.batch_forecast.__annotations__.pop(
    "not_implemented_batch", None
)


class BusinessDayAheadTest(unittest.TestCase):
    def test_weighted_accuracy_uses_absolute_error_weights(self):
        actual = np.array([[10.0], [20.0]])
        predicted = np.array([[8.0], [16.0]])
        expected_rmse = np.sqrt(2.0**2 * (2.0 / 6.0) + 4.0**2 * (4.0 / 6.0))
        self.assertAlmostEqual(
            weighted_accuracy(actual, predicted, cap=50.0),
            1.0 - expected_rmse / 50.0,
        )

    def test_weighted_accuracy_is_one_for_perfect_prediction(self):
        actual = np.array([[1.0], [2.0]])
        self.assertEqual(weighted_accuracy(actual, actual.copy(), cap=50.0), 1.0)

    def test_read_data_accepts_wide_business_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "business.csv"
            path.write_text(
                "time,power,ws\n"
                "2026-02-28 00:00:00,1.0,2.0\n"
                "2026-02-28 00:15:00,1.5,2.5\n",
                encoding="utf-8",
            )
            data = read_data(str(path))

        self.assertEqual(list(data.columns), ["power", "ws"])
        self.assertEqual(data.index.name, "time")
        self.assertEqual(data.iloc[1]["power"], 1.5)

    def test_business_strategy_evaluates_all_march_days(self):
        index = pd.date_range(
            "2026-01-01 00:00:00", "2026-04-01 00:00:00", freq="15min", inclusive="left"
        )
        values = np.arange(len(index), dtype=float)
        series = pd.DataFrame({"power": values, "ws": values}, index=index)
        model = FutureCovariateModel()
        strategy = BusinessDayAheadForecast(
            {
                "strategy_name": "business_day_ahead",
                "horizon": 156,
                "evaluation_horizon": 96,
                "issue_hour": 9,
                "evaluation_year": 2026,
                "evaluation_month": 3,
                "capacity": {"business.csv": 50.0},
                "train_ratio_in_tv": 0.875,
                "save_true_pred": False,
                "target_column": "power",
                "seed": 2021,
                "deterministic": "none",
            },
            Evaluator([{"name": "rmse"}]),
        )

        result = strategy._execute(series, None, lambda: model, "business.csv")
        result_by_name = dict(zip(strategy.field_names, result))
        daily_accuracy = __import__("json").loads(result_by_name["daily_accuracy"])

        self.assertEqual(len(daily_accuracy), 31)
        self.assertEqual(result_by_name["march_accuracy_mean"], 1.0)
        self.assertEqual(result_by_name["rmse"], 0.0)
        self.assertLess(model.fit_end, pd.Timestamp("2026-03-01"))
        self.assertEqual(model.forecast_horizons, [156] * 31)

    def test_business_strategy_supports_batch_only_models(self):
        index = pd.date_range(
            "2026-01-01", "2026-04-01", freq="15min", inclusive="left"
        )
        values = np.arange(len(index), dtype=float)
        series = pd.DataFrame({"power": values, "ws": values}, index=index)
        model = BatchFutureCovariateModel()
        strategy = BusinessDayAheadForecast(
            {
                "strategy_name": "business_day_ahead",
                "horizon": 156,
                "evaluation_horizon": 96,
                "issue_hour": 9,
                "evaluation_year": 2026,
                "evaluation_month": 3,
                "capacity": 50.0,
                "train_ratio_in_tv": 0.875,
                "save_true_pred": False,
                "target_column": "power",
                "seed": 2021,
                "deterministic": "none",
            },
            Evaluator([{"name": "rmse"}]),
        )

        result = strategy._execute(series, None, lambda: model, "business.csv")
        result_by_name = dict(zip(strategy.field_names, result))
        self.assertEqual(result_by_name["march_accuracy_mean"], 1.0)
        self.assertEqual(model.forecast_horizons, [156] * 31)


if __name__ == "__main__":
    unittest.main()
