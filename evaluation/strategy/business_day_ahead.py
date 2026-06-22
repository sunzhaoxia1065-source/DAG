# -*- coding: utf-8 -*-
import json
import time
from typing import List, Optional

import numpy as np
import pandas as pd

from ts_benchmark.evaluation.metrics import regression_metrics
from ts_benchmark.evaluation.strategy.constants import FieldNames
from ts_benchmark.evaluation.strategy.forecasting import ForecastingStrategy
from ts_benchmark.models import ModelFactory
from ts_benchmark.models.model_base import BatchMaker
from ts_benchmark.utils.data_processing import split_channel
from ts_benchmark.utils.business_forecast import slice_business_evaluation_window


DAILY_ACCURACY = "daily_accuracy"
MONTHLY_ACCURACY_MEAN = "march_accuracy_mean"


def weighted_accuracy(actual: np.ndarray, predicted: np.ndarray, cap: float) -> float:
    """Calculate the business accuracy metric on original-scale values."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError(
            f"Actual and predicted shapes differ: {actual.shape} != {predicted.shape}"
        )
    if cap <= 0:
        raise ValueError("cap must be greater than zero")

    abs_errors = np.abs(actual - predicted)
    total_abs_error = np.sum(abs_errors)
    if total_abs_error == 0:
        return 1.0
    weights = abs_errors / total_abs_error
    weighted_rmse = np.sqrt(np.sum(np.square(actual - predicted) * weights))
    return float(1 - weighted_rmse / cap)


class SingleWindowBatchMaker(BatchMaker):
    """Build one historical window for models that only support batch forecasting."""

    def __init__(
        self,
        target_history: pd.DataFrame,
        exog_history: Optional[pd.DataFrame],
    ):
        self.target_history = target_history
        self.exog_history = exog_history

    def make_batch(self, batch_size: int, win_size: int) -> dict:
        del batch_size
        if len(self.target_history) < win_size:
            raise ValueError(
                f"History has {len(self.target_history)} points, but model needs {win_size}"
            )
        target = self.target_history.iloc[-win_size:]
        covariates = {}
        if self.exog_history is not None:
            covariates["exog"] = self.exog_history.iloc[-win_size:].to_numpy()[None]
        return {
            "input": target.to_numpy()[None],
            "time_stamps": target.index.to_numpy()[None],
            "covariates": covariates,
        }

    def has_more_batches(self) -> bool:
        return False


class BusinessDayAheadForecast(ForecastingStrategy):
    """Evaluate 09:00 day-ahead forecasts on each complete day in a month."""

    REQUIRED_CONFIGS = [
        "horizon",
        "evaluation_horizon",
        "issue_hour",
        "evaluation_year",
        "evaluation_month",
        "capacity",
        "train_ratio_in_tv",
        "save_true_pred",
        "target_column",
    ]

    def _execute(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        model_factory: ModelFactory,
        series_name: str,
    ) -> List:
        del meta_info
        self._validate_series(series)

        horizon = self._get_scalar_config_value("horizon", series_name)
        evaluation_horizon = self._get_scalar_config_value(
            "evaluation_horizon", series_name
        )
        issue_hour = self._get_scalar_config_value("issue_hour", series_name)
        evaluation_year = self._get_scalar_config_value(
            "evaluation_year", series_name
        )
        evaluation_month = self._get_scalar_config_value(
            "evaluation_month", series_name
        )
        capacity = self._get_scalar_config_value("capacity", series_name)
        target_column = self._get_scalar_config_value("target_column", series_name)
        if target_column not in series.columns:
            raise ValueError(
                f"Target column {target_column!r} does not exist in {series_name}"
            )
        target_channel = [series.columns.get_loc(target_column)]
        train_ratio_in_tv = self._get_scalar_config_value(
            "train_ratio_in_tv", series_name
        )

        points_before_midnight = (24 - issue_hour) * 4
        expected_horizon = points_before_midnight + evaluation_horizon
        if horizon != expected_horizon:
            raise ValueError(
                f"horizon must be {expected_horizon} for a {issue_hour:02d}:00 issue "
                f"time and {evaluation_horizon}-point evaluation, got {horizon}"
            )

        month_start = pd.Timestamp(evaluation_year, evaluation_month, 1)
        month_end = month_start + pd.offsets.MonthBegin(1)
        train_valid_data = series.loc[series.index < month_start]
        if train_valid_data.empty:
            raise ValueError("No training data exists before the evaluation month")

        target_train_valid, exog_train_valid = split_channel(
            train_valid_data, target_channel
        )
        if target_train_valid.shape[1] != 1:
            raise ValueError("Business accuracy evaluation requires one target column")

        model = model_factory()
        fit_covariates = {"exog": exog_train_valid}
        start_fit_time = time.time()
        fit_method = model.forecast_fit if hasattr(model, "forecast_fit") else model.fit
        fit_method(
            target_train_valid,
            covariates=fit_covariates,
            train_ratio_in_tv=train_ratio_in_tv,
        )
        end_fit_time = time.time()

        eval_scaler = self._get_eval_scaler(
            target_train_valid, train_ratio_in_tv
        )
        month_days = pd.date_range(month_start, month_end, freq="D", inclusive="left")
        daily_accuracy = {}
        daily_metrics = []
        all_actual = []
        all_predicted = []
        inference_times = []
        metric_logs = []

        for evaluation_day in month_days:
            forecast_start = evaluation_day - pd.Timedelta(days=1) + pd.Timedelta(
                hours=issue_hour
            )
            forecast_end = forecast_start + pd.Timedelta(minutes=15 * horizon)
            evaluation_end = evaluation_day + pd.Timedelta(days=1)

            forecast_frame = series.loc[
                (series.index >= forecast_start) & (series.index < forecast_end)
            ]
            evaluation_frame = series.loc[
                (series.index >= evaluation_day) & (series.index < evaluation_end)
            ]
            if len(forecast_frame) != horizon or len(evaluation_frame) != evaluation_horizon:
                continue

            history = series.loc[series.index < forecast_start]
            target_history, exog_history = split_channel(history, target_channel)
            _, exog_future = split_channel(forecast_frame, target_channel)
            forecast_covariates = {
                "exog": self._append_future_exog(exog_history, exog_future),
                "exog_future": exog_future,
            }

            start_inference_time = time.time()
            prediction = self._forecast(
                model,
                horizon,
                target_history,
                exog_history,
                exog_future,
                forecast_covariates,
            )
            inference_times.append(time.time() - start_inference_time)
            if prediction.shape[0] != horizon:
                raise ValueError(
                    f"Model returned {prediction.shape[0]} points, expected {horizon}"
                )

            evaluated_prediction = slice_business_evaluation_window(
                prediction, evaluation_horizon
            )
            target_evaluation, _ = split_channel(evaluation_frame, target_channel)
            actual = target_evaluation.to_numpy()
            if evaluated_prediction.ndim == 1:
                evaluated_prediction = evaluated_prediction[:, None]

            metric_values, log_info = self.evaluator.evaluate_with_log(
                actual,
                evaluated_prediction,
                eval_scaler,
                target_train_valid.values,
            )
            daily_metrics.append(metric_values)
            if log_info:
                metric_logs.append(f"{evaluation_day.date()}: {log_info}")

            accuracy = weighted_accuracy(actual, evaluated_prediction, capacity)
            daily_accuracy[evaluation_day.strftime("%Y-%m-%d")] = accuracy
            all_actual.append(target_evaluation)
            all_predicted.append(
                pd.DataFrame(
                    evaluated_prediction,
                    columns=target_evaluation.columns,
                    index=target_evaluation.index,
                )
            )

        if not daily_accuracy:
            raise ValueError(
                f"No complete {evaluation_year:04d}-{evaluation_month:02d} daily "
                "evaluation windows were found"
            )

        average_metrics = np.mean(np.asarray(daily_metrics, dtype=float), axis=0).tolist()
        monthly_accuracy_mean = float(np.mean(list(daily_accuracy.values())))
        save_true_pred = self._get_scalar_config_value("save_true_pred", series_name)
        actual_data = self._encode_data(all_actual) if save_true_pred else np.nan
        inference_data = self._encode_data(all_predicted) if save_true_pred else np.nan

        return average_metrics + [
            json.dumps(daily_accuracy, ensure_ascii=True, sort_keys=True),
            monthly_accuracy_mean,
            series_name,
            end_fit_time - start_fit_time,
            float(np.mean(inference_times)),
            actual_data,
            inference_data,
            "\n".join(metric_logs),
        ]

    @staticmethod
    def _forecast(
        model,
        horizon: int,
        target_history: pd.DataFrame,
        exog_history: Optional[pd.DataFrame],
        exog_future: Optional[pd.DataFrame],
        forecast_covariates: dict,
    ) -> np.ndarray:
        if not model.batch_forecast.__annotations__.get("not_implemented_batch"):
            batch_maker = SingleWindowBatchMaker(target_history, exog_history)
            future = None if exog_future is None else exog_future.to_numpy()[None]
            prediction = model.batch_forecast(
                horizon, batch_maker, future, 0
            )
            prediction = np.asarray(prediction)
            if prediction.ndim == 3 and prediction.shape[0] == 1:
                prediction = prediction[0]
            return prediction
        return np.asarray(
            model.forecast(horizon, target_history, covariates=forecast_covariates)
        )

    @staticmethod
    def _append_future_exog(
        exog_history: Optional[pd.DataFrame], exog_future: Optional[pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        if exog_history is None:
            return None
        if exog_future is None:
            raise ValueError("Future covariates are missing")
        return pd.concat([exog_history, exog_future])

    @staticmethod
    def _validate_series(series: pd.DataFrame) -> None:
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("Business evaluation requires a datetime index")
        if not series.index.is_monotonic_increasing:
            raise ValueError("Timestamps must be sorted in ascending order")
        if series.index.has_duplicates:
            raise ValueError("Timestamps must be unique")
        differences = series.index.to_series().diff().dropna()
        if not differences.eq(pd.Timedelta(minutes=15)).all():
            raise ValueError("Business evaluation requires continuous 15-minute data")

    @staticmethod
    def accepted_metrics():
        return regression_metrics.__all__

    @property
    def field_names(self) -> List[str]:
        return self.evaluator.metric_names + [
            DAILY_ACCURACY,
            MONTHLY_ACCURACY_MEAN,
            FieldNames.FILE_NAME,
            FieldNames.FIT_TIME,
            FieldNames.INFERENCE_TIME,
            FieldNames.ACTUAL_DATA,
            FieldNames.INFERENCE_DATA,
            FieldNames.LOG_INFO,
        ]
