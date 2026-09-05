#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立模型推理脚本

功能: 接收新的时间范围数据集, 自动加载模型, 执行推理流程以生成 power 值预测结果,
      并按照业务评测口径计算月度准确率指标。

模块结构:
  1. DataPreprocessor  — 数据预处理 (读取、校验、内生/外生分离)
  2. ModelLoader       — 模型加载 (实例化、超参注入、架构开关)
  3. InferenceEngine   — 推理执行 (训练、逐日预测、结果收集)
  4. AccuracyCalculator — 准确率计算 (weighted_accuracy、月度均值)

用法示例:
  # 基本用法 (使用 config 中的默认参数)
  python inference.py --data dataset.csv --config-path config/business_day_ahead_config.json

  # 指定模型超参和内生变量
  python inference.py --data dataset.csv \\
      --config-path config/business_day_ahead_config.json \\
      --model-hyper-params '{"lr":0.001,"d_model":256,"seq_len":576,"patch_len":96,"stride":48}' \\
      --endogenous-columns power,sr \\
      --target-column power

  # 从 checkpoint 加载已训练模型 (跳过训练)
  python inference.py --data dataset.csv --config-path config.json --checkpoint /path/to/model.pth

  # 自定义模型架构开关
  python inference.py --data dataset.csv --config-path config.json \\
      --use-c true --use-t true --use-c-exog true --use-t-exog true \\
      --fusion-method mlp --loss Huber
"""

import argparse
import calendar
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ts_benchmark 框架导入 (延迟导入, 仅在实际需要时加载)
_TSB_AVAILABLE = False
try:
    from ts_benchmark.models.model_loader import get_model_info, get_model_hyper_params
    from ts_benchmark.baselines.utils import train_val_split
    _TSB_AVAILABLE = True
except ImportError:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# 模块 1: 数据预处理
# =============================================================================
class DataPreprocessor:
    """
    数据预处理模块: 读取 CSV, 解析时间索引, 校验连续性, 分离内生/外生变量。

    数据集要求:
      - CSV 格式, 列名包含 'time' 和目标列 (如 'power')
      - 'time' 列解析为 pd.DatetimeIndex
      - 时间戳必须连续 15 分钟间隔
      - 列顺序: time, power, sr, [外生变量...]
    """

    def __init__(
        self,
        data_path: str,
        target_column: str = "power",
        endogenous_columns: Optional[List[str]] = None,
        time_column: str = "time",
    ):
        self.data_path = data_path
        self.target_column = target_column
        self.endogenous_columns = endogenous_columns or [target_column]
        self.time_column = time_column

    def load_and_validate(self) -> pd.DataFrame:
        """
        读取 CSV 并校验, 返回以时间为索引的 DataFrame。

        Returns
        -------
        pd.DataFrame
            以 DatetimeIndex 为索引的数据

        Raises
        ------
        FileNotFoundError
            文件不存在
        ValueError
            时间列缺失、时间不连续、目标列缺失等
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据集文件不存在: {self.data_path}")

        logger.info(f"读取数据集: {self.data_path}")
        df = pd.read_csv(self.data_path, low_memory=False)

        if self.time_column not in df.columns:
            raise ValueError(
                f"时间列 '{self.time_column}' 不在数据集中, 可用列: {list(df.columns)}"
            )

        if self.target_column not in df.columns:
            raise ValueError(
                f"目标列 '{self.target_column}' 不在数据集中, 可用列: {list(df.columns)}"
            )

        # 解析时间列
        df[self.time_column] = pd.to_datetime(df[self.time_column], errors="raise")
        df = df.set_index(self.time_column)
        df.index.name = "time"

        # 去重 (保留最后一条)
        if df.index.has_duplicates:
            n_dup = df.index.duplicated().sum()
            logger.warning(f"发现 {n_dup} 个重复时间戳, 保留最后一条")
            df = df[~df.index.duplicated(keep="last")]

        # 排序
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()

        # 校验连续性
        self._validate_continuity(df.index)

        logger.info(f"数据加载完成: {len(df)} 行, {len(df.columns)} 列, "
                     f"时间范围: {df.index[0]} ~ {df.index[-1]}")

        # 校验内生列
        for col in self.endogenous_columns:
            if col not in df.columns:
                raise ValueError(
                    f"内生变量列 '{col}' 不在数据集中, 可用列: {list(df.columns)}"
                )

        return df

    @staticmethod
    def _validate_continuity(index: pd.DatetimeIndex) -> None:
        """校验时间索引是否为连续 15 分钟间隔。"""
        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError("时间索引必须为 DatetimeIndex 类型")
        diffs = index.to_series().diff().dropna()
        if not diffs.eq(pd.Timedelta(minutes=15)).all():
            # 找出不连续的位置
            bad = diffs[~diffs.eq(pd.Timedelta(minutes=15))]
            raise ValueError(
                f"时间序列不连续, 要求 15 分钟间隔。"
                f"前 5 个异常间隔: {bad.head().tolist()}"
            )

    def split_channels(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        将 DataFrame 分为内生 (target) 和外生 (exog) 两部分。

        Parameters
        ----------
        df : pd.DataFrame
            完整数据集

        Returns
        -------
        target_df : pd.DataFrame
            内生变量 (endogenous_columns 指定的列)
        exog_df : Optional[pd.DataFrame]
            外生变量 (endogenous_columns 之外的列, 不含 target), 无外生时为 None
        """
        endo_idx = [df.columns.get_loc(c) for c in self.endogenous_columns]
        exog_idx = [i for i in range(len(df.columns)) if i not in endo_idx]

        target_df = df.iloc[:, endo_idx]
        exog_df = df.iloc[:, exog_idx] if exog_idx else None

        return target_df, exog_df

    def get_target_channel_idx(self, df: pd.DataFrame) -> List[int]:
        """获取目标列在内生 DataFrame 中的索引。"""
        return [self.endogenous_columns.index(self.target_column)]


# =============================================================================
# 模块 2: 模型加载
# =============================================================================
class ModelLoader:
    """
    模型加载模块: 实例化模型, 注入超参, 管理架构开关。

    支持两种模式:
      1. 全新训练: 实例化模型 → fit() → forecast()
      2. Checkpoint 加载: 实例化模型 → 加载权重 → forecast()
    """

    def __init__(
        self,
        model_name: str = "dag.DAG",
        model_hyper_params: Optional[Dict] = None,
        gpus: str = "0",
    ):
        self.model_name = model_name
        self.model_hyper_params = model_hyper_params or {}
        self.gpus = gpus

    def load(self) -> Any:
        """
        实例化并返回模型对象。

        Returns
        -------
        model
            符合 ModelBase 接口的模型实例

        Raises
        ------
        ImportError
            ts_benchmark 框架未安装
        RuntimeError
            模型实例化失败
        """
        if not _TSB_AVAILABLE:
            raise ImportError(
                "ts_benchmark 框架未安装, 请确保在正确的环境中运行此脚本"
            )

        # 设置 GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpus

        # 获取模型信息
        model_config = {"model_name": self.model_name}
        model_info = get_model_info(model_config)

        if isinstance(model_info, dict):
            model_factory = model_info["model_factory"]
            model_hp = dict(model_info.get("model_hyper_params", {}))
            model_hp.update(self.model_hyper_params)
        elif callable(model_info):
            model_factory = model_info
            model_hp = dict(self.model_hyper_params)
        else:
            raise RuntimeError(f"无法解析模型信息: {model_info}")

        logger.info(f"实例化模型: {self.model_name}, 超参: {model_hp}")

        try:
            model = model_factory(**model_hp)
        except Exception as e:
            raise RuntimeError(f"模型实例化失败: {e}") from e

        return model

    def load_checkpoint(self, model: Any, checkpoint_path: str) -> Any:
        """
        从 checkpoint 文件加载模型权重。

        Parameters
        ----------
        model : Any
            已实例化的模型对象
        checkpoint_path : str
            checkpoint 文件路径 (.pth / .pt)

        Returns
        -------
        model : Any
            加载权重后的模型对象
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint 文件不存在: {checkpoint_path}")

        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if "Model" in checkpoint:
            model.model.load_state_dict(checkpoint["Model"])
            logger.info(f"已加载模型权重: {checkpoint_path}")
            if (
                hasattr(model, "CovariateFusion")
                and model.CovariateFusion is not None
                and "CovariateFusion" in checkpoint
            ):
                model.CovariateFusion.load_state_dict(checkpoint["CovariateFusion"])
                logger.info("已加载 CovariateFusion 权重")
        else:
            model.model.load_state_dict(checkpoint)
            logger.info(f"已加载模型权重 (直接 state_dict): {checkpoint_path}")

        model.check_point = checkpoint
        return model


# =============================================================================
# 模块 3: 推理执行
# =============================================================================
class InferenceEngine:
    """
    推理执行模块: 训练模型, 逐日构造预测窗口, 调用模型推理, 收集结果。

    预测流程 (与 business_day_ahead.py 一致):
      1. 在评测月之前的数据上训练模型 (train_ratio_in_tv 划分训练/验证)
      2. 遍历评测月每一天, 构造 history (发布时刻之前的全部数据)
      3. 调用 model.forecast 生成 horizon 个点的预测
      4. 取后 evaluation_horizon 个点与真实值对比
    """

    def __init__(
        self,
        config: Dict,
        target_column: str = "power",
        endogenous_columns: Optional[List[str]] = None,
        capacity: Optional[float] = None,
    ):
        self.config = config
        self.target_column = target_column
        self.endogenous_columns = endogenous_columns or [target_column]

        # 从 config 读取评测参数
        self.horizon = config.get("horizon", 156)
        self.evaluation_horizon = config.get("evaluation_horizon", 96)
        self.issue_hour = config.get("issue_hour", 9)
        self.evaluation_year = config.get("evaluation_year", 2026)
        self.evaluation_month = config.get("evaluation_month", 6)
        self.train_ratio_in_tv = config.get("train_ratio_in_tv", 0.875)
        self.exclude_days = self._parse_exclude_days(config.get("exclude_days", []))

        # capacity 可以是 dict 或标量
        if capacity is not None:
            self.capacity = capacity
        else:
            cap = config.get("capacity", {})
            if isinstance(cap, dict):
                self.capacity = list(cap.values())[0] if cap else 100.0
            else:
                self.capacity = cap

        # 校验 horizon
        points_before_midnight = (24 - self.issue_hour) * 4
        expected_horizon = points_before_midnight + self.evaluation_horizon
        if self.horizon != expected_horizon:
            logger.warning(
                f"horizon={self.horizon} 与期望值 {expected_horizon} 不匹配, "
                f"已自动修正"
            )
            self.horizon = expected_horizon

    @staticmethod
    def _parse_exclude_days(exclude_days: Union[List, str]) -> set:
        """解析 exclude_days 配置为 YYYY-MM-DD 字符串集合。"""
        if isinstance(exclude_days, str):
            exclude_days = [exclude_days]
        return {pd.Timestamp(d).strftime("%Y-%m-%d") for d in exclude_days}

    def train_model(
        self,
        model: Any,
        train_valid_data: pd.DataFrame,
        exog_train_valid: Optional[pd.DataFrame],
    ) -> Any:
        """
        训练模型。

        Parameters
        ----------
        model : Any
            模型实例
        train_valid_data : pd.DataFrame
            内生变量训练数据 (评测月之前)
        exog_train_valid : Optional[pd.DataFrame]
            外生变量训练数据

        Returns
        -------
        model : Any
            训练后的模型
        """
        fit_covariates = {"exog": exog_train_valid} if exog_train_valid is not None else {}
        fit_method = (
            model.forecast_fit if hasattr(model, "forecast_fit") else model.fit
        )

        logger.info(
            f"开始训练: 内生数据 {train_valid_data.shape}"
            f"{f', 外生数据 {exog_train_valid.shape}' if exog_train_valid is not None else ''}"
        )

        start_time = time.time()
        fit_method(
            train_valid_data,
            covariates=fit_covariates,
            train_ratio_in_tv=self.train_ratio_in_tv,
        )
        elapsed = time.time() - start_time
        logger.info(f"训练完成, 耗时 {elapsed:.1f}s")

        return model

    def run(
        self,
        model: Any,
        series: pd.DataFrame,
        endogenous_columns: List[str],
    ) -> Tuple[List, Dict[str, float], List[pd.DataFrame], List[pd.DataFrame]]:
        """
        执行逐日推理。

        Parameters
        ----------
        model : Any
            已训练的模型
        series : pd.DataFrame
            完整数据集 (含内生+外生)
        endogenous_columns : List[str]
            内生列名列表

        Returns
        -------
        all_actual : list[pd.DataFrame]
            每天的真实值
        daily_accuracy : dict
            {日期字符串: 准确率}
        all_predicted : list[pd.DataFrame]
            每天的预测值
        eval_days : list[str]
            实际参与评测的日期列表
        """
        endo_idx = [series.columns.get_loc(c) for c in endogenous_columns]
        target_idx = [series.columns.get_loc(self.target_column)]

        month_start = pd.Timestamp(self.evaluation_year, self.evaluation_month, 1)
        month_end = month_start + pd.offsets.MonthBegin(1)
        month_days = pd.date_range(month_start, month_end, freq="D", inclusive="left")

        all_actual = []
        all_predicted = []
        daily_accuracy = {}
        eval_days = []

        logger.info(
            f"开始推理: 评测月 {self.evaluation_year}-{self.evaluation_month:02d}, "
            f"共 {len(month_days)} 天, 排除 {len(self.exclude_days)} 天"
        )

        for evaluation_day in month_days:
            if evaluation_day.strftime("%Y-%m-%d") in self.exclude_days:
                continue

            # 构造预测窗口
            forecast_start = evaluation_day - pd.Timedelta(days=1) + pd.Timedelta(
                hours=self.issue_hour
            )
            forecast_end = forecast_start + pd.Timedelta(minutes=15 * self.horizon)
            evaluation_end = evaluation_day + pd.Timedelta(days=1)

            forecast_frame = series.loc[
                (series.index >= forecast_start) & (series.index < forecast_end)
            ]
            evaluation_frame = series.loc[
                (series.index >= evaluation_day) & (series.index < evaluation_end)
            ]

            if len(forecast_frame) != self.horizon or len(evaluation_frame) != self.evaluation_horizon:
                logger.warning(
                    f"{evaluation_day.date()}: 数据不完整 "
                    f"(forecast={len(forecast_frame)}/{self.horizon}, "
                    f"eval={len(evaluation_frame)}/{self.evaluation_horizon}), 跳过"
                )
                continue

            # 准备历史数据
            history = series.loc[series.index < forecast_start]
            target_history, exog_history = self._split_channel(history, endo_idx)
            _, exog_future = self._split_channel(forecast_frame, endo_idx)

            # 构造协变量
            forecast_covariates = {}
            if exog_history is not None:
                if exog_future is None:
                    raise ValueError("外生变量有历史值但缺少未来值")
                forecast_covariates["exog"] = pd.concat([exog_history, exog_future])
                forecast_covariates["exog_future"] = exog_future

            # 调用模型推理
            start_inference = time.time()
            prediction = self._forecast(
                model, target_history, exog_history, exog_future, forecast_covariates
            )
            inference_time = time.time() - start_inference

            if prediction.shape[0] != self.horizon:
                raise ValueError(
                    f"模型返回 {prediction.shape[0]} 个点, 期望 {self.horizon}"
                )

            # 取后 evaluation_horizon 个点
            evaluated_prediction = prediction[-self.evaluation_horizon:]
            target_evaluation, _ = self._split_channel(evaluation_frame, target_idx)
            actual = target_evaluation.to_numpy()

            if evaluated_prediction.ndim == 1:
                evaluated_prediction = evaluated_prediction[:, None]

            # 计算准确率
            accuracy = AccuracyCalculator.weighted_accuracy(
                actual, evaluated_prediction, self.capacity
            )
            day_str = evaluation_day.strftime("%Y-%m-%d")
            daily_accuracy[day_str] = accuracy
            eval_days.append(day_str)

            all_actual.append(target_evaluation)
            all_predicted.append(
                pd.DataFrame(
                    evaluated_prediction,
                    columns=target_evaluation.columns,
                    index=target_evaluation.index,
                )
            )

            logger.info(
                f"{day_str}: accuracy={accuracy:.4f}, inference={inference_time:.2f}s"
            )

        if not daily_accuracy:
            raise ValueError(
                f"评测月 {self.evaluation_year}-{self.evaluation_month:02d} "
                f"没有完整的日度评测窗口"
            )

        logger.info(
            f"推理完成: {len(daily_accuracy)} 天, "
            f"月度均值: {np.mean(list(daily_accuracy.values())):.4f}"
        )

        return all_actual, daily_accuracy, all_predicted, eval_days

    @staticmethod
    def _split_channel(
        df: pd.DataFrame, target_idx: List[int]
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        将 DataFrame 按 target_idx 拆分为 target 和 exog 两部分。

        与 ts_benchmark.utils.data_processing.split_channel 逻辑一致。
        """
        from ts_benchmark.utils.data_processing import _parse_target_channel

        target_columns = _parse_target_channel(target_idx, df.shape[1])
        if target_idx is not None:
            all_columns = set(range(df.shape[1]))
            exog_columns = sorted(all_columns - set(target_columns))
        else:
            exog_columns = []

        target_df = df.iloc[:, target_columns]
        exog_df = df.iloc[:, exog_columns] if exog_columns else None

        return target_df, exog_df

    def _forecast(
        self,
        model: Any,
        target_history: pd.DataFrame,
        exog_history: Optional[pd.DataFrame],
        exog_future: Optional[pd.DataFrame],
        forecast_covariates: dict,
    ) -> np.ndarray:
        """
        调用模型推理, 与 business_day_ahead._forecast 逻辑一致。

        优先使用 batch_forecast (效率更高), 回退到 forecast。
        """
        prediction = model.forecast(
            self.horizon, target_history, covariates=forecast_covariates
        )
        return np.asarray(prediction)


# =============================================================================
# 模块 4: 准确率计算
# =============================================================================
class AccuracyCalculator:
    """
    准确率计算模块: 实现 weighted_accuracy 和月度均值计算。

    计算口径与 business_day_ahead.py 完全一致:
      1. 每日 accuracy = 1 - weighted_rmse / capacity
         weighted_rmse = sqrt(sum(error² × weights)), weights = |error| / sum(|error|)
      2. 月度均值 = mean(所有日 accuracy)
    """

    @staticmethod
    def weighted_accuracy(
        actual: np.ndarray, predicted: np.ndarray, cap: float
    ) -> float:
        """
        计算加权准确率 (业务口径)。

        Parameters
        ----------
        actual : np.ndarray
            真实值 (原始尺度)
        predicted : np.ndarray
            预测值 (原始尺度, norm=true 时已 inverse_transform)
        cap : float
            装机容量 (MW)

        Returns
        -------
        float
            准确率, 范围 (-inf, 1.0]
        """
        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)

        if actual.shape != predicted.shape:
            raise ValueError(
                f"真实值与预测值形状不匹配: {actual.shape} != {predicted.shape}"
            )
        if cap <= 0:
            raise ValueError(f"装机容量必须大于 0, 当前: {cap}")

        abs_errors = np.abs(actual - predicted)
        total_abs_error = np.sum(abs_errors)

        if total_abs_error == 0:
            return 1.0

        weights = abs_errors / total_abs_error
        weighted_rmse = np.sqrt(np.sum(np.square(actual - predicted) * weights))

        return float(1 - weighted_rmse / cap)

    @staticmethod
    def monthly_mean(daily_accuracy: Dict[str, float]) -> float:
        """
        计算月度准确率均值。

        Parameters
        ----------
        daily_accuracy : dict
            {日期字符串: 当日准确率}

        Returns
        -------
        float
            月度准确率均值
        """
        if not daily_accuracy:
            return float("nan")
        return float(np.mean(list(daily_accuracy.values())))

    @staticmethod
    def get_monthly_metric_name(month: int) -> str:
        """
        根据月份生成指标名 (与 business_day_ahead.field_names 一致)。

        Parameters
        ----------
        month : int
            月份 (1-12)

        Returns
        -------
        str
            如 "june_accuracy_mean"
        """
        try:
            month_name = calendar.month_name[int(month)].lower()
            return f"{month_name}_accuracy_mean"
        except Exception:
            return "march_accuracy_mean"


# =============================================================================
# 结果保存
# =============================================================================
class ResultSaver:
    """结果保存模块: 将预测结果和准确率保存为 CSV。"""

    @staticmethod
    def save_prediction_detail(
        all_actual: List[pd.DataFrame],
        all_predicted: List[pd.DataFrame],
        daily_accuracy: Dict[str, float],
        output_dir: str,
        series_name: str = "series",
    ) -> str:
        """
        保存逐点预测对比 CSV。

        文件: {output_dir}/prediction_detail_{series_name}.csv
        列: date, time, actual_power, predicted_power, error, daily_accuracy
        """
        os.makedirs(output_dir, exist_ok=True)

        rows = []
        for actual_df, pred_df in zip(all_actual, all_predicted):
            date_str = actual_df.index[0].strftime("%Y-%m-%d")
            acc = daily_accuracy.get(date_str, float("nan"))

            for t, (a, p) in enumerate(
                zip(actual_df.values, pred_df.values)
            ):
                for col_idx, col_name in enumerate(actual_df.columns):
                    rows.append(
                        {
                            "date": date_str,
                            "time": actual_df.index[t].strftime("%Y-%m-%d %H:%M:%S"),
                            f"actual_{col_name}": a[col_idx],
                            f"predicted_{col_name}": p[col_idx],
                            "error": a[col_idx] - p[col_idx],
                            "daily_accuracy": acc,
                        }
                    )

        detail_path = os.path.join(
            output_dir, f"prediction_detail_{series_name}.csv"
        )
        pd.DataFrame(rows).to_csv(detail_path, index=False)
        logger.info(f"逐点预测已保存: {detail_path}")
        return detail_path

    @staticmethod
    def save_daily_summary(
        daily_accuracy: Dict[str, float],
        monthly_mean: float,
        output_dir: str,
        series_name: str = "series",
        metric_name: str = "june_accuracy_mean",
    ) -> str:
        """
        保存日度准确率汇总 CSV。

        文件: {output_dir}/daily_summary_{series_name}.csv
        列: date, daily_accuracy
        末尾: monthly_mean 行
        """
        os.makedirs(output_dir, exist_ok=True)

        rows = [
            {"date": day, "daily_accuracy": acc}
            for day, acc in sorted(daily_accuracy.items())
        ]
        rows.append(
            {"date": f"MONTHLY_MEAN ({metric_name})", "daily_accuracy": monthly_mean}
        )

        summary_path = os.path.join(
            output_dir, f"daily_summary_{series_name}.csv"
        )
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        logger.info(f"日度准确率已保存: {summary_path}")
        return summary_path


# =============================================================================
# 配置加载
# =============================================================================
def load_config(config_path: str) -> Dict:
    """
    加载策略配置 JSON 文件。

    Parameters
    ----------
    config_path : str
        配置文件路径

    Returns
        Dict
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 支持 __default__ 和 per-series 配置
    if "__default__" in config:
        return config["__default__"]
    return config


# =============================================================================
# 主流程
# =============================================================================
def run_inference(args: argparse.Namespace) -> None:
    """
    完整推理流程: 加载数据 → 实例化模型 → 训练 → 逐日推理 → 计算准确率 → 保存结果。

    Parameters
    ----------
    args : argparse.Namespace
        命令行参数
    """
    # --- 1. 加载配置 ---
    config = load_config(args.config_path)

    # 合并命令行覆盖
    if args.evaluation_year is not None:
        config["evaluation_year"] = args.evaluation_year
    if args.evaluation_month is not None:
        config["evaluation_month"] = args.evaluation_month
    if args.capacity is not None:
        config["capacity"] = args.capacity
    if args.horizon is not None:
        config["horizon"] = args.horizon
    if args.issue_hour is not None:
        config["issue_hour"] = args.issue_hour

    # --- 2. 解析内生/外生变量 ---
    target_column = args.target_column or config.get("target_column", "power")
    if args.endogenous_columns:
        endogenous_columns = args.endogenous_columns.split(",")
    else:
        endogenous_columns = config.get(
            "endogenous_columns", [target_column]
        )

    # --- 3. 解析模型超参 ---
    model_hp = {}
    if args.model_hyper_params:
        model_hp = json.loads(args.model_hyper_params)

    # 注入架构开关 (命令行参数优先)
    arch_flags = {
        "use_c": args.use_c,
        "use_c_exog": args.use_c_exog,
        "use_t": args.use_t,
        "use_t_exog": args.use_t_exog,
        "fusion_method": args.fusion_method,
        "loss": args.loss,
        "norm": args.norm,
        "horizon": config.get("horizon", 156),
    }
    for k, v in arch_flags.items():
        if v is not None:
            model_hp[k] = v

    # pred_dim: 只预测 power (内生变量中的第一个目标列)
    if "pred_dim" not in model_hp:
        model_hp["pred_dim"] = 1

    # --- 4. 数据预处理 ---
    preprocessor = DataPreprocessor(
        data_path=args.data,
        target_column=target_column,
        endogenous_columns=endogenous_columns,
    )
    series = preprocessor.load_and_validate()

    # --- 5. 模型加载 ---
    loader = ModelLoader(
        model_name=args.model_name,
        model_hyper_params=model_hp,
        gpus=args.gpus,
    )
    model = loader.load()

    # 从 checkpoint 加载 (如果指定)
    if args.checkpoint:
        model = loader.load_checkpoint(model, args.checkpoint)
        logger.info("从 checkpoint 加载模型, 跳过训练")
    else:
        # --- 6. 训练模型 ---
        engine = InferenceEngine(
            config=config,
            target_column=target_column,
            endogenous_columns=endogenous_columns,
        )
        month_start = pd.Timestamp(
            engine.evaluation_year, engine.evaluation_month, 1
        )
        train_valid_data = series.loc[series.index < month_start]
        if train_valid_data.empty:
            raise ValueError(
                f"评测月 {engine.evaluation_year}-{engine.evaluation_month:02d} "
                f"之前没有训练数据"
            )

        target_train_valid, exog_train_valid = preprocessor.split_channels(
            train_valid_data
        )
        model = engine.train_model(
            model, target_train_valid, exog_train_valid
        )

    # --- 7. 逐日推理 ---
    if not args.checkpoint:
        # 使用已初始化的 engine
        pass
    else:
        engine = InferenceEngine(
            config=config,
            target_column=target_column,
            endogenous_columns=endogenous_columns,
        )

    all_actual, daily_accuracy, all_predicted, eval_days = engine.run(
        model=model,
        series=series,
        endogenous_columns=endogenous_columns,
    )

    # --- 8. 计算准确率 ---
    monthly_mean = AccuracyCalculator.monthly_mean(daily_accuracy)
    metric_name = AccuracyCalculator.get_monthly_metric_name(
        engine.evaluation_month
    )

    # --- 9. 输出结果 ---
    print("\n" + "=" * 60)
    print("推理结果汇总")
    print("=" * 60)
    print(f"数据集:         {args.data}")
    print(f"模型:           {args.model_name}")
    print(f"评测月份:       {engine.evaluation_year}-{engine.evaluation_month:02d}")
    print(f"评测天数:       {len(daily_accuracy)} 天")
    print(f"排除天数:       {len(engine.exclude_days)} 天")
    print(f"装机容量:       {engine.capacity} MW")
    print(f"{metric_name}:  {monthly_mean:.4f}")
    print(f"最高日精度:     {max(daily_accuracy.values()):.4f} "
          f"({max(daily_accuracy, key=daily_accuracy.get)})")
    print(f"最低日精度:     {min(daily_accuracy.values()):.4f} "
          f"({min(daily_accuracy, key=daily_accuracy.get)})")
    print("=" * 60)

    # 保存结果
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.data), "inference_results"
    )
    series_name = os.path.splitext(os.path.basename(args.data))[0]

    ResultSaver.save_prediction_detail(
        all_actual, all_predicted, daily_accuracy, output_dir, series_name
    )
    ResultSaver.save_daily_summary(
        daily_accuracy, monthly_mean, output_dir, series_name, metric_name
    )

    print(f"\n结果已保存到: {output_dir}/")


# =============================================================================
# 命令行入口
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="独立模型推理: 数据输入 → 模型训练/加载 → 逐日预测 → 准确率计算",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本推理
  python inference.py --data dataset.csv \\
      --config-path config/business_day_ahead_config.json

  # 指定超参和内生变量
  python inference.py --data dataset.csv \\
      --config-path config.json \\
      --model-hyper-params '{"lr":0.001,"d_model":256}' \\
      --endogenous-columns power,sr

  # 从 checkpoint 加载
  python inference.py --data dataset.csv \\
      --config-path config.json \\
      --checkpoint /path/to/model.pth

  # 自定义架构开关
  python inference.py --data dataset.csv --config-path config.json \\
      --use-c true --use-t true --fusion-method mlp --loss Huber --norm true
        """,
    )

    # 数据参数
    parser.add_argument("--data", "-d", required=True, help="数据集 CSV 文件路径")
    parser.add_argument(
        "--config-path", "-c", required=True, help="策略配置 JSON 文件路径"
    )
    parser.add_argument(
        "--target-column", default=None, help="预测目标列名 (默认: power)"
    )
    parser.add_argument(
        "--endogenous-columns",
        default=None,
        help="内生变量列名, 逗号分隔 (如 power,sr)",
    )

    # 模型参数
    parser.add_argument(
        "--model-name", default="dag.DAG", help="模型名称 (默认: dag.DAG)"
    )
    parser.add_argument(
        "--model-hyper-params",
        default=None,
        help="模型超参 JSON 字符串",
    )
    parser.add_argument("--gpus", default="0", help="GPU 编号 (默认: 0)")
    parser.add_argument(
        "--checkpoint", default=None, help="模型 checkpoint 路径 (指定则跳过训练)"
    )

    # 架构开关
    parser.add_argument(
        "--use-c", type=str, default=None, help="是否使用 CC encoder (true/false)"
    )
    parser.add_argument(
        "--use-t", type=str, default=None, help="是否使用 TC encoder (true/false)"
    )
    parser.add_argument(
        "--use-c-exog", type=str, default=None, help="CC 是否使用外生变量 (true/false)"
    )
    parser.add_argument(
        "--use-t-exog", type=str, default=None, help="TC 是否使用外生变量 (true/false)"
    )
    parser.add_argument(
        "--fusion-method", type=str, default=None,
        help="协变量融合方法 (mlp/conv/cross_attention/空字符串)"
    )
    parser.add_argument(
        "--loss", type=str, default=None,
        help="损失函数 (MSE/MAE/Huber)"
    )
    parser.add_argument(
        "--norm", type=str, default=None,
        help="是否标准化数据 (true/false)"
    )

    # 评测参数覆盖
    parser.add_argument("--evaluation-year", type=int, default=None, help="评测年份")
    parser.add_argument("--evaluation-month", type=int, default=None, help="评测月份")
    parser.add_argument("--capacity", type=float, default=None, help="装机容量 (MW)")
    parser.add_argument("--horizon", type=int, default=None, help="预测长度")
    parser.add_argument("--issue-hour", type=int, default=None, help="发布时刻")

    # 输出
    parser.add_argument(
        "--output-dir", default=None, help="结果输出目录 (默认: 数据集同目录/inference_results)"
    )

    # 日志
    parser.add_argument(
        "--log-level", default="INFO", help="日志级别 (DEBUG/INFO/WARNING/ERROR)"
    )

    args = parser.parse_args()

    # 解析布尔参数
    bool_map = {"true": True, "false": False, "1": True, "0": False}
    for attr in ["use_c", "use_c_exog", "use_t", "use_t_exog", "norm"]:
        val = getattr(args, attr)
        if val is not None:
            setattr(args, attr, bool_map.get(val.lower(), val))

    # 配置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s(%(lineno)d): %(message)s",
    )

    # 执行
    try:
        run_inference(args)
    except FileNotFoundError as e:
        print(f"[错误] 文件不存在: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"[错误] 数据/配置错误: {e}", file=sys.stderr)
        sys.exit(1)
    except ImportError as e:
        print(f"[错误] 依赖缺失: {e}", file=sys.stderr)
        print("请确保在安装了 ts_benchmark 的环境中运行此脚本", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[错误] 推理失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
