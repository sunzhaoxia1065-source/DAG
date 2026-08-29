#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
两文件合并与处理工具 (自包含, 针对指定属性, 使用显式列分类)

文件a: time, power, sr (内生变量)
文件b: time + 多个气象属性, 只筛选用户需要的外生列, 默认 4 个:
  relatively_humidity_isobaric_950                     (never_zero)
  total_sky_direct_solar_radiation_at_surface_surface  (solar)
  surface_thermal_radiation_downwards_surface          (never_zero)
  dewpoint_temperature_surface_2metre                  (never_zero)

合并: 以文件a的时间为基准, 文件b向a对齐 (a left join b)
处理: 缺失值插补 + 异常0值按显式分类检测修正

列分类采用显式字典, 不依赖关键词匹配, 避免 "radiation" 把
surface_thermal_radiation_downwards_surface 误判为 solar (应为 never_zero)。

用法:
  python merge_and_process.py --file-a a.csv --file-b b.csv --output merged.csv
  python merge_and_process.py -a a.csv -b b.csv --test-cutoff 2026-03-01 -o merged.csv
  # 自定义文件b筛选列
  python merge_and_process.py -a a.csv -b b.csv --columns-b col1,col2 -o merged.csv
  python merge_and_process.py            # 交互式
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# 日志 (模块级 logger, main 中初始化 handler)
# =============================================================================
logger = logging.getLogger("merge_and_process")
if not logger.handlers:
    _ch = logging.StreamHandler(sys.stdout)
    _ch.setLevel(logging.INFO)
    _ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_ch)
    logger.setLevel(logging.INFO)


# =============================================================================
# 显式列分类 (不依赖关键词匹配, 避免 solar 的 "radiation" 抢占 thermal_radiation)
# - nighttime_zero: 内生变量, 夜间为0正常, 全天为0才算异常 (限电)
# - solar: 外生变量, 夜间为0正常, 白天全0才算异常
# - never_zero: 外生变量, 全天不应为0, 任何0都是异常
# - other: 其他 (不检测)
# =============================================================================
COLUMN_CLASSIFICATION: Dict[str, str] = {
    "power": "nighttime_zero",
    "sr": "nighttime_zero",
    "total_sky_direct_solar_radiation_at_surface_surface": "solar",
    "relatively_humidity_isobaric_950": "never_zero",
    "surface_thermal_radiation_downwards_surface": "never_zero",
    "dewpoint_temperature_surface_2metre": "never_zero",
}

# 关键词回退: 仅对未在显式字典中登记的列使用, 永远不会覆盖显式登记
_SOLAR_KEYWORDS = ["direct_solar", "日照"]
_NEVER_ZERO_KEYWORDS = ["thermal_radiation", "humidity", "热辐射", "湿度"]

# 周期长度: 1天 = 96 个 15min 点
PERIOD = 96

# 文件a必须包含的列
REQUIRED_COLUMNS_A = ["time", "power", "sr"]
# 文件b默认筛选的外生列 (用户需要的4个, 顺序即模型输入的外生列顺序)
DEFAULT_EXOG_COLUMNS_B = [
    "relatively_humidity_isobaric_950",
    "total_sky_direct_solar_radiation_at_surface_surface",
    "surface_thermal_radiation_downwards_surface",
    "dewpoint_temperature_surface_2metre",
]


# =============================================================================
# 列分类 (显式字典优先, 关键词仅作回退)
# =============================================================================
def classify_column(col: str) -> str:
    """根据显式字典分类列; 未登记的列用关键词回退; 都不命中返回 'other'."""
    if col in COLUMN_CLASSIFICATION:
        return COLUMN_CLASSIFICATION[col]
    col_lower = col.lower()
    if any(k in col_lower for k in _NEVER_ZERO_KEYWORDS):
        return "never_zero"
    if any(k in col_lower for k in _SOLAR_KEYWORDS):
        return "solar"
    return "other"


# =============================================================================
# 数据读取与列筛选
# =============================================================================
def read_and_select(filepath: str, required_cols: List[str], label: str,
                    select_exog: Optional[List[str]] = None) -> pd.DataFrame:
    """读取CSV并筛选列。

    - 文件a (select_exog is None): 保留 required_cols (time, power, sr)
    - 文件b (select_exog 给定): 保留 time + select_exog (指定外生列)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{label}不存在: {filepath}")

    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    logger.info(f"读取{label}: {filepath} (大小: {file_size_mb:.2f} MB)")

    try:
        df = pd.read_csv(filepath, low_memory=False)
    except pd.errors.EmptyDataError:
        raise ValueError(f"{label}为空: {filepath}")
    except Exception as e:
        raise ValueError(f"读取{label}失败 [{filepath}]: {e}")

    if df.empty:
        raise ValueError(f"{label}无有效数据: {filepath}")

    logger.info(f"{label}原始: {len(df)} 行, {len(df.columns)} 列, 列: {list(df.columns)}")

    if select_exog is not None:
        # 文件b: 保留 time + 指定外生列
        missing_exog = [c for c in select_exog if c not in df.columns]
        if missing_exog:
            raise ValueError(f"{label}缺少外生列: {missing_exog}, 可用: {list(df.columns)}")
        if "time" not in df.columns:
            raise ValueError(f"{label}缺少 time 列, 可用: {list(df.columns)}")
        keep = ["time"] + select_exog
    else:
        # 文件a: 保留 required_cols
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{label}缺少必要列: {missing}, 可用: {list(df.columns)}")
        keep = required_cols

    df = df[keep].copy()
    logger.info(f"{label}筛选后保留: {list(df.columns)}")
    return df


# =============================================================================
# 时间处理
# =============================================================================
def smart_parse_time(series: pd.Series) -> pd.Series:
    """智能解析时间列, 自动检测格式。

    优先检测 YYYYMMDDHHMM (12位纯数字) / YYYYMMDDHHMMSS (14位纯数字),
    避免被 pd.to_datetime 误当作 Unix 时间戳解析为 1970 年。
    其余格式由 pandas 自动推断。
    """
    sample = str(series.dropna().iloc[0]).strip() if series.notna().any() else ""
    # 12位纯数字 → YYYYMMDDHHMM
    if len(sample) == 12 and sample.isdigit():
        return pd.to_datetime(series.astype(str), format="%Y%m%d%H%M", errors="coerce")
    # 14位纯数字 → YYYYMMDDHHMMSS
    if len(sample) == 14 and sample.isdigit():
        return pd.to_datetime(series.astype(str), format="%Y%m%d%H%M%S", errors="coerce")
    # 其他格式由 pandas 自动解析
    parsed = pd.to_datetime(series, errors="coerce")
    # 安全回退: 若解析结果落在 1970 年附近, 说明被误当作时间戳
    if not parsed.isna().all() and parsed.min().year < 2000:
        retry = pd.to_datetime(series.astype(str), format="%Y%m%d%H%M", errors="coerce")
        if not retry.isna().all() and retry.min().year >= 2000:
            return retry
    return parsed


# =============================================================================
# 合并 (以文件a为基准)
# =============================================================================
def merge_on_a(df_a: pd.DataFrame, df_b: pd.DataFrame,
               selected_columns_b: List[str]) -> pd.DataFrame:
    """以文件a的时间为基准, 文件b向a对齐 (a left join b)。

    a缺失b有的时间丢弃; a有b无的时间, b列填NaN (后续插补)。
    """
    df_a = df_a.copy()
    df_b = df_b.copy()
    # 对齐到分钟, 去重
    df_a["time"] = df_a["time"].dt.floor("min")
    df_b["time"] = df_b["time"].dt.floor("min")
    df_a = df_a.drop_duplicates(subset="time", keep="last")
    df_b = df_b.drop_duplicates(subset="time", keep="last")

    df_b_subset = df_b[["time"] + selected_columns_b].copy()
    # 以a为基准: a left join b
    merged = df_a.merge(df_b_subset, on="time", how="left")
    merged = merged.sort_values("time").reset_index(drop=True)

    # 统计b列对齐情况
    b_aligned = merged[selected_columns_b].notna().all(axis=1).sum()
    logger.info(
        f"合并完成(以a为基准): 文件a {len(df_a)} 行, 文件b {len(df_b)} 行 "
        f"-> 合并后 {len(merged)} 行 (b全部对齐 {b_aligned} 行, "
        f"b部分缺失 {len(merged) - b_aligned} 行待插补)"
    )
    return merged


# =============================================================================
# 缺失值检测与处理
# =============================================================================
def detect_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """全面检测缺失值, 生成统计报告。"""
    total = len(df)
    missing_count = df.isna().sum()
    missing_ratio = missing_count / total if total else missing_count

    report = pd.DataFrame({
        "列名": df.columns,
        "缺失数量": missing_count.values,
        "缺失率": missing_ratio.values,
        "非空数量": (total - missing_count).values,
    })

    total_missing = int(missing_count.sum())
    logger.info(f"缺失值检测: 共 {total_missing} 个缺失值")
    for _, row in report.iterrows():
        if row["缺失数量"] > 0:
            logger.info(f"  列 '{row['列名']}': {row['缺失数量']} 缺失 ({row['缺失率']:.2%})")

    return report


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "interpolate",
    periodic_neighbors: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """处理缺失值。

    strategy:
    - "drop": 删除含缺失值的行
    - "fill_mean"/"fill_median": 均值/中位数填充
    - "interpolate": 线性插值 (默认)
    - "fill_periodic_mean": 周期插补, 用其他周期同位置均值填充
    - "ffill"/"bfill": 前/后向填充
    """
    df = df.copy()
    missing_before = int(df.isna().sum().sum())
    record: Dict = {"strategy": strategy, "missing_before": missing_before, "details": {}}

    if missing_before == 0:
        logger.info("无缺失值, 跳过处理")
        record["missing_after"] = 0
        return df, record

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    missing_per_col = df[numeric_cols].isna().sum()
    missing_per_col = missing_per_col[missing_per_col > 0]

    if strategy == "drop":
        before_len = len(df)
        df = df.dropna().reset_index(drop=True)
        record["details"]["dropped_rows"] = before_len - len(df)
        logger.info(f"删除缺失行: {before_len} -> {len(df)}")

    elif strategy in ("fill_mean", "fill_median"):
        for col in missing_per_col.index:
            fill_val = df[col].mean() if strategy == "fill_mean" else df[col].median()
            df[col] = df[col].fillna(fill_val)
            record["details"][col] = f"{strategy}={fill_val:.4f}"
        logger.info(f"用{'均值' if strategy == 'fill_mean' else '中位数'}填充缺失值")

    elif strategy == "interpolate":
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
        logger.info("线性插值填充缺失值")

    elif strategy == "fill_periodic_mean":
        # 周期插补: 用前后各 neighbors 个周期同位置均值填充
        neighbors = periodic_neighbors if periodic_neighbors is not None else 3
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            df = df.sort_values("time").reset_index(drop=True)

        for col in missing_per_col.index:
            na_mask = df[col].isna()
            if na_mask.sum() == 0:
                continue
            positions = np.arange(len(df)) % PERIOD
            col_values = df[col].copy()

            for pos in range(PERIOD):
                pos_mask = positions == pos
                pos_na = na_mask & pos_mask
                if pos_na.sum() == 0:
                    continue
                pos_valid = pos_mask & (~na_mask)
                valid_indices = np.where(pos_valid)[0]
                na_indices = np.where(pos_na)[0]

                if len(valid_indices) == 0:
                    continue

                fill_vals = []
                for na_idx in na_indices:
                    lo = na_idx - neighbors * PERIOD
                    hi = na_idx + neighbors * PERIOD
                    neighbor_valid = valid_indices[(valid_indices >= lo) & (valid_indices <= hi)]
                    if len(neighbor_valid) > 0:
                        fill_vals.append(col_values.iloc[neighbor_valid].mean())
                    else:
                        fill_vals.append(col_values.iloc[valid_indices].mean())
                df.loc[na_indices, col] = fill_vals

            filled_count = int(na_mask.sum() - df[col].isna().sum())
            record["details"][col] = f"periodic_mean(filled={filled_count})"
        logger.info(f"周期插补填充缺失值 (period={PERIOD}, neighbors={neighbors})")

    elif strategy == "ffill":
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
        logger.info("前向填充缺失值")

    elif strategy == "bfill":
        df[numeric_cols] = df[numeric_cols].bfill().ffill()
        logger.info("后向填充缺失值")

    else:
        raise ValueError(f"未知缺失值策略: {strategy}")

    missing_after = int(df.isna().sum().sum())
    record["missing_after"] = missing_after
    logger.info(f"缺失值处理: {missing_before} -> {missing_after}")

    return df, record


# =============================================================================
# 异常0值检测与处理 (按显式分类)
# =============================================================================
def detect_special_cases(
    df: pd.DataFrame,
    target_columns: List[str],
    test_cutoff: Optional[str] = None,
    zero_threshold: float = 0.0,
    time_col: str = "time",
    daytime_start: int = 7,
    daytime_end: int = 19,
) -> List[Dict]:
    """检测异常0值, 按列类型和昼夜规律分类处理。

    - nighttime_zero (power, sr): 全天为0才判定异常 (限电), 夜间为0正常。
      检测范围排除测试期数据。
    - solar: 只检测白天时段 (07:00~19:00) 是否全为0。夜间为0正常。
      检测范围排除测试期数据。
    - never_zero: 任何时刻为0即为异常。全时段检测, 不排除测试期
      (外生变量, 修正不影响评估公平性)。
    - other: 不检测。
    """
    if not target_columns:
        return []

    df = df.copy()
    df["_hour"] = df[time_col].dt.hour
    df["_date"] = df[time_col].dt.date

    test_cutoff_date = None
    if test_cutoff:
        test_cutoff_date = pd.Timestamp(test_cutoff).date()
        logger.info(f"测试期起始: {test_cutoff}, 内生变量不检测此日期之后的数据")

    results: List[Dict] = []

    for col in target_columns:
        if col not in df.columns:
            logger.warning(f"检测列不存在: '{col}', 跳过")
            continue

        col_type = classify_column(col)

        if col_type == "other":
            logger.info(f"列 '{col}' 分类为 'other', 跳过特殊检测")
            continue

        # 内生变量受测试期限制, 外生变量 (solar/never_zero) 不受限制
        is_endogenous = col_type == "nighttime_zero"

        for date, day_df in df.groupby("_date"):
            # 内生变量: 跳过测试期
            if is_endogenous and test_cutoff_date and date >= test_cutoff_date:
                continue

            is_abnormal = False

            if col_type == "nighttime_zero":
                # power, sr: 全天为0才算异常 (夜间为0正常但白天也0说明限电)
                is_abnormal = (day_df[col].abs() <= zero_threshold).all()

            elif col_type == "solar":
                # 太阳能类: 白天时段全为0才算异常
                day_hours = day_df[
                    (day_df["_hour"] >= daytime_start) & (day_df["_hour"] < daytime_end)
                ]
                if day_hours.empty:
                    continue
                is_abnormal = (day_hours[col].abs() <= zero_threshold).all()

            elif col_type == "never_zero":
                # 全天不应为0: 任何时刻为0都算异常
                zero_mask = day_df[col].abs() <= zero_threshold
                if zero_mask.any():
                    is_abnormal = True

            if is_abnormal:
                if col_type == "never_zero":
                    # never_zero: 只标记为0的那些行, 不是整天
                    zero_rows = day_df[day_df[col].abs() <= zero_threshold]
                    start_idx = zero_rows.index[0]
                    end_idx = zero_rows.index[-1]
                    start_time = df.loc[start_idx, time_col]
                    end_time = df.loc[end_idx, time_col]
                    results.append({
                        "column": col,
                        "col_type": col_type,
                        "date": str(date),
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration_hours": round(
                            (end_time - start_time).total_seconds() / 3600, 2
                        ),
                        "row_count": len(zero_rows),
                        "start_idx": int(start_idx),
                        "end_idx": int(end_idx),
                    })
                else:
                    # nighttime_zero / solar: 标记整天
                    start_time = day_df[time_col].iloc[0]
                    end_time = day_df[time_col].iloc[-1]
                    start_idx = day_df.index[0]
                    end_idx = day_df.index[-1]
                    results.append({
                        "column": col,
                        "col_type": col_type,
                        "date": str(date),
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration_hours": round(
                            (end_time - start_time).total_seconds() / 3600, 2
                        ),
                        "row_count": len(day_df),
                        "start_idx": int(start_idx),
                        "end_idx": int(end_idx),
                    })

    df = df.drop(columns=["_hour", "_date"])

    if results:
        logger.info(f"检测到 {len(results)} 个特殊情况:")
        for r in results:
            logger.info(
                f"  列 '{r['column']}' [{r['col_type']}]: "
                f"{r['date']}, {r['start_time']} ~ {r['end_time']}, "
                f"{r['row_count']} 行"
            )
    else:
        logger.info("未检测到特殊情况")

    return results


def handle_special_cases(
    df: pd.DataFrame,
    special_cases: List[Dict],
    strategy: str = "fill_periodic_mean",
    time_col: str = "time",
    **kwargs,
) -> Tuple[pd.DataFrame, Dict]:
    """处理检测到的特殊情况。

    - never_zero 类: 始终用插值替换异常0值 (忽略 strategy 参数)。
    - nighttime_zero / solar 类: 按 strategy 处理:
      drop / fill_mean / fill_median / interpolate / fill_periodic_mean / keep
    """
    df = df.copy()
    record: Dict = {"strategy": strategy, "cases_handled": 0, "details": []}

    if not special_cases:
        logger.info("特殊情况: 无需处理")
        return df, record

    # 分类处理
    never_zero_cases = [c for c in special_cases if c.get("col_type") == "never_zero"]
    other_cases = [c for c in special_cases if c.get("col_type") != "never_zero"]

    # ---- never_zero 类: 强制插值替换 ----
    if never_zero_cases:
        for case in never_zero_cases:
            col = case["column"]
            mask = (df.index >= case["start_idx"]) & (df.index <= case["end_idx"])
            zero_mask = mask & (df[col].abs() <= 1e-10)
            if zero_mask.any():
                df.loc[zero_mask, col] = np.nan
                record["cases_handled"] += 1
                count = int(zero_mask.sum())
                record["details"].append(
                    f"列 '{col}' {case['date']}: {count} 个异常0值 -> 插值替换"
                )

        # 对 never_zero 列做插值
        never_zero_cols = list(set(c["column"] for c in never_zero_cases))
        for col in never_zero_cols:
            if col in df.columns:
                df[col] = df[col].interpolate(method="linear", limit_direction="both")
                df[col] = df[col].ffill().bfill()

        logger.info(f"never_zero 类处理: {len(never_zero_cases)} 个区间, 插值替换")

    # ---- nighttime_zero / solar 类: 按用户策略处理 ----
    if not other_cases or strategy == "keep":
        if other_cases:
            logger.info("nighttime_zero/solar 类: 保留不处理")
        return df, record

    if strategy == "drop":
        drop_indices = set()
        for case in other_cases:
            drop_indices.update(range(case["start_idx"], case["end_idx"] + 1))
        existing_indices = set(df.index)
        drop_indices = drop_indices & existing_indices
        before_len = len(df)
        df = df.drop(index=drop_indices).reset_index(drop=True)
        record["cases_handled"] += len(other_cases)
        record["details"].append(f"删除 {len(drop_indices)} 行 (限电/太阳能异常)")
        logger.info(f"nighttime_zero/solar 处理(删除): {before_len} -> {len(df)} 行")

    elif strategy in ("fill_mean", "fill_median"):
        for case in other_cases:
            col = case["column"]
            mask = (df.index >= case["start_idx"]) & (df.index <= case["end_idx"])
            non_special = df.loc[~mask, col]
            fill_val = non_special.mean() if strategy == "fill_mean" else non_special.median()
            df.loc[mask, col] = fill_val
            record["cases_handled"] += 1
            record["details"].append(
                f"列 '{col}' {case['start_time']}~{case['end_time']}: "
                f"{strategy}={fill_val:.4f}"
            )
        logger.info(f"nighttime_zero/solar 处理({strategy}): 处理 {len(other_cases)} 个区间")

    elif strategy == "interpolate":
        for case in other_cases:
            col = case["column"]
            mask = (df.index >= case["start_idx"]) & (df.index <= case["end_idx"])
            df.loc[mask, col] = np.nan
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
        record["cases_handled"] += len(other_cases)
        logger.info(f"nighttime_zero/solar 处理(插值): 处理 {len(other_cases)} 个区间")

    elif strategy == "fill_periodic_mean":
        # 多周期平均: 用相邻周期 (天) 同一时刻位置的均值替换异常区间
        neighbors = kwargs.get("periodic_neighbors", 3)
        total_len = len(df)
        for case in other_cases:
            col = case["column"]
            start_idx = case["start_idx"]
            end_idx = case["end_idx"]
            positions_in_period = (np.arange(start_idx, end_idx + 1)) % PERIOD
            case_period_num = start_idx // PERIOD
            min_period = max(0, case_period_num - neighbors)
            max_period = (total_len - 1) // PERIOD
            max_period = min(max_period, case_period_num + neighbors)
            fill_values = np.full(end_idx - start_idx + 1, np.nan)
            for i, pos in enumerate(positions_in_period):
                neighbor_values = []
                for p in range(min_period, max_period + 1):
                    if p == case_period_num:
                        continue  # 跳过异常区间所在周期
                    idx = p * PERIOD + pos
                    if 0 <= idx < total_len:
                        val = df.loc[idx, col]
                        if not pd.isna(val):
                            neighbor_values.append(val)
                fill_values[i] = np.mean(neighbor_values) if neighbor_values else 0.0
            for i, idx in enumerate(range(start_idx, end_idx + 1)):
                df.loc[idx, col] = fill_values[i]
            record["cases_handled"] += 1
            record["details"].append(
                f"列 '{col}' {case['start_time']}~{case['end_time']}: "
                f"多周期平均替换 (周期={PERIOD}点, 前后各{neighbors}天)"
            )
        logger.info(
            f"nighttime_zero/solar 处理(多周期平均, 前后各{neighbors}天): "
            f"处理 {len(other_cases)} 个区间"
        )

    else:
        raise ValueError(f"未知特殊情况策略: {strategy}")

    return df, record


# =============================================================================
# 数据校验
# =============================================================================
def validate_data(df: pd.DataFrame, time_col: str = "time") -> Dict:
    """数据校验: 检查各属性是否符合昼夜分布规律。"""
    df = df.copy()
    df["_hour"] = df[time_col].dt.hour
    issues: Dict[str, List[str]] = {}

    for col in df.select_dtypes(include=[np.number]).columns:
        col_type = classify_column(col)
        col_issues: List[str] = []

        if col_type == "never_zero":
            zero_count = int((df[col].abs() <= 1e-10).sum())
            if zero_count > 0:
                col_issues.append(f"存在 {zero_count} 个0值 (应为非零)")

        elif col_type == "nighttime_zero":
            day_df = df[(df["_hour"] >= 7) & (df["_hour"] < 19)]
            day_zero_count = int((day_df[col].abs() <= 1e-10).sum())
            day_total = len(day_df)
            if day_total > 0 and day_zero_count == day_total:
                col_issues.append("白天时段全部为0 (可能限电)")

        elif col_type == "solar":
            day_df = df[(df["_hour"] >= 7) & (df["_hour"] < 19)]
            day_zero_count = int((day_df[col].abs() <= 1e-10).sum())
            day_total = len(day_df)
            if day_total > 0 and day_zero_count == day_total:
                col_issues.append("白天时段全部为0 (异常)")

        if col_issues:
            issues[col] = col_issues

    df = df.drop(columns=["_hour"])

    if issues:
        logger.warning(f"数据校验发现 {len(issues)} 列存在潜在问题:")
        for col, col_issues in issues.items():
            for issue in col_issues:
                logger.warning(f"  列 '{col}': {issue}")
    else:
        logger.info("数据校验通过: 各属性昼夜分布符合预期")

    return issues


# =============================================================================
# 输出模块
# =============================================================================
def preview_data(df: pd.DataFrame, label: str = "数据", n: int = 5) -> None:
    """数据预览, 展示前后n行和基本统计。"""
    print(f"\n{'=' * 60}")
    print(f" {label} 预览")
    print(f"{'=' * 60}")
    print(f"记录数: {len(df)}, 列数: {len(df.columns)}")
    print(f"列名: {list(df.columns)}")
    print(f"\n前 {n} 行:")
    print(df.head(n).to_string())
    print(f"\n后 {n} 行:")
    print(df.tail(n).to_string())
    print(f"{'=' * 60}\n")


def save_data(df: pd.DataFrame, output_path: str) -> None:
    """保存数据到CSV。time 列输出为带分隔符 ISO 格式, 确保 pd.to_datetime(无 format) 能正确解析。"""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    df = df.copy()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"已保存结果: {output_path} ({len(df)} 行, {len(df.columns)} 列)")


# =============================================================================
# 主流程
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="两文件合并与处理工具 (自包含, 显式列分类)")
    parser.add_argument("--file-a", "-a", type=str, default=None,
                        help="文件a路径 (time, power, sr)")
    parser.add_argument("--file-b", "-b", type=str, default=None,
                        help="文件b路径 (time + 气象属性)")
    parser.add_argument("--columns-b", type=str, default=None,
                        help="文件b要筛选的外生列, 逗号分隔 (默认4个外生列)")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出 CSV 路径")
    parser.add_argument("--missing-strategy", type=str, default="interpolate",
                        choices=["drop", "fill_mean", "fill_median", "interpolate",
                                 "fill_periodic_mean", "ffill", "bfill"],
                        help="缺失值处理策略 (默认 interpolate)")
    parser.add_argument("--special-strategy", type=str, default="fill_periodic_mean",
                        choices=["drop", "fill_mean", "fill_median", "interpolate",
                                 "fill_periodic_mean", "keep"],
                        help="nighttime_zero/solar 异常处理策略 (默认 fill_periodic_mean)")
    parser.add_argument("--periodic-neighbors", type=int, default=3,
                        help="周期均值使用前后各几天的数据 (默认3)")
    parser.add_argument("--test-cutoff", type=str, default=None,
                        help="测试期起始日期(如 2026-03-01), 内生变量不检测此日期之后")
    parser.add_argument("--log-dir", type=str, default=".", help="日志目录")
    args = parser.parse_args()

    # 初始化日志
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"merge_and_process_{timestamp}.txt")
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)

    logger.info(f"处理开始: {datetime.now()}")

    try:
        # ---- 文件路径 ----
        file_a = args.file_a or input("请输入文件a路径 (time, power, sr): ").strip().strip('"').strip("'")
        file_b = args.file_b or input("请输入文件b路径 (time + 气象属性): ").strip().strip('"').strip("'")
        if not file_a or not file_b:
            raise ValueError("未输入文件路径")

        # ---- 文件b筛选列 ----
        if args.columns_b:
            exog_cols = [c.strip() for c in args.columns_b.split(",")]
        else:
            exog_cols = DEFAULT_EXOG_COLUMNS_B
        logger.info(f"从文件b筛选外生列: {exog_cols}")

        # ---- 读取与筛选 ----
        logger.info("=" * 40 + " 读取文件 " + "=" * 40)
        df_a = read_and_select(file_a, REQUIRED_COLUMNS_A, "文件a")
        df_b = read_and_select(file_b, ["time"] + exog_cols, "文件b", select_exog=exog_cols)

        preview_data(df_a, "文件a (筛选后)")
        preview_data(df_b, "文件b (筛选后)")

        # ---- 时间处理 ----
        logger.info("=" * 40 + " 时间处理 " + "=" * 40)
        df_a["time"] = smart_parse_time(df_a["time"])
        df_b["time"] = smart_parse_time(df_b["time"])
        for name, d in [("文件a", df_a), ("文件b", df_b)]:
            na = int(d["time"].isna().sum())
            if na > 0:
                logger.warning(f"{name}时间解析失败 {na} 行, 已剔除")
                d.dropna(subset=["time"], inplace=True)
        logger.info(f"文件a时间范围: {df_a['time'].min()} ~ {df_a['time'].max()}")
        logger.info(f"文件b时间范围: {df_b['time'].min()} ~ {df_b['time'].max()}")

        # ---- 合并 (以a为基准) ----
        logger.info("=" * 40 + " 数据合并 (以文件a为基准) " + "=" * 40)
        merged = merge_on_a(df_a, df_b, exog_cols)

        # 校验最终列顺序: time, power, sr, [外生列...]
        expected = ["time", "power", "sr"] + exog_cols
        if list(merged.columns) != expected:
            merged = merged[expected]
            logger.info(f"调整列顺序为模型输入约定: {list(merged.columns)}")

        preview_data(merged, "合并后 (缺失值处理前)")

        # ---- 缺失值处理 ----
        logger.info("=" * 40 + " 缺失值处理 " + "=" * 40)
        detect_missing_values(merged)
        merged, missing_record = handle_missing_values(
            merged, strategy=args.missing_strategy, periodic_neighbors=args.periodic_neighbors
        )

        # ---- 异常0值检测与处理 ----
        logger.info("=" * 40 + " 异常0值检测 " + "=" * 40)
        # 打印分类
        logger.info("属性昼夜分类:")
        for col in expected:
            if col == "time":
                continue
            col_type = classify_column(col)
            desc = {
                "nighttime_zero": "内生, 夜间为0正常 (全天0=限电, 检测排除测试期)",
                "solar": "外生, 夜间为0正常 (白天全0=异常, 检测排除测试期)",
                "never_zero": "外生, 全天不应为0 (任何0=异常, 全时段检测)",
                "other": "其他 (不检测)",
            }
            logger.info(f"  {col}: {col_type} -- {desc.get(col_type, '')}")

        target_cols = [c for c in expected if c != "time"]
        special_cases = detect_special_cases(
            merged, target_columns=target_cols, test_cutoff=args.test_cutoff
        )
        if special_cases:
            print(f"\n检测到 {len(special_cases)} 个特殊情况:")
            for i, case in enumerate(special_cases, 1):
                print(f"  {i}. 列 '{case['column']}' [{case['col_type']}]: "
                      f"{case['date']}, {case['start_time']} ~ {case['end_time']}, "
                      f"{case['row_count']} 行")
            merged, special_record = handle_special_cases(
                merged, special_cases, strategy=args.special_strategy,
                periodic_neighbors=args.periodic_neighbors,
            )
        else:
            logger.info("无特殊情况")

        # ---- 校验 ----
        logger.info("=" * 40 + " 数据校验 " + "=" * 40)
        remaining_nan = int(merged.isna().sum().sum())
        if remaining_nan > 0:
            logger.warning(f"处理后仍有 {remaining_nan} 个缺失值")
        else:
            logger.info("校验通过: 无残留缺失值")
        validate_data(merged, time_col="time")

        # ---- 输出 ----
        logger.info("=" * 40 + " 输出结果 " + "=" * 40)
        preview_data(merged, "最终结果")
        output_path = args.output
        if not output_path:
            output_path = input("请输入输出文件路径 (回车使用默认): ").strip()
            if not output_path:
                output_path = f"merged_processed_{timestamp}.csv"
        save_data(merged, output_path)

        logger.info(f"日志文件: {log_path}")
        logger.info(f"处理完成: {datetime.now()}")

    except Exception as e:
        logger.error(f"错误: {e}")
        raise


if __name__ == "__main__":
    main()
