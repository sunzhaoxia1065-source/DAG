#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
玖天数据集统一处理模块

功能:
    接收原始 EC 气象预报数据 + 功率实测数据, 经过 6 步流水线处理,
    输出符合 DAG 模型训练要求的格式化 CSV 文件。

处理流水线 (6 步):
    1. 按经纬度 (point_id) 拆分原始 EC 数据
    2. 清理无用列 + 时间区间过滤
    3. 计算水平梯度 / 垂直风切变 / 幂指数 (仅风电)
    4. 合并主文件 + 网格点风速 + 梯度 (风电) / 简单合并 (光伏)
    5. 合并功率数据 + 缺失值处理
    6. 特征选择 (风电/光伏独立规则)

支持两种发电类型:
    - wind (风力发电): 内生变量 = power, ws; 侧重风速/梯度/风切变
    - solar (光伏发电): 内生变量 = power, sr; 侧重辐射/温度/云量

用法:
    # 风电数据处理 (完整 6 步)
    python tools/data_processor.py --plant-type wind \
        --ec-data 玖天数据集处理/haojiaying.csv \
        --power-data 玖天数据集处理/郝家营风电场二期power-ws.xlsx \
        --work-dir 玖天数据集处理/processed_wind \
        --time-start 202407030000 --time-end 202607312345 \
        --center-grid 41_400_114_900

    # 光伏数据处理 (跳过梯度计算)
    python tools/data_processor.py --plant-type solar \
        --ec-data 玖天数据集处理/ninghe.csv \
        --power-data 玖天数据集处理/宁河光伏电站power-sr.csv \
        --work-dir 玖天数据集处理/processed_solar \
        --time-start 202407030000 --time-end 202607312345 \
        --center-grid 39_200_117_400

    # 仅执行指定步骤 (调试用)
    python tools/data_processor.py --plant-type wind --ec-data ... --steps 1,2,5
"""

import argparse
import logging
import math
import os
import sys
import time
from datetime import datetime  
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# 配置常量
# =============================================================================

TIME_COL = "time"
POINTS_PER_DAY = 96  # 15 分钟分辨率, 24*4=96
PERIOD = 96  # 周期长度: 1天 = 96 个 15min 点 (光伏异常0值周期插补用)

# =============================================================================
# 光伏列分类系统 (异常0值检测用, 与 merge_csv_ninghe.py 对齐)
# =============================================================================
# - nighttime_zero: 全天为0才算异常 (限电), 夜间为0正常 (power/sr)
# - solar: 白天时段 (07:00~19:00) 全为0才算异常 (太阳辐射)
# - never_zero: 任何时刻为0即为异常 (热辐射/湿度/露点), 全时段检测
# - other: 不检测
COLUMN_CLASSIFICATION: Dict[str, str] = {
    "power": "nighttime_zero",
    "sr": "nighttime_zero",
    "total_sky_direct_solar_radiation_at_surface_surface": "solar",
    "surface_thermal_radiation_downwards_surface": "never_zero",
    "relative_humidity_isobaric_950": "never_zero",
    "relatively_humidity_isobaric_950": "never_zero",  # 兼容变体
    "dewpoint_temperature_surface_2_metre": "never_zero",
    "dewpoint_temperature_surface_2metre": "never_zero",  # 兼容变体
}
_SOLAR_KEYWORDS = ["direct_solar", "日照"]
_NEVER_ZERO_KEYWORDS = ["thermal_radiation", "humidity", "热辐射", "湿度"]


def classify_column(col: str) -> str:
    """光伏列分类: 显式字典优先, 关键词回退, 都不命中返回 'other'."""
    if col in COLUMN_CLASSIFICATION:
        return COLUMN_CLASSIFICATION[col]
    col_lower = col.lower()
    if any(k in col_lower for k in _NEVER_ZERO_KEYWORDS):
        return "never_zero"
    if any(k in col_lower for k in _SOLAR_KEYWORDS):
        return "solar"
    return "other"


# 5x5 网格定义 (郝家营二期风电场)
WIND_LAT_RANGE = [41.2, 41.3, 41.4, 41.5, 41.6]
WIND_LON_RANGE = [114.7, 114.8, 114.9, 115.0, 115.1]

# 风速分量列名
U_LOW_COL = "u_wind_component_surface_10_metre"
V_LOW_COL = "v_wind_component_surface_10_metre"
U_HIGH_COL = "u_wind_component_surface_100_metre"
V_HIGH_COL = "v_wind_component_surface_100_metre"

# u/v 配对的高度层 (用于梯度计算)
WIND_LEVELS = [
    ("hybrid_130", "u_component_of_wind_hybrid_130", "v_component_of_wind_hybrid_130"),
    ("hybrid_131", "u_component_of_wind_hybrid_131", "v_component_of_wind_hybrid_131"),
    ("hybrid_132", "u_component_of_wind_hybrid_132", "v_component_of_wind_hybrid_132"),
    ("hybrid_133", "u_component_of_wind_hybrid_133", "v_component_of_wind_hybrid_133"),
    ("hybrid_134", "u_component_of_wind_hybrid_134", "v_component_of_wind_hybrid_134"),
    ("hybrid_135", "u_component_of_wind_hybrid_135", "v_component_of_wind_hybrid_135"),
    ("hybrid_136", "u_component_of_wind_hybrid_136", "v_component_of_wind_hybrid_136"),
    ("hybrid_137", "u_component_of_wind_hybrid_137", "v_component_of_wind_hybrid_137"),
    ("10m", "u_wind_component_surface_10_metre", "v_wind_component_surface_10_metre"),
    ("100m", "u_wind_component_surface_100_metre", "v_wind_component_surface_100_metre"),
    ("isobaric_500", "u_component_of_wind_isobaric_500", "v_component_of_wind_isobaric_500"),
    ("isobaric_600", "u_component_of_wind_isobaric_600", "v_component_of_wind_isobaric_600"),
    ("isobaric_700", "u_component_of_wind_isobaric_700", "v_component_of_wind_isobaric_700"),
    ("isobaric_800", "u_component_of_wind_isobaric_800", "v_component_of_wind_isobaric_800"),
    ("isobaric_850", "u_component_of_wind_isobaric_850", "v_component_of_wind_isobaric_850"),
    ("isobaric_900", "u_component_of_wind_isobaric_900", "v_component_of_wind_isobaric_900"),
    ("isobaric_925", "u_component_of_wind_isobaric_925", "v_component_of_wind_isobaric_925"),
    ("isobaric_950", "u_component_of_wind_isobaric_950", "v_component_of_wind_isobaric_950"),
    ("isobaric_1000", "u_component_of_wind_isobaric_1000", "v_component_of_wind_isobaric_1000"),
]

EARTH_RADIUS_KM = 6371.0

# =============================================================================
# 默认特征集 (Step 6 默认模式使用)
# =============================================================================

# 光伏默认特征 (7 列: time/power/sr + 4 个辐射/温度/湿度特征)
SOLAR_DEFAULT_FEATURES = [
    "time",
    "power",
    "sr",
    "relative_humidity_isobaric_950",
    "total_sky_direct_solar_radiation_at_surface_surface",
    "surface_thermal_radiation_downwards_surface",
    "dewpoint_temperature_surface_2_metre",
]

# 风电默认特征由 WindDataProcessor.step6_select_features() 内联生成 (~130 列)
# 组成: 地表风分量 + 温湿压辐射 + 垂直风廓线 + 派生特征 + 梯度(5点×19层) + 风速


# =============================================================================
# 工具函数
# =============================================================================

def setup_logger(log_file: str = None) -> logging.Logger:
    """配置日志, 同时输出到控制台和文件"""
    lg = logging.getLogger("data_processor")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    lg.addHandler(sh)
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        try:
            fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
            fh.setFormatter(fmt)
            lg.addHandler(fh)
        except Exception as e:
            lg.warning(f"无法创建日志文件 {log_file}: {e}")
    return lg


def point_id_to_filename(point_id: str) -> str:
    """point_id -> 文件名 (41.200_115.100 -> 41_200_115_100.csv)"""
    return point_id.replace(".", "_").replace("/", "_").replace("\\", "_").replace(" ", "") + ".csv"


def parse_filename_to_coord(filename: str, prefix: str = "") -> Tuple[float, float]:
    """文件名 -> (lat, lon) (41_200_114_700.csv -> 41.2, 114.7)"""
    name = os.path.basename(filename).lower()
    if name.endswith(".csv"):
        name = name[:-4]
    if prefix and name.startswith(prefix.lower()):
        name = name[len(prefix):]
    parts = name.split("_")
    if len(parts) != 4:
        raise ValueError(f"文件名格式不符: {filename}")
    return float(f"{parts[0]}.{parts[1]}"), float(f"{parts[2]}.{parts[3]}")


def format_coord_str(lat: float, lon: float) -> str:
    """(lat, lon) -> 坐标字符串 (41.4, 114.9 -> 41_400_114_900)"""
    return f"{int(lat):02d}_{int(round((lat-int(lat))*1000)):03d}_{int(lon):03d}_{int(round((lon-int(lon))*1000)):03d}"


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """球面距离 (km)"""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlam = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def parse_time(time_val) -> pd.Timestamp:
    """解析多种时间格式"""
    if isinstance(time_val, (pd.Timestamp, datetime)):
        return pd.Timestamp(time_val)
    s = str(time_val).strip()
    if s.isdigit() and len(s) == 12:
        try:
            return pd.to_datetime(s, format="%Y%m%d%H%M")
        except ValueError:
            pass
    try:
        return pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S")
    except ValueError:
        pass
    try:
        return pd.to_datetime(s)
    except (ValueError, TypeError):
        return pd.NaT


def compute_wind_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """合成风速 V = sqrt(u^2 + v^2)"""
    return np.sqrt(u**2 + v**2)


# =============================================================================
# 基类: 通用数据处理
# =============================================================================

class DataProcessor:
    """数据处理基类, 实现 6 步流水线中通用的步骤"""

    def __init__(
        self,
        ec_data_path: str,
        power_data_path: str,
        work_dir: str,
        time_start: str = "202407030000",
        time_end: str = "202607312345",
        center_grid: str = None,
        fill_method: str = "fill_periodic_mean",
        periodic_neighbors: int = 3,
        features: List[str] = None,
        features_file: str = None,
        test_cutoff: str = None,
        log_file: str = None,
    ):
        self.ec_data_path = ec_data_path
        self.power_data_path = power_data_path
        self.work_dir = work_dir
        self.center_grid = center_grid
        self.fill_method = fill_method
        self.periodic_neighbors = periodic_neighbors
        self.test_cutoff = test_cutoff  # 测试期起始日期 (光伏内生变量不检测此日期之后)

        # 时间过滤: 统一转为 YYYYMMDDHHmm 12 位数字 (用于字符串比较)
        self.time_start = self._normalize_time_filter(time_start)
        self.time_end = self._normalize_time_filter(time_end)

        # 自定义特征列表 (Step 6 自定义模式)
        self.custom_features = features
        if features_file:
            self.custom_features = self._load_features_from_file(features_file)

        # 工作子目录
        self.split_dir = os.path.join(work_dir, "by_point")
        self.gradient_dir = os.path.join(work_dir, "gradient")
        self.merged_dir = os.path.join(work_dir, "merged")
        self.output_dir = os.path.join(work_dir, "final")

        # 日志
        self.logger = setup_logger(log_file or os.path.join(work_dir, "data_processor.log"))

        # 网格定义 (子类覆盖)
        self.lat_range = WIND_LAT_RANGE
        self.lon_range = WIND_LON_RANGE
        self.grid_size = 5

    def _normalize_time_filter(self, time_str: str) -> str:
        """将时间字符串统一转为 YYYYMMDDHHmm 12 位数字格式"""
        if not time_str:
            return ""
        s = str(time_str).strip()
        # 已经是 12 位纯数字
        if s.isdigit() and len(s) == 12:
            return s
        # 尝试 YYYY-MM-DD HH:MM:SS 格式
        try:
            ts = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S")
            return ts.strftime("%Y%m%d%H%M")
        except ValueError:
            pass
        # 尝试 YYYY-MM-DD HH:MM 格式
        try:
            ts = pd.to_datetime(s, format="%Y-%m-%d %H:%M")
            return ts.strftime("%Y%m%d%H%M")
        except ValueError:
            pass
        # 尝试 YYYY-MM-DD 格式 (补全为 00:00)
        try:
            ts = pd.to_datetime(s, format="%Y-%m-%d")
            return ts.strftime("%Y%m%d%H%M")
        except ValueError:
            pass
        # 兜底: 让 pandas 自动推断
        try:
            ts = pd.to_datetime(s)
            if pd.notna(ts):
                return ts.strftime("%Y%m%d%H%M")
        except (ValueError, TypeError):
            pass
        # 无法解析, 原样返回 (后续会跳过时间过滤)
        self.logger.warning(f"无法解析时间 '{time_str}', 时间过滤将被禁用")
        return ""

    def _load_features_from_file(self, filepath: str) -> List[str]:
        """从文本文件加载特征列表 (逗号分隔或每行一个)"""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"特征文件不存在: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
        # 支持逗号分隔或换行分隔
        features = []
        for line in content.split("\n"):
            for col in line.split(","):
                col = col.strip()
                if col:
                    features.append(col)
        self.logger.info(f"从文件加载 {len(features)} 个特征: {filepath}")
        return features

    # ------------------------------------------------------------------
    # Step 1: 按经纬度拆分
    # ------------------------------------------------------------------
    def step1_split_by_point_id(self) -> str:
        """将原始 EC 数据按 point_id 拆分为多个 CSV 文件"""
        import csv
        from collections import defaultdict

        self.logger.info("=" * 60)
        self.logger.info("Step 1: 按经纬度拆分 EC 数据")
        self.logger.info("=" * 60)

        os.makedirs(self.split_dir, exist_ok=True)
        output_base = os.path.join(self.split_dir, os.path.splitext(os.path.basename(self.ec_data_path))[0])
        os.makedirs(output_base, exist_ok=True)

        file_handles = {}
        point_id_counts = defaultdict(int)
        total_rows = 0
        header = None
        point_id_col_idx = None

        start_time = time.time()
        try:
            with open(self.ec_data_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
                if "point_id" not in header:
                    raise ValueError(f"EC 数据缺少 point_id 列, 实际前5列: {header[:5]}")
                point_id_col_idx = header.index("point_id")
                self.logger.info(f"point_id 列索引: {point_id_col_idx}, 总列数: {len(header)}")

                for row in reader:
                    total_rows += 1
                    if total_rows % 100000 == 0:
                        self.logger.info(f"  已处理 {total_rows} 行, point_id 数={len(file_handles)}")

                    if len(row) <= point_id_col_idx:
                        continue
                    point_id = row[point_id_col_idx].strip()
                    if not point_id:
                        continue

                    if point_id not in file_handles:
                        fname = point_id_to_filename(point_id)
                        fpath = os.path.join(output_base, fname)
                        fh = open(fpath, "w", encoding="utf-8", newline="")
                        writer = csv.writer(fh)
                        writer.writerow(header)
                        file_handles[point_id] = (fh, writer)
                        self.logger.info(f"  创建: {fname}")

                    file_handles[point_id][1].writerow(row)
                    point_id_counts[point_id] += 1
        finally:
            for fh, _ in file_handles.values():
                fh.close()

        elapsed = time.time() - start_time
        self.logger.info(f"拆分完成: 总行={total_rows}, point_id 数={len(point_id_counts)}, 耗时={elapsed:.1f}s")
        self.logger.info(f"输出目录: {output_base}")
        self._split_dir = output_base
        return output_base

    # ------------------------------------------------------------------
    # Step 2: 清理列 + 时间过滤
    # ------------------------------------------------------------------
    def step2_clean_and_filter(self, split_dir: str = None) -> str:
        """删除 batch/point_id 列, 按时间区间过滤"""
        import csv
        import tempfile

        split_dir = split_dir or getattr(self, "_split_dir", os.path.join(self.split_dir, os.path.splitext(os.path.basename(self.ec_data_path))[0]))
        self.logger.info("=" * 60)
        self.logger.info("Step 2: 清理列 + 时间过滤")
        if self.time_start and self.time_end:
            self.logger.info(f"  时间范围: {self.time_start} ~ {self.time_end} (YYYYMMDDHHmm)")
        else:
            self.logger.info("  时间过滤: 已禁用 (未指定或解析失败)")
        self.logger.info(f"  输入目录: {split_dir}")
        self.logger.info("=" * 60)

        csv_files = sorted(f for f in os.listdir(split_dir) if f.lower().endswith(".csv"))
        self.logger.info(f"待处理文件数: {len(csv_files)}")

        columns_to_drop = {"batch", "point_id"}
        total_original = 0
        total_kept = 0

        for i, fname in enumerate(csv_files, 1):
            fpath = os.path.join(split_dir, fname)
            self.logger.info(f"[{i}/{len(csv_files)}] {fname}")

            with open(fpath, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
                time_idx = header.index(TIME_COL)
                drop_indices = {header.index(c) for c in columns_to_drop if c in header}
                new_header = [c for i, c in enumerate(header) if i not in drop_indices]

                tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=split_dir)
                os.close(tmp_fd)
                with open(tmp_path, "w", encoding="utf-8", newline="") as out_f:
                    writer = csv.writer(out_f)
                    writer.writerow(new_header)
                    file_total = 0
                    file_kept = 0
                    for row in reader:
                        file_total += 1
                        time_val = row[time_idx].strip()
                        # 时间过滤 (如果已禁用则跳过过滤)
                        if self.time_start and self.time_end:
                            if len(time_val) != 12 or not time_val.isdigit():
                                continue
                            if not (self.time_start <= time_val <= self.time_end):
                                continue
                        new_row = [v for i, v in enumerate(row) if i not in drop_indices]
                        writer.writerow(new_row)
                        file_kept += 1

            os.replace(tmp_path, fpath)
            total_original += file_total
            total_kept += file_kept
            self.logger.info(f"  原始 {file_total} -> 保留 {file_kept}")

        self.logger.info(f"汇总: 原始 {total_original} -> 保留 {total_kept}")
        return split_dir

    # ------------------------------------------------------------------
    # Step 3: 计算梯度 (基类默认空实现, 风电子类覆盖)
    # ------------------------------------------------------------------
    def step3_compute_gradient(self, split_dir: str = None) -> Optional[str]:
        """基类默认跳过 (光伏不需要)"""
        self.logger.info("=" * 60)
        self.logger.info("Step 3: 跳过梯度计算 (当前类型不需要)")
        self.logger.info("=" * 60)
        return None

    # ------------------------------------------------------------------
    # Step 4: 合并主文件 + 网格点 + 梯度
    # ------------------------------------------------------------------
    def step4_merge_master(self, split_dir: str = None, gradient_dir: str = None) -> str:
        """合并主文件 + 25 点风速 + 9 点梯度 (风电) / 仅合并中心点 (光伏)"""
        split_dir = split_dir or getattr(self, "_split_dir", "")
        os.makedirs(self.merged_dir, exist_ok=True)

        self.logger.info("=" * 60)
        self.logger.info("Step 4: 合并主文件 + 网格点风速 + 梯度")
        self.logger.info("=" * 60)

        # 加载主文件
        master_name = f"{self.center_grid}.csv"
        master_path = os.path.join(split_dir, master_name)
        if not os.path.isfile(master_path):
            raise FileNotFoundError(f"中心点主文件不存在: {master_path}, 请检查 --center-grid 参数")
        self.logger.info(f"主文件: {master_path}")
        master_df = pd.read_csv(master_path, encoding="utf-8")
        master_df[TIME_COL] = master_df[TIME_COL].astype(str).str.strip()
        master_df = master_df.drop_duplicates(subset=TIME_COL, keep="first").sort_values(TIME_COL).reset_index(drop=True)
        merged_df = master_df
        self.logger.info(f"主文件: 行={len(master_df)}, 列={len(master_df.columns)}")

        # 合并 25 个网格点的 10m/100m 风速
        point_files = sorted(f for f in os.listdir(split_dir) if f.lower().endswith(".csv"))
        self.logger.info(f"阶段1: 合并 {len(point_files)} 个网格点风速")
        for i, fname in enumerate(point_files, 1):
            fpath = os.path.join(split_dir, fname)
            coord_tag = os.path.splitext(fname)[0]
            df = pd.read_csv(fpath, encoding="utf-8")
            if TIME_COL not in df.columns:
                continue
            df[TIME_COL] = df[TIME_COL].astype(str).str.strip()
            df = df.drop_duplicates(subset=TIME_COL, keep="first").sort_values(TIME_COL).reset_index(drop=True)

            required = [U_LOW_COL, V_LOW_COL, U_HIGH_COL, V_HIGH_COL]
            if all(c in df.columns for c in required):
                wind_10 = compute_wind_speed(df[U_LOW_COL].values, df[V_LOW_COL].values)
                wind_100 = compute_wind_speed(df[U_HIGH_COL].values, df[V_HIGH_COL].values)
                wind_df = pd.DataFrame({
                    TIME_COL: df[TIME_COL],
                    f"wind_speed_10m__{coord_tag}": wind_10,
                    f"wind_speed_100m__{coord_tag}": wind_100,
                })
                merged_df = merged_df.merge(wind_df, on=TIME_COL, how="left")
            if i % 5 == 0:
                self.logger.info(f"  [{i}/{len(point_files)}] 已合并, 列数={len(merged_df.columns)}")

        self.logger.info(f"风速合并后: 行={len(merged_df)}, 列={len(merged_df.columns)}")

        # 合并梯度文件 (如果存在)
        if gradient_dir and os.path.isdir(gradient_dir):
            grad_files = sorted(f for f in os.listdir(gradient_dir) if f.lower().startswith("gradient_") and f.lower().endswith(".csv"))
            self.logger.info(f"阶段2: 合并 {len(grad_files)} 个梯度文件")
            for i, fname in enumerate(grad_files, 1):
                fpath = os.path.join(gradient_dir, fname)
                coord_tag = os.path.splitext(fname)[0].replace("gradient_", "")
                df = pd.read_csv(fpath, encoding="utf-8")
                if TIME_COL not in df.columns:
                    continue
                df[TIME_COL] = df[TIME_COL].astype(str).str.strip()
                df = df.drop_duplicates(subset=TIME_COL, keep="first").sort_values(TIME_COL).reset_index(drop=True)
                rename = {c: f"{c}__{coord_tag}" for c in df.columns if c != TIME_COL}
                df = df.rename(columns=rename)
                merged_df = merged_df.merge(df, on=TIME_COL, how="left")
                self.logger.info(f"  [{i}/{len(grad_files)}] {fname}, 列数={len(merged_df.columns)}")

        # 保存
        out_path = os.path.join(self.merged_dir, f"merged_{self.center_grid}.csv")
        merged_df.to_csv(out_path, index=False, encoding="utf-8")
        self.logger.info(f"合并结果: {out_path}, 行={len(merged_df)}, 列={len(merged_df.columns)}")
        self._merged_path = out_path
        return out_path

    # ------------------------------------------------------------------
    # Step 5: 合并功率数据 + 缺失值处理
    # ------------------------------------------------------------------
    def step5_merge_power_and_fillna(self, merged_path: str = None) -> str:
        """合并功率数据并处理缺失值"""
        merged_path = merged_path or getattr(self, "_merged_path", "")
        os.makedirs(self.output_dir, exist_ok=True)

        self.logger.info("=" * 60)
        self.logger.info("Step 5: 合并功率数据 + 缺失值处理")
        self.logger.info("=" * 60)

        # 加载合并文件
        self.logger.info(f"读取合并文件: {merged_path}")
        merge_df = pd.read_csv(merged_path, encoding="utf-8", dtype={TIME_COL: "string"})
        merge_df["_timestamp"] = merge_df[TIME_COL].apply(parse_time)
        merge_df = merge_df.drop_duplicates(subset="_timestamp", keep="first").sort_values("_timestamp").reset_index(drop=True)
        self.logger.info(f"合并文件: 行={len(merge_df)}, 列={len(merge_df.columns)-1}")

        # 加载功率数据
        self.logger.info(f"读取功率数据: {self.power_data_path}")
        ext = os.path.splitext(self.power_data_path)[1].lower()
        if ext in [".xlsx", ".xls"]:
            power_df = pd.read_excel(self.power_data_path)
        else:
            power_df = pd.read_csv(self.power_data_path, encoding="utf-8", dtype={TIME_COL: "string"})
        power_df[TIME_COL] = power_df[TIME_COL].astype(str).str.strip()
        power_df["_timestamp"] = power_df[TIME_COL].apply(parse_time)
        power_df = power_df.drop_duplicates(subset="_timestamp", keep="first").sort_values("_timestamp").reset_index(drop=True)
        self.logger.info(f"功率数据: 行={len(power_df)}, 列={list(power_df.columns)}")

        # 合并 (left-join on _timestamp)
        merge_df_join = merge_df.drop(columns=[TIME_COL], errors="ignore")
        merged = power_df.merge(merge_df_join, on="_timestamp", how="left")
        self.logger.info(f"合并后: 行={len(merged)}, 列={len(merged.columns)}")

        # 异常0值检测与处理 (钩子: 风电默认跳过, 光伏重写为完整逻辑)
        data_cols = [c for c in merged.columns if c not in (TIME_COL, "_timestamp")]
        merged = self._handle_special_zero_values(merged, data_cols)

        # 常规缺失值检测
        data_cols = [c for c in merged.columns if c not in (TIME_COL, "_timestamp")]
        before_miss = sum(merged[c].isna().sum() for c in data_cols)
        self.logger.info(f"处理前缺失值总数: {before_miss}")

        # 缺失值处理
        if self.fill_method == "fill_periodic_mean":
            merged = self._fill_periodic_all(merged, data_cols)
        elif self.fill_method == "interpolate":
            merged = self._fill_interpolate_all(merged, data_cols)
        elif self.fill_method == "mean":
            merged = self._fill_mean_all(merged, data_cols)
        else:
            raise ValueError(f"不支持的填充方法: {self.fill_method} (可选: fill_periodic_mean, interpolate, mean)")

        # 兜底: 整列空的用 0 填充
        for col in data_cols:
            n_miss = pd.to_numeric(merged[col], errors="coerce").isna().sum()
            if n_miss > 0:
                col_mean = pd.to_numeric(merged[col], errors="coerce").mean()
                if pd.notna(col_mean):
                    merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(col_mean)
                else:
                    merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

        after_miss = sum(merged[c].isna().sum() for c in data_cols)
        self.logger.info(f"处理后缺失值总数: {after_miss} (减少 {before_miss - after_miss})")

        # 格式化 time 列
        merged[TIME_COL] = merged["_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        merged = merged.drop(columns=["_timestamp"])

        # 保存
        out_path = os.path.join(self.output_dir, f"final_{self.center_grid}.csv")
        merged.to_csv(out_path, index=False, encoding="utf-8")
        self.logger.info(f"最终数据: {out_path}, 行={len(merged)}, 列={len(merged.columns)}")
        self._final_path = out_path
        return out_path

    def _fill_periodic_all(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """周期插补: 按天为周期, 用前后 N 天同一时刻均值填充"""
        ts = df["_timestamp"]
        valid_mask = ts.notna()
        dates = ts.dt.date
        slots = ts.dt.hour * 4 + ts.dt.minute // 15
        all_dates = sorted(set(dates.dropna()))

        date_slot_to_idx = {}
        for idx in df.index[valid_mask]:
            key = (dates.iloc[idx], slots.iloc[idx])
            if key not in date_slot_to_idx:
                date_slot_to_idx[key] = idx

        for col in cols:
            s = pd.to_numeric(df[col], errors="coerce")
            missing_mask = s.isna() & valid_mask
            n_miss = missing_mask.sum()
            if n_miss == 0:
                continue
            filled = 0
            for idx in s.index[missing_mask]:
                cur_date = dates[idx]
                cur_slot = slots[idx]
                if pd.isna(cur_date):
                    continue
                try:
                    date_pos = all_dates.index(cur_date)
                except ValueError:
                    continue
                start_pos = max(0, date_pos - self.periodic_neighbors)
                end_pos = min(len(all_dates), date_pos + self.periodic_neighbors + 1)
                vals = []
                for nd in all_dates[start_pos:end_pos]:
                    if nd == cur_date:
                        continue
                    key = (nd, cur_slot)
                    if key in date_slot_to_idx:
                        v = s.iloc[date_slot_to_idx[key]]
                        if pd.notna(v):
                            vals.append(v)
                if vals:
                    s.iloc[idx] = np.mean(vals)
                    filled += 1
            df[col] = s.astype("float32")
            self.logger.info(f"  周期插补 {col}: {filled}/{n_miss}")
        return df

    def _fill_interpolate_all(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """线性插值"""
        for col in cols:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.isna().any():
                before = s.isna().sum()
                s = s.interpolate(method="linear", limit_direction="both")
                after = s.isna().sum()
                if after < before:
                    self.logger.info(f"  线性插值 {col}: {before - after}")
                df[col] = s
        return df

    def _fill_mean_all(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """均值填充: 用列均值填充缺失值"""
        for col in cols:
            s = pd.to_numeric(df[col], errors="coerce")
            n_miss = s.isna().sum()
            if n_miss > 0:
                col_mean = s.mean()
                if pd.notna(col_mean):
                    df[col] = s.fillna(col_mean).astype("float32")
                    self.logger.info(f"  均值填充 {col}: {n_miss} 个 (均值={col_mean:.4f})")
                else:
                    df[col] = s.fillna(0).astype("float32")
                    self.logger.warning(f"  均值填充 {col}: 列全空, 用 0 填充 {n_miss} 个")
        return df

    # ------------------------------------------------------------------
    # 异常0值处理钩子 (风电默认跳过, 光伏重写)
    # ------------------------------------------------------------------
    def _handle_special_zero_values(self, df: pd.DataFrame, data_cols: List[str]) -> pd.DataFrame:
        """基类默认: 跳过异常0值检测 (风电只需常规NaN填充)"""
        self.logger.info("异常0值检测: 跳过 (风电场景只处理NaN缺失值)")
        return df

    # ------------------------------------------------------------------
    # Step 6: 特征选择 (子类覆盖)
    # ------------------------------------------------------------------
    def step6_select_features(self, final_path: str = None) -> str:
        """基类默认保留全部特征, 子类覆盖"""
        final_path = final_path or getattr(self, "_final_path", "")
        self.logger.info("=" * 60)
        self.logger.info("Step 6: 特征选择 (基类默认跳过, 保留全部)")
        self.logger.info("=" * 60)
        return final_path

    # ------------------------------------------------------------------
    # 运行完整流水线
    # ------------------------------------------------------------------
    def run(self, steps: List[int] = None) -> str:
        """运行 6 步流水线, steps 指定执行哪些步骤 (默认全部)"""
        steps = steps or [1, 2, 3, 4, 5, 6]
        os.makedirs(self.work_dir, exist_ok=True)

        self.logger.info("#" * 60)
        self.logger.info(f"数据处理流水线启动 | 类型={self.__class__.__name__}")
        self.logger.info(f"EC 数据: {self.ec_data_path}")
        self.logger.info(f"功率数据: {self.power_data_path}")
        self.logger.info(f"工作目录: {self.work_dir}")
        self.logger.info(f"执行步骤: {steps}")
        self.logger.info("#" * 60)

        result_path = None
        if 1 in steps:
            self.step1_split_by_point_id()
        if 2 in steps:
            self.step2_clean_and_filter()
        if 3 in steps:
            grad_dir = self.step3_compute_gradient()
            if grad_dir:
                self._gradient_dir = grad_dir
            else:
                self._gradient_dir = None
        if 4 in steps:
            grad_dir = getattr(self, "_gradient_dir", None)
            self.step4_merge_master(gradient_dir=grad_dir)
        if 5 in steps:
            self.step5_merge_power_and_fillna()
        if 6 in steps:
            result_path = self.step6_select_features()

        self.logger.info("#" * 60)
        self.logger.info(f"流水线完成! 输出文件: {result_path or '未执行特征选择'}")
        self.logger.info("#" * 60)
        return result_path


# =============================================================================
# 风电数据处理
# =============================================================================

class WindDataProcessor(DataProcessor):
    """风力发电数据处理 (含梯度/风切变计算)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lat_range = WIND_LAT_RANGE
        self.lon_range = WIND_LON_RANGE

    def step3_compute_gradient(self, split_dir: str = None) -> str:
        """计算 9 个内部点的水平梯度 + 垂直风切变 + 幂指数"""
        split_dir = split_dir or getattr(self, "_split_dir", "")
        os.makedirs(self.gradient_dir, exist_ok=True)

        self.logger.info("=" * 60)
        self.logger.info("Step 3 (风电): 计算水平梯度 + 垂直风切变 + 幂指数")
        self.logger.info(f"  网格: {self.grid_size}x{self.grid_size}, lat {self.lat_range}, lon {self.lon_range}")
        self.logger.info("=" * 60)

        # 加载 25 个网格点
        grid = {}
        coords = {}
        csv_files = sorted(f for f in os.listdir(split_dir) if f.lower().endswith(".csv"))
        for fname in csv_files:
            fpath = os.path.join(split_dir, fname)
            try:
                lat, lon = parse_filename_to_coord(fname)
                x = self.lon_range.index(round(lon, 3))
                y = self.lat_range.index(round(lat, 3))
                coords[(x, y)] = (lat, lon)
                df = pd.read_csv(fpath, encoding="utf-8")
                df[TIME_COL] = df[TIME_COL].astype(str).str.strip()
                df = df.drop_duplicates(subset=TIME_COL).sort_values(TIME_COL).set_index(TIME_COL)
                grid[(x, y)] = df
            except Exception as e:
                self.logger.warning(f"加载 {fname} 失败: {e}")

        self.logger.info(f"加载 {len(grid)} 个网格点")

        # 内部点 (1<=x<=3, 1<=y<=3)
        interior = [(x, y) for x in range(1, self.grid_size-1) for y in range(1, self.grid_size-1)]
        self.logger.info(f"内部点 {len(interior)} 个: {interior}")

        success = 0
        for (x, y) in interior:
            lat_c, lon_c = coords[(x, y)]
            self.logger.info(f"处理 ({x},{y}): lat={lat_c}, lon={lon_c}")
            try:
                # 网格步长 (km)
                dx_km = haversine(lat_c, 0, lat_c, self.lon_range[1] - self.lon_range[0])
                dy_km = haversine(0, 0, self.lat_range[1] - self.lat_range[0], 0)

                # 水平梯度
                neighbors = {"east": (x+1, y), "west": (x-1, y), "north": (x, y+1), "south": (x, y-1)}
                df_c = grid[(x, y)]
                common_idx = df_c.index
                for key in neighbors.values():
                    if key in grid:
                        common_idx = common_idx.intersection(grid[key].index)

                c = df_c.loc[common_idx]
                e = grid[neighbors["east"]].loc[common_idx]
                w = grid[neighbors["west"]].loc[common_idx]
                n = grid[neighbors["north"]].loc[common_idx]
                s = grid[neighbors["south"]].loc[common_idx]

                grad_df = pd.DataFrame(index=common_idx)
                for label, u_col, v_col in WIND_LEVELS:
                    all_dfs = [c, e, w, n, s]
                    if not all(u_col in d.columns and v_col in d.columns for d in all_dfs):
                        continue
                    V_c = compute_wind_speed(c[u_col].values, c[v_col].values)
                    V_e = compute_wind_speed(e[u_col].values, e[v_col].values)
                    V_w = compute_wind_speed(w[u_col].values, w[v_col].values)
                    V_n = compute_wind_speed(n[u_col].values, n[v_col].values)
                    V_s = compute_wind_speed(s[u_col].values, s[v_col].values)
                    dV_dx = (V_e - V_w) / (2.0 * dx_km)
                    dV_dy = (V_n - V_s) / (2.0 * dy_km)
                    grad_df[f"grad_wind_speed_{label}"] = np.sqrt(dV_dx**2 + dV_dy**2)

                # 垂直风切变 + 幂指数
                required = [U_LOW_COL, V_LOW_COL, U_HIGH_COL, V_HIGH_COL]
                if all(c in df_c.columns for c in required):
                    u10, v10 = c[U_LOW_COL].values, c[V_LOW_COL].values
                    u100, v100 = c[U_HIGH_COL].values, c[V_HIGH_COL].values
                    dz = 90.0
                    shear_u = (u100 - u10) / dz
                    shear_v = (v100 - v10) / dz
                    v_10 = np.sqrt(u10**2 + v10**2)
                    v_100 = np.sqrt(u100**2 + v100**2)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ratio = np.where(v_10 > 1e-6, v_100 / np.where(v_10 > 1e-6, v_10, np.nan), np.nan)
                        alpha = np.log(ratio) / math.log(100.0 / 10.0)
                    shear_df = pd.DataFrame({
                        "wind_speed_10m": v_10,
                        "wind_speed_100m": v_100,
                        "vertical_wind_shear_u": shear_u,
                        "vertical_wind_shear_v": shear_v,
                        "vertical_wind_shear_u_per_100m": shear_u * 100,
                        "vertical_wind_shear_v_per_100m": shear_v * 100,
                        "power_law_alpha": alpha,
                    }, index=common_idx)
                    result = pd.concat([grad_df, shear_df], axis=1)
                else:
                    result = grad_df

                result.index.name = TIME_COL
                result = result.reset_index()
                coord_str = format_coord_str(lat_c, lon_c)
                out_name = f"gradient_{coord_str}.csv"
                result.to_csv(os.path.join(self.gradient_dir, out_name), index=False, encoding="utf-8")
                self.logger.info(f"  保存 {out_name}: 行={len(result)}, 列={len(result.columns)}")
                success += 1
            except Exception as e:
                self.logger.error(f"  ({x},{y}) 失败: {e}")

        self.logger.info(f"梯度计算完成: {success}/{len(interior)} 个内部点")
        return self.gradient_dir

    def step6_select_features(self, final_path: str = None) -> str:
        """风电 Step 6: 双模式特征选择 (默认 ~130 列, 自定义用 --features)"""
        final_path = final_path or getattr(self, "_final_path", "")
        self.logger.info("=" * 60)
        self.logger.info("Step 6 (风电): 特征选择")
        self.logger.info("=" * 60)

        df = pd.read_csv(final_path, low_memory=False)
        n_before = len(df.columns)
        self.logger.info(f"读取: 行={len(df)}, 列={n_before}")

        # 双模式: 自定义优先, 否则用默认 (动态生成含 center_grid 的特征列表)
        if self.custom_features:
            target_cols = self.custom_features
            self.logger.info(f"模式: 自定义 ({len(target_cols)} 列)")
        else:
            # 默认模式: 内联生成 ~130 列风电特征
            # (地表风分量 + 温湿压辐射 + 垂直风廓线 + 派生特征 + 梯度 + 风速)
            center = self.center_grid
            surrounding = []
            # 推断周围 4 个网格点
            parts = center.split("_") if center else []
            if len(parts) == 4:
                lat = float(f"{parts[0]}.{parts[1]}")
                lon = float(f"{parts[2]}.{parts[3]}")
                lat_step = self.lat_range[1] - self.lat_range[0]
                lon_step = self.lon_range[1] - self.lon_range[0]
                surrounding = [
                    format_coord_str(lat + lat_step, lon),  # 北
                    format_coord_str(lat - lat_step, lon),  # 南
                    format_coord_str(lat, lon - lon_step),  # 西
                    format_coord_str(lat, lon + lon_step),  # 东
                ]

            all_grids = [center] + surrounding
            hybrid_levels = list(range(130, 138))
            isobaric_levels = [500, 600, 700, 800, 850, 900, 925, 950, 1000]

            target_cols = ["time", "power", "ws"]
            # 1. 地表风分量
            target_cols += ["u_wind_component_surface_10_metre", "u_wind_component_surface_100_metre",
                            "v_wind_component_surface_10_metre", "v_wind_component_surface_100_metre"]
            # 2. 温湿压辐射
            target_cols += ["temperature_surface_2_metre", "dewpoint_temperature_surface_2_metre",
                            "surface_thermal_radiation_downwards_surface", "surface_pressure_surface",
                            "total_sky_direct_solar_radiation_at_surface_surface"]
            # 3. 混合层垂直风廓线
            for lv in hybrid_levels:
                target_cols.append(f"u_component_of_wind_hybrid_{lv}")
            for lv in hybrid_levels:
                target_cols.append(f"v_component_of_wind_hybrid_{lv}")
            # 4. 派生特征
            target_cols += [f"vertical_wind_shear_u__{center}", f"vertical_wind_shear_v__{center}",
                            f"vertical_wind_shear_u_per_100m__{center}", f"vertical_wind_shear_v_per_100m__{center}",
                            f"power_law_alpha__{center}"]
            # 5. 梯度 (5 点 × 19 层)
            for grid in all_grids:
                for lv in hybrid_levels:
                    target_cols.append(f"grad_wind_speed_hybrid_{lv}__{grid}")
                target_cols.append(f"grad_wind_speed_10m__{grid}")
                target_cols.append(f"grad_wind_speed_100m__{grid}")
                for lv in isobaric_levels:
                    target_cols.append(f"grad_wind_speed_isobaric_{lv}__{grid}")
            # 6. 中心点风速 (动态匹配 _x/_y/无后缀)
            for suffix in ["_x", "_y", ""]:
                col = f"wind_speed_10m__{center}{suffix}"
                if col in df.columns:
                    target_cols.append(col)
                    break
            for suffix in ["_x", "_y", ""]:
                col = f"wind_speed_100m__{center}{suffix}"
                if col in df.columns:
                    target_cols.append(col)
                    break
            self.logger.info(f"模式: 默认 ({len(target_cols)} 列, center={center})")

        # 模糊匹配 + 去重
        actual_set = set(df.columns)
        selected = []
        seen = set()
        missing = []
        for t in target_cols:
            matched = None
            if t in actual_set:
                matched = t
            elif t.endswith("_x"):
                alt = t[:-2]
                if alt in actual_set:
                    matched = alt
                elif alt + "_y" in actual_set:
                    matched = alt + "_y"
            elif t + "_x" in actual_set:
                matched = t + "_x"
            elif t + "_y" in actual_set:
                matched = t + "_y"
            if matched and matched not in seen:
                selected.append(matched)
                seen.add(matched)
            elif not matched and t not in ("time", "power", "ws"):
                missing.append(t)

        df_out = df[selected].copy()
        rename = {c: c[:-2] for c in df_out.columns if c.endswith("_x") or c.endswith("_y")}
        df_out = df_out.rename(columns=rename)

        n_after = len(df_out.columns)
        out_path = os.path.join(self.output_dir, f"train_data_wind_{self.center_grid}.csv")
        df_out.to_csv(out_path, index=False, encoding="utf-8")
        self.logger.info(f"特征选择: {n_before} -> {n_after} 列 (删除 {n_before - n_after})")
        if missing:
            self.logger.warning(f"缺失 {len(missing)} 列: {missing[:10]}")
        self.logger.info(f"输出: {out_path}")
        return out_path


# =============================================================================
# 光伏数据处理
# =============================================================================

class SolarDataProcessor(DataProcessor):
    """光伏发电数据处理 (跳过梯度计算, 侧重辐射特征)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 光伏电站网格定义可由子类覆盖 (宁河: 39.x, 117.x)
        # 默认沿用风电网格, 实际使用时由 --lat-range/--lon-range 覆盖

    def step3_compute_gradient(self, split_dir: str = None) -> Optional[str]:
        """光伏不需要梯度计算, 直接跳过"""
        self.logger.info("=" * 60)
        self.logger.info("Step 3 (光伏): 跳过梯度计算 (光伏不依赖风场梯度)")
        self.logger.info("=" * 60)
        return None

    # ------------------------------------------------------------------
    # 光伏异常0值检测与处理 (与 merge_csv_ninghe.py 对齐)
    # ------------------------------------------------------------------
    def _handle_special_zero_values(self, df: pd.DataFrame, data_cols: List[str]) -> pd.DataFrame:
        """光伏: 检测并处理异常0值 (限电日/太阳辐射白天为0/热辐射湿度不应为0)"""
        self.logger.info("-" * 40)
        self.logger.info("光伏异常0值检测 (nighttime_zero/solar/never_zero)")
        special_cases = self._detect_special_cases(df, data_cols)
        if special_cases:
            self.logger.info(f"检测到 {len(special_cases)} 个异常区间, 开始处理...")
            df = self._handle_special_cases_impl(df, special_cases)
        else:
            self.logger.info("未检测到异常0值")
        return df

    def _detect_special_cases(
        self, df: pd.DataFrame, target_columns: List[str],
        zero_threshold: float = 0.0, daytime_start: int = 7, daytime_end: int = 19,
    ) -> List[Dict]:
        """检测异常0值, 按列类型和昼夜规律分类 (与 merge_csv_ninghe.py 对齐)"""
        if not target_columns:
            return []
        df = df.copy()
        df["_hour"] = df["_timestamp"].dt.hour
        df["_date"] = df["_timestamp"].dt.date
        test_cutoff_date = None
        if self.test_cutoff:
            try:
                test_cutoff_date = pd.Timestamp(self.test_cutoff).date()
                self.logger.info(f"测试期起始: {self.test_cutoff}, 内生变量不检测此日期之后")
            except Exception:
                self.logger.warning(f"无法解析 test_cutoff '{self.test_cutoff}', 忽略")
        results: List[Dict] = []
        for col in target_columns:
            if col not in df.columns:
                self.logger.warning(f"检测列不存在: '{col}', 跳过")
                continue
            col_type = classify_column(col)
            if col_type == "other":
                self.logger.info(f"列 '{col}' 分类为 'other', 跳过特殊检测")
                continue
            is_endogenous = col_type == "nighttime_zero"
            for date, day_df in df.groupby("_date"):
                if is_endogenous and test_cutoff_date and date >= test_cutoff_date:
                    continue
                is_abnormal = False
                if col_type == "nighttime_zero":
                    is_abnormal = (day_df[col].abs() <= zero_threshold).all()
                elif col_type == "solar":
                    day_hours = day_df[(day_df["_hour"] >= daytime_start) & (day_df["_hour"] < daytime_end)]
                    if day_hours.empty:
                        continue
                    is_abnormal = (day_hours[col].abs() <= zero_threshold).all()
                elif col_type == "never_zero":
                    zero_mask = day_df[col].abs() <= zero_threshold
                    if zero_mask.any():
                        is_abnormal = True
                if is_abnormal:
                    if col_type == "never_zero":
                        zero_rows = day_df[day_df[col].abs() <= zero_threshold]
                        start_idx, end_idx = zero_rows.index[0], zero_rows.index[-1]
                        start_time = df.loc[start_idx, "_timestamp"]
                        end_time = df.loc[end_idx, "_timestamp"]
                        results.append({"column": col, "col_type": col_type, "date": str(date),
                            "start_time": start_time, "end_time": end_time,
                            "duration_hours": round((end_time - start_time).total_seconds() / 3600, 2),
                            "row_count": len(zero_rows), "start_idx": int(start_idx), "end_idx": int(end_idx)})
                    else:
                        start_time = day_df["_timestamp"].iloc[0]
                        end_time = day_df["_timestamp"].iloc[-1]
                        start_idx, end_idx = day_df.index[0], day_df.index[-1]
                        results.append({"column": col, "col_type": col_type, "date": str(date),
                            "start_time": start_time, "end_time": end_time,
                            "duration_hours": round((end_time - start_time).total_seconds() / 3600, 2),
                            "row_count": len(day_df), "start_idx": int(start_idx), "end_idx": int(end_idx)})
        df = df.drop(columns=["_hour", "_date"])
        if results:
            self.logger.info(f"检测到 {len(results)} 个异常0值区间:")
            for r in results:
                self.logger.info(f"  列 '{r['column']}' [{r['col_type']}]: {r['date']}, {r['row_count']} 行")
        return results

    def _handle_special_cases_impl(self, df: pd.DataFrame, special_cases: List[Dict]) -> pd.DataFrame:
        """处理异常0值: never_zero强制插值, nighttime_zero/solar按fill_method"""
        df = df.copy()
        never_zero_cases = [c for c in special_cases if c.get("col_type") == "never_zero"]
        other_cases = [c for c in special_cases if c.get("col_type") != "never_zero"]
        # never_zero 类: 强制插值替换 (忽略策略)
        if never_zero_cases:
            for case in never_zero_cases:
                col = case["column"]
                mask = (df.index >= case["start_idx"]) & (df.index <= case["end_idx"])
                zero_mask = mask & (df[col].abs() <= 1e-10)
                if zero_mask.any():
                    df.loc[zero_mask, col] = np.nan
            for col in set(c["column"] for c in never_zero_cases):
                if col in df.columns:
                    df[col] = df[col].interpolate(method="linear", limit_direction="both")
                    df[col] = df[col].ffill().bfill()
            self.logger.info(f"never_zero 类处理: {len(never_zero_cases)} 个区间, 插值替换")
        # nighttime_zero / solar 类
        if not other_cases or self.fill_method == "keep":
            if other_cases:
                self.logger.info("nighttime_zero/solar 类: 保留不处理 (fill_method=keep)")
            return df
        if self.fill_method == "drop":
            drop_indices = set()
            for case in other_cases:
                drop_indices.update(range(case["start_idx"], case["end_idx"] + 1))
            drop_indices = drop_indices & set(df.index)
            before_len = len(df)
            df = df.drop(index=drop_indices).reset_index(drop=True)
            self.logger.info(f"nighttime_zero/solar 处理(删除): {before_len} -> {len(df)} 行")
        elif self.fill_method in ("mean", "fill_mean", "fill_median"):
            for case in other_cases:
                col = case["column"]
                mask = (df.index >= case["start_idx"]) & (df.index <= case["end_idx"])
                non_special = df.loc[~mask, col]
                fill_val = non_special.mean() if self.fill_method in ("mean", "fill_mean") else non_special.median()
                df.loc[mask, col] = fill_val
            self.logger.info(f"nighttime_zero/solar 处理({self.fill_method}): {len(other_cases)} 个区间")
        elif self.fill_method == "interpolate":
            for case in other_cases:
                col = case["column"]
                mask = (df.index >= case["start_idx"]) & (df.index <= case["end_idx"])
                df.loc[mask, col] = np.nan
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
            df[numeric_cols] = df[numeric_cols].ffill().bfill()
            self.logger.info(f"nighttime_zero/solar 处理(插值): {len(other_cases)} 个区间")
        elif self.fill_method == "fill_periodic_mean":
            neighbors = self.periodic_neighbors
            total_len = len(df)
            for case in other_cases:
                col = case["column"]
                start_idx, end_idx = case["start_idx"], case["end_idx"]
                positions_in_period = np.arange(start_idx, end_idx + 1) % PERIOD
                case_period_num = start_idx // PERIOD
                min_period = max(0, case_period_num - neighbors)
                max_period = min((total_len - 1) // PERIOD, case_period_num + neighbors)
                fill_values = np.full(end_idx - start_idx + 1, np.nan)
                for i, pos in enumerate(positions_in_period):
                    neighbor_values = []
                    for p in range(min_period, max_period + 1):
                        if p == case_period_num:
                            continue
                        idx = p * PERIOD + pos
                        if 0 <= idx < total_len:
                            val = df.loc[idx, col]
                            if not pd.isna(val):
                                neighbor_values.append(val)
                    fill_values[i] = np.mean(neighbor_values) if neighbor_values else 0.0
                for i, idx in enumerate(range(start_idx, end_idx + 1)):
                    df.loc[idx, col] = fill_values[i]
            self.logger.info(f"nighttime_zero/solar 处理(多周期平均, 前后各{neighbors}天): {len(other_cases)} 个区间")
        else:
            self.logger.warning(f"未知异常处理策略: {self.fill_method}, 跳过 nighttime_zero/solar 类处理")
        return df

    def step4_merge_master(self, split_dir: str = None, gradient_dir: str = None) -> str:
        """光伏 Step 4: 仅加载主文件, 跳过 25 点风速和梯度合并"""
        split_dir = split_dir or getattr(self, "_split_dir", "")
        os.makedirs(self.merged_dir, exist_ok=True)

        self.logger.info("=" * 60)
        self.logger.info("Step 4 (光伏): 仅加载主文件 (跳过风速/梯度合并)")
        self.logger.info("=" * 60)

        master_name = f"{self.center_grid}.csv"
        master_path = os.path.join(split_dir, master_name)
        if not os.path.isfile(master_path):
            raise FileNotFoundError(f"中心点主文件不存在: {master_path}")
        self.logger.info(f"主文件: {master_path}")
        master_df = pd.read_csv(master_path, encoding="utf-8")
        master_df[TIME_COL] = master_df[TIME_COL].astype(str).str.strip()
        master_df = master_df.drop_duplicates(subset=TIME_COL, keep="first").sort_values(TIME_COL).reset_index(drop=True)
        self.logger.info(f"主文件: 行={len(master_df)}, 列={len(master_df.columns)}")

        out_path = os.path.join(self.merged_dir, f"merged_{self.center_grid}.csv")
        master_df.to_csv(out_path, index=False, encoding="utf-8")
        self.logger.info(f"光伏合并结果 (仅主文件): {out_path}")
        self._merged_path = out_path
        return out_path

    def step6_select_features(self, final_path: str = None) -> str:
        """光伏 Step 6: 双模式特征选择 (默认模式用 SOLAR_DEFAULT_FEATURES, 自定义模式用 --features)"""
        final_path = final_path or getattr(self, "_final_path", "")
        self.logger.info("=" * 60)
        self.logger.info("Step 6 (光伏): 特征选择")
        self.logger.info("=" * 60)

        df = pd.read_csv(final_path, low_memory=False)
        n_before = len(df.columns)
        self.logger.info(f"读取: 行={len(df)}, 列={n_before}")

        # 双模式: 自定义优先, 否则用默认
        if self.custom_features:
            target_cols = self.custom_features
            self.logger.info(f"模式: 自定义 ({len(target_cols)} 列)")
        else:
            target_cols = list(SOLAR_DEFAULT_FEATURES)
            self.logger.info(f"模式: 默认 ({len(target_cols)} 列: {SOLAR_DEFAULT_FEATURES})")

        # 模糊匹配 + 去重 (与风电逻辑一致)
        actual_set = set(df.columns)
        selected = []
        seen = set()
        missing = []
        for t in target_cols:
            matched = None
            if t in actual_set:
                matched = t
            elif t.endswith("_x"):
                alt = t[:-2]
                if alt in actual_set:
                    matched = alt
                elif alt + "_y" in actual_set:
                    matched = alt + "_y"
            elif t + "_x" in actual_set:
                matched = t + "_x"
            elif t + "_y" in actual_set:
                matched = t + "_y"
            if matched and matched not in seen:
                selected.append(matched)
                seen.add(matched)
            elif not matched and t not in ("time", "power", "sr"):
                missing.append(t)

        df_out = df[selected].copy()
        rename = {c: c[:-2] for c in df_out.columns if c.endswith("_x") or c.endswith("_y")}
        df_out = df_out.rename(columns=rename)

        n_after = len(df_out.columns)
        out_path = os.path.join(self.output_dir, f"train_data_solar_{self.center_grid or 'default'}.csv")
        df_out.to_csv(out_path, index=False, encoding="utf-8")
        self.logger.info(f"特征选择: {n_before} -> {n_after} 列 (删除 {n_before - n_after})")
        if missing:
            self.logger.warning(f"缺失 {len(missing)} 列: {missing[:10]}")
        self.logger.info(f"最终列: {list(df_out.columns)}")
        self.logger.info(f"输出: {out_path}")
        return out_path


# =============================================================================
# 主入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="玖天数据集统一处理模块 (风电/光伏)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 风电 (郝家营)
  python tools/data_processor.py --plant-type wind \\
      --ec-data "玖天数据集处理/haojiaying.csv" \\
      --power-data "玖天数据集处理/郝家营风电场二期power-ws.xlsx" \\
      --work-dir "玖天数据集处理/processed_wind" \\
      --center-grid 41_400_114_900

  # 光伏 (宁河)
  python tools/data_processor.py --plant-type solar \\
      --ec-data "玖天数据集处理/ninghe.csv" \\
      --power-data "玖天数据集处理/宁河光伏电站power-sr.csv" \\
      --work-dir "玖天数据集处理/processed_solar" \\
      --center-grid 39_200_117_400

  # 仅执行指定步骤
  python tools/data_processor.py --plant-type wind ... --steps 1,2,5,6
        """,
    )
    parser.add_argument("--plant-type", required=True, choices=["wind", "solar"], help="发电类型: wind(风电) / solar(光伏)")
    parser.add_argument("--ec-data", required=True, help="原始 EC 气象数据 CSV 路径")
    parser.add_argument("--power-data", required=True, help="功率实测数据路径 (CSV/Excel)")
    parser.add_argument("--work-dir", required=True, help="工作目录 (存放中间和最终结果)")
    parser.add_argument("--time-start", default="202407030000", help="起始时间 (YYYYMMDDHHmm, 默认 202407030000)")
    parser.add_argument("--time-end", default="202607312345", help="结束时间 (YYYYMMDDHHmm, 默认 202607312345)")
    parser.add_argument("--center-grid", default=None, help="中心网格点坐标 (如 41_400_114_900), 不指定则用第一个点")
    parser.add_argument("--fill-method", default="fill_periodic_mean", choices=["fill_periodic_mean", "interpolate", "mean", "keep"], help="缺失值填充方法: fill_periodic_mean(默认) / interpolate(线性插值) / mean(均值填充) / keep(光伏异常0值保留不处理)")
    parser.add_argument("--periodic-neighbors", type=int, default=3, help="周期插补前后天数 (默认 3)")
    parser.add_argument("--test-cutoff", default=None, help="光伏测试期起始日期 (如 2026-03-01), 内生变量(power/sr)不检测此日期之后的异常0值, 保护评估数据")
    parser.add_argument("--steps", default=None, help="执行步骤 (逗号分隔, 如 1,2,5,6; 默认全部)")
    parser.add_argument("--features", default=None, help="自定义特征列表 (逗号分隔, 如 'time,power,ws,u_wind_component_surface_10_metre')")
    parser.add_argument("--features-file", default=None, help="从文件加载自定义特征列表 (每行一个或逗号分隔)")
    parser.add_argument("--lat-range", default=None, help="纬度范围 (逗号分隔, 如 41.2,41.3,41.4,41.5,41.6)")
    parser.add_argument("--lon-range", default=None, help="经度范围 (逗号分隔, 如 114.7,114.8,114.9,115.0,115.1)")
    args = parser.parse_args()

    steps = [int(s) for s in args.steps.split(",")] if args.steps else None

    # 解析自定义特征列表 (逗号分隔字符串 -> List[str])
    custom_features = None
    if args.features:
        custom_features = [c.strip() for c in args.features.split(",") if c.strip()]

    # 选择处理器
    if args.plant_type == "wind":
        processor = WindDataProcessor(
            ec_data_path=args.ec_data,
            power_data_path=args.power_data,
            work_dir=args.work_dir,
            time_start=args.time_start,
            time_end=args.time_end,
            center_grid=args.center_grid,
            fill_method=args.fill_method,
            periodic_neighbors=args.periodic_neighbors,
            features=custom_features,
            features_file=args.features_file,
            test_cutoff=args.test_cutoff,
        )
    else:
        processor = SolarDataProcessor(
            ec_data_path=args.ec_data,
            power_data_path=args.power_data,
            work_dir=args.work_dir,
            time_start=args.time_start,
            time_end=args.time_end,
            center_grid=args.center_grid,
            fill_method=args.fill_method,
            periodic_neighbors=args.periodic_neighbors,
            features=custom_features,
            features_file=args.features_file,
            test_cutoff=args.test_cutoff,
        )

    # 覆盖网格范围
    if args.lat_range:
        processor.lat_range = [float(x) for x in args.lat_range.split(",")]
    if args.lon_range:
        processor.lon_range = [float(x) for x in args.lon_range.split(",")]

    # 自动推断 center_grid (如果未指定)
    if not processor.center_grid:
        # Step 1 后再推断, 暂时设为 None, 在 step4 中报错提示
        processor.logger.warning("未指定 --center-grid, 将在 Step 1 后自动选取第一个点")

    processor.run(steps=steps)


if __name__ == "__main__":
    main()
