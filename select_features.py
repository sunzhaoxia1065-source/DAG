#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集二次处理: 从全量气象协变量中选择对预测 power 有用的特征列。

选择规则 (6 类):
  1. 中心点地表风分量: 10m/100m 的 u、v 分量 (4列)
  2. 中心点地表温湿压辐射: 2m温度、地表气压、地表热辐射、太阳辐射、露点温度 (5列)
  3. 中心点混合层垂直风廓线: hybrid 130-137 的 u、v 分量 (16列)
  4. 中心点派生特征: 垂直风切变(u/v)、每百米风切变(u/v)、风切变幂指数 (5列)
  5. 中心点+周围4网格点的梯度 grad_wind_speed: hybrid(8)+10m/100m(2)+isobaric(9)=19列/点 × 5点 (95列)
  6. 中心网格点 10m/100m 风速 (2列)

总计: 3(base) + 4 + 5 + 16 + 5 + 95 + 2 = 130列

用法:
  python select_features.py --input 输入文件.csv --output 输出文件.csv
  python select_features.py -i 输入文件.csv -o 输出文件.csv --center 41_400_114_900
  python select_features.py -i 输入文件.csv -o 输出文件.csv --interactive
"""

import argparse
import os
import re
import sys
from typing import List, Set

import pandas as pd


# =============================================================================
# 网格点配置
# =============================================================================
# 中心网格点 (3×3 grad 网格的几何中心)
# grad 数据覆盖 3×3 网格:
#   41_300_114_800  41_300_114_900  41_300_115_000
#   41_400_114_800  41_400_114_900  41_400_115_000
#   41_500_114_800  41_500_114_900  41_500_115_000
# 中心 = 41_400_114_900, 周围4点 = 上下左右
CENTER_GRID = "41_400_114_900"
SURROUNDING_GRIDS = [
    "41_300_114_900",  # 北
    "41_500_114_900",  # 南
    "41_400_114_800",  # 西
    "41_400_115_000",  # 东
]
ALL_GRAD_GRIDS = [CENTER_GRID] + SURROUNDING_GRIDS  # 5个网格点

# hybrid 层级
HYBRID_LEVELS = list(range(130, 138))  # 130-137

# isobaric 层级 (grad 中的等压面)
ISOBARIC_LEVELS = [500, 600, 700, 800, 850, 900, 925, 950, 1000]


# =============================================================================
# 列选择规则
# =============================================================================
def build_selected_columns(center_grid: str) -> List[str]:
    """
    根据选择规则构建目标列名列表 (有序)。

    Parameters
    ----------
    center_grid : str
        中心网格点坐标, 如 "41_400_114_900"

    Returns
    -------
    list[str]
        按顺序排列的目标列名列表
    """
    cols = []

    # --- 基础列 (始终保留) ---
    cols += ["time", "power", "ws"]

    # --- 1. 中心点地表风分量 (10m/100m u/v) ---
    cols += [
        "u_wind_component_surface_10_metre",
        "u_wind_component_surface_100_metre",
        "v_wind_component_surface_10_metre",
        "v_wind_component_surface_100_metre",
    ]

    # --- 2. 中心点地表温湿压辐射 (5个) ---
    cols += [
        "temperature_surface_2_metre",
        "dewpoint_temperature_surface_2_metre",
        "surface_thermal_radiation_downwards_surface",
        "surface_pressure_surface",
        "total_sky_direct_solar_radiation_at_surface_surface",
    ]

    # --- 3. 中心点混合层垂直风廓线 (hybrid 130-137 u/v, 16列) ---
    for level in HYBRID_LEVELS:
        cols.append(f"u_component_of_wind_hybrid_{level}")
    for level in HYBRID_LEVELS:
        cols.append(f"v_component_of_wind_hybrid_{level}")

    # --- 4. 中心点派生特征 (垂直风切变 + 幂指数, 5列) ---
    cols += [
        f"vertical_wind_shear_u__{center_grid}",
        f"vertical_wind_shear_v__{center_grid}",
        f"vertical_wind_shear_u_per_100m__{center_grid}",
        f"vertical_wind_shear_v_per_100m__{center_grid}",
        f"power_law_alpha__{center_grid}",
    ]

    # --- 5. 中心点+周围4网格点的梯度 grad_wind_speed (19列/点 × 5点 = 95列) ---
    # 每个网格点的 grad 列:
    #   grad_wind_speed_hybrid_130~137 (8)
    #   grad_wind_speed_10m, grad_wind_speed_100m (2)
    #   grad_wind_speed_isobaric_500~1000 (9)
    all_grids = [center_grid] + SURROUNDING_GRIDS
    for grid in all_grids:
        for level in HYBRID_LEVELS:
            cols.append(f"grad_wind_speed_hybrid_{level}__{grid}")
        cols.append(f"grad_wind_speed_10m__{grid}")
        cols.append(f"grad_wind_speed_100m__{grid}")
        for level in ISOBARIC_LEVELS:
            cols.append(f"grad_wind_speed_isobaric_{level}__{grid}")

    # --- 6. 中心网格点 10m/100m 风速 (2列) ---
    # 风速列有 _x (wind_speed段) 和 _y (grad段) 两套重复, 优先用 _x
    cols.append(f"wind_speed_10m__{center_grid}_x")
    cols.append(f"wind_speed_100m__{center_grid}_x")

    return cols


# =============================================================================
# 模糊匹配: 处理列名后缀差异 (_x / _y / 无后缀)
# =============================================================================
def match_columns(target_cols: List[str], actual_cols: List[str]) -> dict:
    """
    将目标列名匹配到实际数据集中的列名。

    匹配规则 (按优先级):
    1. 精确匹配
    2. 去掉 _x 后缀匹配 (wind_speed_10m__41_400_114_900_x → wind_speed_10m__41_400_114_900)
    3. _x 不存在时尝试 _y 后缀
    4. 都不存在时尝试无后缀

    Parameters
    ----------
    target_cols : list[str]
        期望的列名列表
    actual_cols : list[str]
        实际数据集中的列名列表

    Returns
    -------
    dict
        {目标列名: 实际列名} 或 {目标列名: None} (未找到)
    """
    actual_set = set(actual_cols)
    mapping = {}

    for target in target_cols:
        # 1. 精确匹配
        if target in actual_set:
            mapping[target] = target
            continue

        # 2. 如果目标列以 _x 结尾, 尝试 _y
        if target.endswith("_x"):
            alt = target[:-2] + "_y"
            if alt in actual_set:
                mapping[target] = alt
                continue
            # 尝试无后缀
            alt_plain = target[:-2]
            if alt_plain in actual_set:
                mapping[target] = alt_plain
                continue

        # 3. 如果目标列不以 _x 结尾, 尝试加 _x
        alt_x = target + "_x"
        if alt_x in actual_set:
            mapping[target] = alt_x
            continue

        # 4. 尝试 _y
        alt_y = target + "_y"
        if alt_y in actual_set:
            mapping[target] = alt_y
            continue

        # 5. 未找到
        mapping[target] = None

    return mapping


# =============================================================================
# 主处理函数
# =============================================================================
def process_dataset(
    input_path: str,
    output_path: str,
    center_grid: str = CENTER_GRID,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    读取数据集, 选择目标列, 输出处理后的 CSV。

    Parameters
    ----------
    input_path : str
        输入 CSV 文件路径
    output_path : str
        输出 CSV 文件路径
    center_grid : str
        中心网格点坐标
    verbose : bool
        是否打印详细统计信息

    Returns
    -------
    pd.DataFrame
        处理后的数据
    """
    # 读取数据
    file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    if verbose:
        print(f"[读取] {input_path} ({file_size_mb:.1f} MB)")

    df = pd.read_csv(input_path, low_memory=False)
    n_rows = len(df)
    n_cols_before = len(df.columns)
    if verbose:
        print(f"[读取] {n_rows} 行, {n_cols_before} 列")

    # 构建目标列名
    target_cols = build_selected_columns(center_grid)

    # 匹配实际列名
    mapping = match_columns(target_cols, list(df.columns))

    # 统计匹配结果
    found = {k: v for k, v in mapping.items() if v is not None}
    missing = {k: v for k, v in mapping.items() if v is None}

    if verbose:
        print(f"\n[选择] 目标列 {len(target_cols)} 个, 匹配到 {len(found)} 个, 缺失 {len(missing)} 个")
        if missing:
            print("[缺失] 以下列在数据集中未找到:")
            for col in missing:
                print(f"  - {col}")

    # 构建实际要保留的列 (保持目标顺序)
    selected_actual = []
    for target in target_cols:
        actual = mapping.get(target)
        if actual is not None:
            selected_actual.append(actual)

    # 重命名: 把 _x 后缀去掉 (如 wind_speed_10m__41_400_114_900_x → wind_speed_10m__41_400_114_900)
    rename_map = {}
    for target, actual in found.items():
        if actual != target:
            rename_map[actual] = target

    # 选取列
    df_out = df[selected_actual].copy()

    # 重命名
    if rename_map:
        df_out = df_out.rename(columns=rename_map)

    # 去重 (防止同一列被选两次)
    df_out = df_out.loc[:, ~df_out.columns.duplicated()]

    n_cols_after = len(df_out.columns)
    if verbose:
        print(f"\n[输出] {n_rows} 行, {n_cols_after} 列 (原始 {n_cols_before} 列, 删除 {n_cols_before - n_cols_after} 列)")
        print(f"[输出] 列名: {list(df_out.columns)}")

    # 保存
    df_out.to_csv(output_path, index=False)
    if verbose:
        out_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[保存] {output_path} ({out_size_mb:.1f} MB)")

    return df_out


# =============================================================================
# 交互式模式
# =============================================================================
def interactive_mode():
    """交互式输入参数。"""
    print("=" * 60)
    print("数据集特征选择工具")
    print("=" * 60)

    input_path = input("输入文件路径: ").strip().strip('"').strip("'")
    if not os.path.exists(input_path):
        print(f"[错误] 文件不存在: {input_path}")
        sys.exit(1)

    output_path = input("输出文件路径 (回车使用默认): ").strip().strip('"').strip("'")
    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_selected{ext}"

    print(f"\n可用网格点 (从列名中检测):")
    # 快速读取列名
    df_cols = pd.read_csv(input_path, nrows=0).columns.tolist()
    grids = set()
    for col in df_cols:
        m = re.search(r"__(\d{3}_\d{3}_\d{3,6})", col)
        if m:
            grids.add(m.group(1))
    for g in sorted(grids):
        marker = " ← 默认中心" if g == CENTER_GRID else ""
        print(f"  {g}{marker}")

    center = input(f"\n中心网格点 (回车使用默认 {CENTER_GRID}): ").strip()
    if not center:
        center = CENTER_GRID

    process_dataset(input_path, output_path, center_grid=center)


# =============================================================================
# 命令行入口
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="从全量气象数据集中选择对预测 power 有用的特征列",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
选择规则:
  1. 中心点地表风分量 (10m/100m u/v, 4列)
  2. 中心点地表温湿压辐射 (5列)
  3. 中心点混合层垂直风廓线 (hybrid 130-137 u/v, 16列)
  4. 中心点派生特征 (风切变+幂指数, 5列)
  5. 中心+周围4点梯度 grad_wind_speed (19列/点 × 5点 = 95列)
  6. 中心点 10m/100m 风速 (2列)
  总计: ~130列

示例:
  python select_features.py -i data.csv -o data_selected.csv
  python select_features.py -i data.csv -o out.csv --center 41_300_114_800
  python select_features.py --interactive
        """,
    )
    parser.add_argument("-i", "--input", help="输入 CSV 文件路径")
    parser.add_argument("-o", "--output", help="输出 CSV 文件路径")
    parser.add_argument(
        "--center",
        default=CENTER_GRID,
        help=f"中心网格点坐标 (默认: {CENTER_GRID})",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="交互式模式",
    )
    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
        return

    if not args.input:
        parser.error("请指定输入文件路径 (--input 或 -i)")
    if not os.path.exists(args.input):
        parser.error(f"文件不存在: {args.input}")

    output = args.output
    if not output:
        base, ext = os.path.splitext(args.input)
        output = f"{base}_selected{ext}"

    process_dataset(args.input, output, center_grid=args.center)


if __name__ == "__main__":
    main()
