#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集合并工具

功能:
  1. 读取CSV数据集文件，筛选 time, power, ws/sr 列
  2. 将 Excel 文件(如 power-ws.xlsx)转换为 CSV 格式
  3. 两个文件按时间轴合并(前半段 + 后半段)
  4. 输出合并后的完整 CSV 文件

使用方式:
  # 基本用法(自动检测ws/sr列, 自动判断时间先后)
  python merge_datasets.py --csv-file "path/to/dataset.csv" --excel-file "power-ws.xlsx" --output "merged.csv"

  # 指定风列名(不自动检测)
  python merge_datasets.py --csv-file "dataset.csv" --excel-file "power-ws.xlsx" --wind-col ws

  # 仅将Excel转为CSV(不做合并)
  python merge_datasets.py --excel-file "power-ws.xlsx" --excel-only --output "converted.csv"

  # 交互模式(不传参数时自动进入)
  python merge_datasets.py
"""

import argparse
import os
import sys
from datetime import datetime

try:
    import pandas as pd
except ImportError:
    print("[错误] 未安装 pandas，请运行: pip install pandas openpyxl")
    sys.exit(1)


# ============================================================
# 常量定义
# ============================================================

# 必须保留的列
REQUIRED_COLUMNS = ["time", "power"]

# 候选风列名(按优先级排列)
WIND_COLUMN_CANDIDATES = ["ws", "sr", "wind_speed", "wind"]

# 时间列可能的格式
TIME_FORMATS = [
    "%Y-%m-%d %H:%M:%S",      # 2026-04-01 00:00:00
    "%Y-%m-%d %H:%M",          # 2026-04-01 00:00
    "%Y%m%d%H%M",              # 202604010000
    "%Y-%m-%d",                # 2026-04-01
]


# ============================================================
# 工具函数
# ============================================================


def detect_wind_column(columns):
    """
    从列名中自动检测风列(ws 或 sr)

    Parameters
    ----------
    columns : list
        数据集的列名列表

    Returns
    -------
    str or None
        检测到的风列名，未找到返回 None
    """
    for candidate in WIND_COLUMN_CANDIDATES:
        if candidate in columns:
            return candidate

    # 模糊匹配：列名中包含 ws 或 sr
    for col in columns:
        col_lower = col.lower()
        if "ws" == col_lower or "wind" in col_lower:
            return col
        if "sr" == col_lower or "solar" in col_lower or "radiation" in col_lower:
            return col

    return None


def parse_time_column(series):
    """
    尝试多种格式解析时间列，统一为 pd.Timestamp

    Parameters
    ----------
    series : pd.Series
        时间列数据

    Returns
    -------
    pd.Series
        解析后的时间列(DatetimeIndex)
    """
    # 先尝试 pd.to_datetime 自动推断
    try:
        result = pd.to_datetime(series, errors="coerce")
        if result.notna().sum() > len(series) * 0.5:
            return result
    except Exception:
        pass

    # 逐个尝试指定格式
    for fmt in TIME_FORMATS:
        try:
            result = pd.to_datetime(series, format=fmt, errors="coerce")
            if result.notna().sum() > len(series) * 0.5:
                return result
        except Exception:
            continue

    # 最后再尝试一次(不指定format,让pandas自己猜)
    return pd.to_datetime(series, errors="coerce")


def get_time_range_string(series):
    """获取时间范围描述字符串"""
    if series is None or len(series) == 0:
        return "空"
    ts = parse_time_column(series)
    valid = ts.dropna()
    if len(valid) == 0:
        return "无有效时间"
    return f"{valid.min()} ~ {valid.max()} (共 {len(valid)} 行)"


# ============================================================
# 核心功能
# ============================================================


def load_and_filter_csv(csv_path, wind_col=None, time_start=None, time_end=None):
    """
    读取CSV文件，筛选 time, power, ws/sr 列，并可按时间范围过滤

    Parameters
    ----------
    csv_path : str
        CSV文件路径
    wind_col : str, optional
        指定风列名，None则自动检测
    time_start : str, optional
        起始时间(含)，支持多种格式如 "2024-07-03"、"2024-07-03 00:00:00"、"202407030000"
    time_end : str, optional
        结束时间(含)，支持多种格式

    Returns
    -------
    pd.DataFrame
        筛选后的数据(含 time, power, wind 列)
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV文件不存在: {csv_path}")

    # 读取CSV(只读前5行获取列名,避免全量加载)
    df_sample = pd.read_csv(csv_path, nrows=5, encoding="utf-8")
    all_columns = list(df_sample.columns)

    # 确定风列
    if wind_col is None:
        wind_col = detect_wind_column(all_columns)
        if wind_col is None:
            raise ValueError(
                f"未找到风列(候选: {WIND_COLUMN_CANDIDATES})，"
                f"可用列: {all_columns}\n"
                f"请通过 --wind-col 参数手动指定"
            )

    # 验证必需列
    missing = [c for c in REQUIRED_COLUMNS if c not in all_columns]
    if missing:
        raise ValueError(f"CSV缺少必需列: {missing}，可用列: {all_columns}")

    if wind_col not in all_columns:
        raise ValueError(f"指定的风列 '{wind_col}' 不存在，可用列: {all_columns}")

    # 读取完整数据(只保留需要的列,节省内存)
    usecols = ["time", "power", wind_col]
    print(f"[信息] CSV筛选列: {usecols}")

    # 尝试不同编码
    for encoding in ["utf-8", "gbk", "gb2312", "latin1"]:
        try:
            df = pd.read_csv(csv_path, usecols=usecols, encoding=encoding)
            print(f"[信息] CSV编码: {encoding}")
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        raise ValueError(f"无法解码CSV文件，尝试了多种编码: utf-8, gbk, gb2312, latin1")

    # 解析时间列
    df["time"] = parse_time_column(df["time"])

    # 移除时间解析失败的行
    invalid_count = df["time"].isna().sum()
    if invalid_count > 0:
        print(f"[警告] {invalid_count} 行时间解析失败，已移除")
        df = df.dropna(subset=["time"])

    # 去重(按time)
    dup_count = df["time"].duplicated().sum()
    if dup_count > 0:
        print(f"[警告] {dup_count} 行时间重复，保留第一条")
        df = df.drop_duplicates(subset=["time"], keep="first")

    # 时间范围筛选
    if time_start is not None:
        t_start = parse_time_column(pd.Series([time_start])).iloc[0]
        if pd.isna(t_start):
            raise ValueError(f"起始时间格式无法解析: {time_start}")
        before_count = len(df)
        df = df[df["time"] >= t_start]
        print(f"[信息] 时间筛选: 起始 >= {t_start}，移除 {before_count - len(df)} 行")

    if time_end is not None:
        t_end = parse_time_column(pd.Series([time_end])).iloc[0]
        if pd.isna(t_end):
            raise ValueError(f"结束时间格式无法解析: {time_end}")
        before_count = len(df)
        df = df[df["time"] <= t_end]
        print(f"[信息] 时间筛选: 结束 <= {t_end}，移除 {before_count - len(df)} 行")

    # 按时间排序
    df = df.sort_values("time").reset_index(drop=True)

    return df, wind_col


def excel_to_csv(excel_path, output_csv=None, wind_col=None):
    """
    将Excel文件转换为CSV格式

    Parameters
    ----------
    excel_path : str
        Excel文件路径(.xlsx或.xls)
    output_csv : str, optional
        输出CSV路径，None则返回DataFrame不写文件
    wind_col : str, optional
        指定风列名，None则自动检测

    Returns
    -------
    pd.DataFrame
        转换后的数据
    str
        使用的风列名
    """
    if not os.path.isfile(excel_path):
        raise FileNotFoundError(f"Excel文件不存在: {excel_path}")

    # 读取Excel
    print(f"[信息] 读取Excel: {excel_path}")
    try:
        df = pd.read_excel(excel_path, engine="openpyxl")
    except Exception as e1:
        # 尝试xlrd引擎(老版.xls)
        try:
            df = pd.read_excel(excel_path, engine="xlrd")
            print(f"[信息] Excel引擎: xlrd")
        except Exception:
            raise ValueError(
                f"Excel读取失败(openpyxl): {e1}\n"
                f"请安装: pip install openpyxl"
            )

    print(f"[信息] Excel原始列: {list(df.columns)}")
    print(f"[信息] Excel行数: {len(df)}")

    # 确定风列
    all_columns = list(df.columns)
    if wind_col is None:
        wind_col = detect_wind_column(all_columns)
        if wind_col is None:
            raise ValueError(
                f"Excel中未找到风列(候选: {WIND_COLUMN_CANDIDATES})，"
                f"可用列: {all_columns}\n"
                f"请通过 --wind-col 参数手动指定"
            )

    # 验证必需列
    missing = [c for c in REQUIRED_COLUMNS if c not in all_columns]
    if missing:
        raise ValueError(f"Excel缺少必需列: {missing}，可用列: {all_columns}")

    if wind_col not in all_columns:
        raise ValueError(f"Excel中指定的风列 '{wind_col}' 不存在，可用列: {all_columns}")

    # 筛选列
    usecols = ["time", "power", wind_col]
    df = df[usecols].copy()
    print(f"[信息] Excel筛选列: {usecols}")

    # 解析时间列
    df["time"] = parse_time_column(df["time"])

    # 移除时间解析失败的行
    invalid_count = df["time"].isna().sum()
    if invalid_count > 0:
        print(f"[警告] Excel中 {invalid_count} 行时间解析失败，已移除")
        df = df.dropna(subset=["time"])

    # 去重
    dup_count = df["time"].duplicated().sum()
    if dup_count > 0:
        print(f"[警告] Excel中 {dup_count} 行时间重复，保留第一条")
        df = df.drop_duplicates(subset=["time"], keep="first")

    # 按时间排序
    df = df.sort_values("time").reset_index(drop=True)

    # 写CSV
    if output_csv:
        df.to_csv(output_csv, index=False, encoding="utf-8")
        print(f"[信息] Excel已转为CSV: {output_csv}")

    return df, wind_col


def merge_by_timeline(df1, df2, label1="CSV", label2="Excel"):
    """
    两个DataFrame按时间轴合并

    自动判断哪个文件是前半段，哪个是后半段。
    如果有重叠时间段，保留第二个文件(后半段)的数据。

    Parameters
    ----------
    df1 : pd.DataFrame
        第一个数据集
    df2 : pd.DataFrame
        第二个数据集
    label1 : str
        第一个数据集的标签(用于日志)
    label2 : str
        第二个数据集的标签(用于日志)

    Returns
    -------
    pd.DataFrame
        合并后的数据
    """
    t1_start = df1["time"].min()
    t1_end = df1["time"].max()
    t2_start = df2["time"].min()
    t2_end = df2["time"].max()

    print(f"\n{'=' * 60}")
    print(f"时间范围对比:")
    print(f"  {label1}: {t1_start} ~ {t1_end} (共 {len(df1)} 行)")
    print(f"  {label2}: {t2_start} ~ {t2_end} (共 {len(df2)} 行)")
    print(f"{'=' * 60}")

    # 判断前后关系
    if t1_start <= t2_start:
        first_df, second_df = df1, df2
        first_label, second_label = label1, label2
    else:
        first_df, second_df = df2, df1
        first_label, second_label = label2, label1

    print(f"[信息] 前半段: {first_label} ({first_df['time'].min()} ~ {first_df['time'].max()})")
    print(f"[信息] 后半段: {second_label} ({second_df['time'].min()} ~ {second_df['time'].max()})")

    # 检查时间连续性
    gap = second_df["time"].min() - first_df["time"].max()
    if gap.total_seconds() > 0:
        print(f"[信息] 两段数据之间有 {gap} 的时间间隔")
    elif gap.total_seconds() < 0:
        overlap = -gap
        print(f"[警告] 两段数据有 {overlap} 的时间重叠(重叠部分保留后半段数据)")

    # 合并: 先拼接，后去重(保留后半段的重复时间点)
    merged = pd.concat([first_df, second_df], ignore_index=True)
    merged = merged.sort_values("time").reset_index(drop=True)

    # 去重: 保留最后一条(即后半段的数据)
    dup_count = merged["time"].duplicated(keep="last").sum()
    if dup_count > 0:
        print(f"[信息] 合并后去重 {dup_count} 条重复时间(保留后半段数据)")
        merged = merged.drop_duplicates(subset=["time"], keep="last")
        merged = merged.sort_values("time").reset_index(drop=True)

    # 检查时间连续性(15分钟间隔)
    time_diffs = merged["time"].diff().dropna()
    expected_freq = pd.Timedelta(minutes=15)
    irregular = time_diffs[time_diffs != expected_freq]
    if len(irregular) > 0:
        print(f"[警告] 合并后有 {len(irregular)} 处时间间隔异常(非15分钟)")
        print(f"  示例: {irregular.iloc[0]} (位置: {irregular.index[0]})")
    else:
        print(f"[信息] 时间序列连续性检查通过(15分钟间隔)")

    print(f"[信息] 合并后总行数: {len(merged)}")
    print(f"[信息] 合并后时间范围: {merged['time'].min()} ~ {merged['time'].max()}")

    return merged


def save_merged_csv(df, output_path, wind_col):
    """
    保存合并后的CSV文件

    Parameters
    ----------
    df : pd.DataFrame
        合并后的数据
    output_path : str
        输出文件路径
    wind_col : str
        风列名(用于列顺序)
    """
    # 确保列顺序: time, power, wind_col
    cols = ["time", "power", wind_col]
    df = df[cols].copy()

    # 确保输出目录存在
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 格式化time列为字符串(避免pandas自动加.0后缀)
    df["time"] = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n[完成] 合并文件已保存: {output_path}")
    print(f"  列: {cols}")
    print(f"  行数: {len(df)}")

    # 文件大小
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  文件大小: {size_mb:.2f} MB")


def print_data_preview(df, title="数据预览"):
    """打印数据前5行和后5行"""
    print(f"\n--- {title} ---")
    print(f"前5行:")
    print(df.head().to_string(index=False))
    print(f"\n后5行:")
    print(df.tail().to_string(index=False))
    print(f"总行数: {len(df)}")


# ============================================================
# 交互模式
# ============================================================


def interactive_mode():
    """交互式输入文件路径"""
    print("\n" + "=" * 60)
    print("数据集合并工具 - 交互模式")
    print("=" * 60)

    # 输入CSV路径
    csv_file = input("\n请输入CSV数据集文件路径: ").strip().strip('"').strip("'")
    if not csv_file:
        print("[错误] 未输入CSV路径")
        return None

    # 输入Excel路径
    excel_file = input("请输入Excel文件路径(如power-ws.xlsx): ").strip().strip('"').strip("'")
    if not excel_file:
        print("[错误] 未输入Excel路径")
        return None

    # 输入输出路径
    default_output = os.path.join(
        os.path.dirname(csv_file) or ".",
        "merged_dataset.csv"
    )
    output_file = input(f"请输入输出文件路径(默认: {default_output}): ").strip().strip('"').strip("'")
    if not output_file:
        output_file = default_output

    # 选择风列
    wind_choice = input("风列名(ws/sr, 留空自动检测): ").strip()
    wind_col = wind_choice if wind_choice else None

    # CSV时间筛选(可选)
    print("\n[可选] 对CSV数据集按时间筛选(留空则不过滤):")
    time_start = input("  起始时间(含, 如 2024-07-03 或 202407030000): ").strip()
    time_start = time_start if time_start else None
    time_end = input("  结束时间(含, 如 2026-07-31 或 202607312345): ").strip()
    time_end = time_end if time_end else None

    return {
        "csv_file": csv_file,
        "excel_file": excel_file,
        "output": output_file,
        "wind_col": wind_col,
        "time_start": time_start,
        "time_end": time_end,
    }


# ============================================================
# 主函数
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="数据集合并工具: CSV筛选 + Excel转CSV + 时间轴合并",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动检测风列, 合并CSV和Excel
  python merge_datasets.py --csv-file data.csv --excel-file power-ws.xlsx --output merged.csv

  # 指定风列名
  python merge_datasets.py --csv-file data.csv --excel-file power-ws.xlsx --wind-col ws

  # 仅Excel转CSV
  python merge_datasets.py --excel-file power-ws.xlsx --excel-only --output converted.csv

  # 交互模式
  python merge_datasets.py
        """,
    )

    parser.add_argument("--csv-file", default=None, help="CSV数据集文件路径")
    parser.add_argument("--excel-file", default=None, help="Excel文件路径(.xlsx/.xls)")
    parser.add_argument("--output", default=None, help="输出合并后的CSV文件路径")
    parser.add_argument("--wind-col", default=None, help="风列名(ws/sr, 留空自动检测)")
    parser.add_argument("--time-start", default=None, help="CSV数据集时间筛选起始时间(含), 如 2024-07-03 或 202407030000")
    parser.add_argument("--time-end", default=None, help="CSV数据集时间筛选结束时间(含), 如 2026-07-31 或 202607312345")
    parser.add_argument("--excel-only", action="store_true", help="仅将Excel转为CSV,不做合并")
    parser.add_argument("--preview", action="store_true", help="显示处理前后数据预览")
    parser.add_argument("-i", "--interactive", action="store_true", help="交互模式")

    args = parser.parse_args()

    # 交互模式
    if args.interactive or (not args.csv_file and not args.excel_file):
        params = interactive_mode()
        if params is None:
            return
        args.csv_file = params["csv_file"]
        args.excel_file = params["excel_file"]
        args.output = params["output"]
        args.wind_col = params["wind_col"]
        args.time_start = params.get("time_start")
        args.time_end = params.get("time_end")

    # 验证参数
    if not args.excel_file:
        print("[错误] 必须指定 --excel-file 参数")
        parser.print_help()
        sys.exit(1)

    # 仅Excel转CSV模式
    if args.excel_only:
        if not args.output:
            base = os.path.splitext(args.excel_file)[0]
            args.output = base + ".csv"
        print("\n--- Excel转CSV模式 ---")
        df, wind_col = excel_to_csv(args.excel_file, args.output, args.wind_col)
        if args.preview:
            print_data_preview(df, "转换结果预览")
        print(f"\n[完成] 转换完成: {args.output}")
        return

    # 合并模式
    if not args.csv_file:
        print("[错误] 合并模式需要同时指定 --csv-file 和 --excel-file")
        parser.print_help()
        sys.exit(1)

    if not args.output:
        base = os.path.splitext(os.path.basename(args.csv_file))[0]
        args.output = os.path.join(
            os.path.dirname(args.csv_file) or ".",
            f"{base}_merged.csv"
        )

    print("\n" + "=" * 60)
    print("数据集合并工具")
    print("=" * 60)

    try:
        # 步骤1: 读取并筛选CSV
        print(f"\n--- 步骤1: 读取CSV数据集 ---")
        print(f"  文件: {args.csv_file}")
        if args.time_start or args.time_end:
            print(f"  时间筛选: [{args.time_start or '不限'} ~ {args.time_end or '不限'}]")
        df_csv, wind_col_csv = load_and_filter_csv(
            args.csv_file, args.wind_col, args.time_start, args.time_end
        )
        print(f"  筛选列: time, power, {wind_col_csv}")
        print(f"  行数: {len(df_csv)}")
        print(f"  时间范围: {get_time_range_string(df_csv['time'])}")
        if args.preview:
            print_data_preview(df_csv, "CSV数据预览")

        # 步骤2: Excel转CSV
        print(f"\n--- 步骤2: 读取Excel文件 ---")
        print(f"  文件: {args.excel_file}")
        df_excel, wind_col_excel = excel_to_csv(args.excel_file, None, args.wind_col)
        print(f"  筛选列: time, power, {wind_col_excel}")
        print(f"  行数: {len(df_excel)}")
        print(f"  时间范围: {get_time_range_string(df_excel['time'])}")
        if args.preview:
            print_data_preview(df_excel, "Excel数据预览")

        # 统一风列名
        wind_col = args.wind_col or wind_col_csv
        if wind_col_csv != wind_col_excel:
            print(f"\n[警告] CSV风列名为 '{wind_col_csv}', Excel风列名为 '{wind_col_excel}'")
            print(f"  使用 '{wind_col}' 作为统一列名")
            # 重命名列
            if wind_col_csv != wind_col:
                df_csv = df_csv.rename(columns={wind_col_csv: wind_col})
            if wind_col_excel != wind_col:
                df_excel = df_excel.rename(columns={wind_col_excel: wind_col})

        # 步骤3: 按时间轴合并
        print(f"\n--- 步骤3: 按时间轴合并 ---")
        merged_df = merge_by_timeline(df_csv, df_excel, "CSV", "Excel")

        if args.preview:
            print_data_preview(merged_df, "合并结果预览")

        # 步骤4: 保存
        print(f"\n--- 步骤4: 保存结果 ---")
        save_merged_csv(merged_df, args.output, wind_col)

        # 统计信息
        print(f"\n{'=' * 60}")
        print(f"处理完成!")
        print(f"{'=' * 60}")
        print(f"  CSV行数:    {len(df_csv)}")
        print(f"  Excel行数: {len(df_excel)}")
        print(f"  合并行数:   {len(merged_df)}")
        print(f"  输出文件:   {args.output}")
        print(f"  风列名:     {wind_col}")
        print(f"  时间范围:   {merged_df['time'].min()} ~ {merged_df['time'].max()}")
        print(f"{'=' * 60}")

    except FileNotFoundError as e:
        print(f"\n[错误] 文件不存在: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n[错误] 数据格式错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[错误] 处理失败: {type(e).__name__}: {e}")
        import traceback
        print(f"\n详细错误信息:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
