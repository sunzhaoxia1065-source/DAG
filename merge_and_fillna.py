# -*- coding: utf-8 -*-
"""
CSV 文件合并与缺失值处理工具

功能:
  1. 文件合并: 以 time 为键, 将基础文件与合并文件 left-join
     - 基础文件 (power-ws.csv): time, power, ws (3列)
       time 格式: YYYY-MM-DD HH:MM:SS 或 YYYYMMDDHHmm 等多种格式 (自动识别)
     - 合并文件 (merged_41_400_114_900.csv): time, 345列衍生特征
       time 格式: YYYYMMDDHHmm (12位数字)
     - 合并时自动统一 time 格式 (解析为 Timestamp), 合并结果只保留一个 time 列
  2. 缺失值处理: 提供两种方法
     a) fill_periodic_mean: 周期插补, 按天为周期(96点), 用前后N天同一时刻均值填充
     b) interpolate: 线性插值 (pandas.Series.interpolate, method='linear')
  3. 统计报告: 处理前后的缺失值数量、比例对比
  4. 数据预览: 展示处理前后的部分数据

命令行用法:
  # 默认参数 (周期插补, 邻居天数=3)
  python merge_and_fillna.py

  # 指定文件和缺失值处理方法
  python merge_and_fillna.py --base "路径1" --merge "路径2" --method interpolate --output "输出路径"

  # 周期插补, 前后各7天
  python merge_and_fillna.py --method fill_periodic_mean --periodic-neighbors 7

  # 查看帮助
  python merge_and_fillna.py --help
"""

import os
import sys
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd

# ======================== 默认配置 ========================
DEFAULT_BASE_FILE = r"F:\华为科研\DAG(start)\玖天数据集处理\郝家营风电场二期power-ws.csv"
DEFAULT_MERGE_FILE = r"F:\华为科研\DAG(start)\玖天数据集处理\haojiaying_merged\merged_41_400_114_900.csv"
DEFAULT_OUTPUT_FILE = r"F:\华为科研\DAG(start)\玖天数据集处理\haojiaying_final.csv"

# 时间列名
TIME_COL = "time"
# 合并文件的 time 格式: YYYYMMDDHHmm (12位数字, 例如 202407030000)
MERGE_TIME_FORMAT = "%Y%m%d%H%M"
# 基础文件的 time 格式: YYYY-MM-DD HH:MM:SS (例如 2026-04-01 00:00:00)
BASE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
# 输出文件的 time 格式: 与合并文件一致, YYYYMMDDHHmm
OUTPUT_TIME_FORMAT = MERGE_TIME_FORMAT
# 一天的时间点数 (15分钟分辨率, 24*4=96)
POINTS_PER_DAY = 96

# 默认周期插补的前后邻居天数
DEFAULT_PERIODIC_NEIGHBORS = 3

# 日志文件
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "merge_and_fillna.log")
# ========================================================


def setup_logger(log_file=LOG_FILE):
    """配置日志: 同时输出到控制台和文件"""
    logger = logging.getLogger("merge_and_fillna")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    try:
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception as e:
        logger.warning(f"无法创建日志文件 {log_file}: {e}, 仅输出到控制台")
    return logger


def parse_time(time_val):
    """
    解析 time 值为 pandas.Timestamp.
    支持多种格式:
      - 字符串 "YYYYMMDDHHmm" (12位数字, 例如 202407030000) -> 合并文件格式
      - 字符串 "YYYY-MM-DD HH:MM:SS" (例如 2026-04-01 00:00:00) -> 基础文件格式
      - pandas.Timestamp / datetime 对象 (直接返回)
    返回: Timestamp 或 NaT (解析失败时)
    """
    # 如果已经是 Timestamp 或 datetime, 直接转换
    if isinstance(time_val, (pd.Timestamp, datetime)):
        return pd.Timestamp(time_val)
    s = str(time_val).strip()
    # 尝试合并文件格式: YYYYMMDDHHmm (纯数字, 12位)
    if s.isdigit() and len(s) == 12:
        try:
            return pd.to_datetime(s, format=MERGE_TIME_FORMAT)
        except ValueError:
            pass
    # 尝试基础文件格式: YYYY-MM-DD HH:MM:SS
    try:
        return pd.to_datetime(s, format=BASE_TIME_FORMAT)
    except ValueError:
        pass
    # 兜底: 让 pandas 自动推断
    try:
        return pd.to_datetime(s)
    except (ValueError, TypeError):
        return pd.NaT


def load_csv(filepath, label, logger):
    """
    加载 CSV 文件, 处理 time 列.
    采用分块读取策略, 避免大文件内存溢出.
    返回: DataFrame (time 已转为字符串去重排序, 另存 timestamp 列辅助对齐)
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"{label}文件不存在: {filepath}")

    # 先读取表头, 确定列类型, 减少内存占用
    logger.info(f"读取{label}表头: {filepath}")
    try:
        header_df = pd.read_csv(filepath, encoding="utf-8", nrows=0)
        encoding_used = "utf-8"
    except UnicodeDecodeError:
        logger.warning(f"{label} UTF-8 解码失败, 尝试 GBK 编码")
        header_df = pd.read_csv(filepath, encoding="gbk", nrows=0)
        encoding_used = "gbk"

    all_columns = list(header_df.columns)
    if TIME_COL not in all_columns:
        raise ValueError(f"{label}缺少 '{TIME_COL}' 列, 实际列前5: {all_columns[:5]}")

    # 构造 dtype: 字符串列用 string, 数值列用 float32 (节省内存)
    # batch, point_id, time 是字符串列; 其余默认 float32
    str_cols = {"batch", "point_id", TIME_COL}
    dtype_map = {}
    for col in all_columns:
        if col in str_cols:
            dtype_map[col] = "string"
        else:
            dtype_map[col] = "float32"

    # 分块读取大文件, 每块 50 万行
    chunksize = 500000
    logger.info(
        f"分块读取{label}: 每块 {chunksize} 行, dtype已优化 "
        f"(数值列 float32, 字符串列 string)"
    )

    chunks = []
    total_rows = 0
    try:
        reader = pd.read_csv(
            filepath,
            encoding=encoding_used,
            dtype=dtype_map,
            chunksize=chunksize,
            low_memory=True,
        )
        for i, chunk in enumerate(reader):
            # 处理 time 列: 转字符串去空白
            chunk[TIME_COL] = chunk[TIME_COL].astype(str).str.strip()
            chunks.append(chunk)
            total_rows += len(chunk)
            if (i + 1) % 5 == 0:
                logger.info(f"  已读取 {total_rows} 行 ({i+1} 块)")
    except Exception as e:
        logger.error(f"分块读取{label}失败: {type(e).__name__}: {e}")
        raise

    logger.info(f"  读取完成, 共 {total_rows} 行, 正在合并块...")
    df = pd.concat(chunks, ignore_index=True)
    # 释放块列表引用
    del chunks

    # 解析 timestamp 用于合并和周期插补 (不输出到最终CSV)
    df["_timestamp"] = df[TIME_COL].apply(parse_time)
    nat_count = df["_timestamp"].isna().sum()
    if nat_count > 0:
        logger.warning(f"{label}有 {nat_count} 条 time 解析失败 (NaT)")

    # 去重: 以 _timestamp 为准 (处理可能的 NaT)
    dup = df["_timestamp"].duplicated().sum()
    if dup > 0:
        logger.warning(f"{label}有 {dup} 条重复 time, 保留第一条")
        df = df.drop_duplicates(subset="_timestamp", keep="first")
    # 按 _timestamp 排序 (NaT 排到最后)
    df = df.sort_values("_timestamp", na_position="last").reset_index(drop=True)

    logger.info(
        f"{label}加载完成: 行数={len(df)}, 列数={len(df.columns) - 1} (排除辅助_timestamp列), "
        f"time范围={df[TIME_COL].iloc[0]}~{df[TIME_COL].iloc[-1]}"
    )
    return df


def load_file(filepath, label, logger):
    """
    加载 CSV 文件 (基础文件和合并文件均使用此函数).
    采用分块读取策略, 避免大文件内存溢出.
    返回: DataFrame (time 已解析为 _timestamp 辅助列, 去重排序)
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext and ext != ".csv":
        logger.warning(f"文件扩展名 {ext} 非标准 CSV, 尝试按 CSV 读取")
    return load_csv(filepath, label, logger)


def merge_by_time(base_df, merge_df, logger):
    """
    以 _timestamp (统一解析后的 Timestamp) 为键, left-join 合并两个 DataFrame.
    基础文件的所有行都保留, 合并文件中匹配不上的行丢弃.
    匹配不上的列填 NaN.
    合并后保留基础文件的 time 列 (原始格式), 移除合并文件的 time 列.
    """
    logger.info("-" * 60)
    logger.info("开始合并文件 (以 _timestamp 为键, 解决两文件 time 格式差异)")
    logger.info(f"  基础文件: {len(base_df)} 行, {len(base_df.columns) - 1} 列")
    logger.info(f"  合并文件: {len(merge_df)} 行, {len(merge_df.columns) - 1} 列")

    # 校验重名列 (除 time 和 _timestamp 外)
    base_cols = set(base_df.columns) - {TIME_COL, "_timestamp"}
    merge_cols = set(merge_df.columns) - {TIME_COL, "_timestamp"}
    overlap = base_cols & merge_cols
    if overlap:
        logger.warning(
            f"两文件有 {len(overlap)} 个重名列 (将添加 _merge 后缀区分): {list(overlap)[:5]}..."
        )
        rename_map = {c: f"{c}_merge" for c in overlap}
        merge_df = merge_df.rename(columns=rename_map)

    # 合并文件去掉原始 time 列 (保留 _timestamp 用于合并), 避免重名
    # 合并后以基础文件的 time 列为准
    merge_df_for_join = merge_df.drop(columns=[TIME_COL], errors="ignore")

    before_cols = len(base_df.columns)
    # 以 _timestamp 为键 left-join
    merged = base_df.merge(merge_df_for_join, on="_timestamp", how="left")
    after_cols = len(merged.columns)

    logger.info(
        f"  合并完成: 行数={len(merged)}, 列数={after_cols} (新增 {after_cols - before_cols} 列)"
    )

    # 检查合并文件未覆盖的行
    new_cols = [c for c in merged.columns if c not in base_df.columns]
    if new_cols:
        first_new_col = new_cols[0]
        uncovered = merged[first_new_col].isna().sum()
        if uncovered > 0:
            logger.warning(
                f"  基础文件有 {uncovered} 行 ({uncovered/len(merged)*100:.1f}%) 在合并文件中无对应数据, "
                f"这些行的衍生特征列将填 NaN"
            )

    return merged


def detect_missing(df, logger, label=""):
    """
    检测缺失值, 返回统计 DataFrame.
    返回: DataFrame[列名, 缺失数, 缺失比例]
    """
    cols = [c for c in df.columns if c != TIME_COL and c != "_timestamp"]
    total = len(df)
    stats = []
    for col in cols:
        n_miss = df[col].isna().sum()
        ratio = n_miss / total * 100 if total > 0 else 0
        stats.append({"列名": col, "缺失数": n_miss, "缺失比例(%)": round(ratio, 4)})
    stats_df = pd.DataFrame(stats)

    total_miss = stats_df["缺失数"].sum()
    cols_with_miss = (stats_df["缺失数"] > 0).sum()
    logger.info(f"  {label}缺失值统计: 总缺失={total_miss}, 有缺失的列数={cols_with_miss}/{len(cols)}")
    return stats_df


def fill_periodic_mean(df, col, neighbors=3, logger=None):
    """
    周期插补: 按天为周期 (96点/天), 用前后 N 天同一时刻的均值填充缺失值.
    原理: 气象数据有强日周期性, 同一时刻前后几天的值接近.

    算法:
      对于缺失位置 (day_i, slot_j):
        收集 [day_{i-N}, day_{i+N}] 范围内 slot_j 的非NaN值
        用这些值的均值填充

    neighbors: 前后各多少天 (总窗口 = 2*N+1 天)
    """
    # 强制转为数值类型, 解决 merge 后可能变为 object/string 的问题
    s = pd.to_numeric(df[col], errors="coerce")
    ts = df["_timestamp"]

    # 构造日期和时刻槽 (0~95)
    # 只对有效 timestamp 操作
    valid_mask = ts.notna()
    if valid_mask.sum() == 0:
        return s

    dates = ts.dt.date
    # 时刻槽 = 小时*4 + 分钟//15
    slots = ts.dt.hour * 4 + ts.dt.minute // 15

    # 按 (date, slot) 建立索引, 方便查询
    # 同一天同一 slot 应只有一条 (已去重), 这里取第一条
    date_slot_to_idx = {}
    for idx in df.index[valid_mask]:
        key = (dates.iloc[idx], slots.iloc[idx])
        if key not in date_slot_to_idx:
            date_slot_to_idx[key] = idx

    # 收集所有出现过的日期, 排序
    all_dates = sorted(set(dates.dropna()))

    # 对每个缺失位置, 用前后N天同一slot的均值填充
    missing_mask = s.isna() & valid_mask
    missing_idx = s.index[missing_mask]
    total_missing = len(missing_idx)

    # 诊断: 检查该列是否有任何非NaN值
    non_na_count = s.notna().sum()
    if logger and total_missing > 0:
        logger.info(
            f"    列 {col}: 缺失={total_missing}, 非空={non_na_count}, "
            f"dtype={s.dtype}, 邻居窗口={2*neighbors+1}天"
        )

    filled_count = 0
    for idx in missing_idx:
        cur_date = dates[idx]
        cur_slot = slots[idx]
        if pd.isna(cur_date):
            continue

        # 找到当前日期在 all_dates 中的位置
        try:
            date_pos = all_dates.index(cur_date)
        except ValueError:
            continue

        # 收集前后 N 天同一 slot 的值
        start_pos = max(0, date_pos - neighbors)
        end_pos = min(len(all_dates), date_pos + neighbors + 1)
        neighbor_dates = all_dates[start_pos:end_pos]

        vals = []
        for nd in neighbor_dates:
            if nd == cur_date:
                continue  # 跳过自己 (虽然是NaN)
            key = (nd, cur_slot)
            if key in date_slot_to_idx:
                v = s.iloc[date_slot_to_idx[key]]
                if pd.notna(v):
                    vals.append(v)

        if vals:
            s.iloc[idx] = np.mean(vals)
            filled_count += 1

    if logger and total_missing > 0:
        logger.info(f"    列 {col}: 周期插补成功填充 {filled_count}/{total_missing} 个缺失值")
    return s


def fillna_periodic_all(df, neighbors=3, logger=None):
    """对所有数据列应用周期插补"""
    cols = [c for c in df.columns if c != TIME_COL and c != "_timestamp"]
    if logger:
        logger.info(f"  应用周期插补: 前后各 {neighbors} 天, 共 {len(cols)} 列")
    for i, col in enumerate(cols, 1):
        n_miss = pd.to_numeric(df[col], errors="coerce").isna().sum()
        if n_miss > 0:
            filled_series = fill_periodic_mean(df, col, neighbors=neighbors, logger=logger)
            df[col] = filled_series.astype("float32")
    return df


def fillna_interpolate_all(df, logger=None):
    """对所有数据列应用线性插值"""
    cols = [c for c in df.columns if c != TIME_COL and c != "_timestamp"]
    if logger:
        logger.info(f"  应用线性插值: 共 {len(cols)} 列")
    for col in cols:
        if df[col].isna().any():
            # 尝试数值转换 (非数值列跳过)
            try:
                numeric_series = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                continue
            before = df[col].isna().sum()
            # 线性插值, limit_direction='both' 填充首尾的 NaN
            filled = numeric_series.interpolate(method="linear", limit_direction="both")
            df[col] = filled
            after = df[col].isna().sum()
            if logger and after < before:
                logger.info(f"    列 {col}: 线性插值填充 {before - after} 个缺失值")
    return df


def fill_missing(df, method, neighbors, logger):
    """根据方法选择缺失值填补"""
    if method == "fill_periodic_mean":
        df = fillna_periodic_all(df, neighbors=neighbors, logger=logger)
    elif method == "interpolate":
        df = fillna_interpolate_all(df, logger=logger)
    else:
        raise ValueError(f"不支持的缺失值处理方法: {method} (可选: fill_periodic_mean, interpolate)")

    # 对于仍无法填补的 NaN (如整列全空), 用列均值兜底
    cols = [c for c in df.columns if c != TIME_COL and c != "_timestamp"]
    for col in cols:
        numeric_series = pd.to_numeric(df[col], errors="coerce")
        n_miss = numeric_series.isna().sum()
        if n_miss > 0:
            col_mean = numeric_series.mean()
            if pd.notna(col_mean):
                df[col] = numeric_series.fillna(col_mean).astype("float32")
                logger.info(f"    列 {col}: 用列均值 {col_mean:.4f} 兜底填充剩余 {n_miss} 个 NaN")
            else:
                # 整列全空, 用0填充
                df[col] = numeric_series.fillna(0).astype("float32")
                logger.warning(f"    列 {col}: 整列无有效值, 用0填充 {n_miss} 个 NaN")
    return df


def preview_data(df, n=5, logger=None):
    """数据预览: 展示前 n 行的关键列"""
    if logger is None:
        return
    cols_to_show = [TIME_COL] + [c for c in df.columns if c != TIME_COL and c != "_timestamp"][:4]
    logger.info("  预览列: " + ", ".join(cols_to_show))
    preview = df[cols_to_show].head(n).to_string(index=False)
    for line in preview.split("\n"):
        logger.info(f"    {line}")


def main():
    parser = argparse.ArgumentParser(
        description="CSV 文件合并与缺失值处理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认参数 (周期插补, 邻居3天)
  python merge_and_fillna.py

  # 指定文件和缺失值处理方法
  python merge_and_fillna.py --base "F:\\base.csv" --merge "F:\\merge.csv" --method interpolate

  # 周期插补, 前后各7天
  python merge_and_fillna.py --method fill_periodic_mean --periodic-neighbors 7
        """,
    )
    parser.add_argument("--base", default=DEFAULT_BASE_FILE, help="基础文件路径 (CSV: time, power, ws 三列)")
    parser.add_argument("--merge", default=DEFAULT_MERGE_FILE, help="合并 CSV 文件路径")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="输出 CSV 文件路径")
    parser.add_argument(
        "--method",
        default="fill_periodic_mean",
        choices=["fill_periodic_mean", "interpolate"],
        help="缺失值处理方法: fill_periodic_mean (周期插补, 默认) 或 interpolate (线性插值)",
    )
    parser.add_argument(
        "--periodic-neighbors",
        type=int,
        default=DEFAULT_PERIODIC_NEIGHBORS,
        help=f"周期插补的前后邻居天数 (默认 {DEFAULT_PERIODIC_NEIGHBORS})",
    )
    parser.add_argument("--log", default=LOG_FILE, help="日志文件路径")
    args = parser.parse_args()

    logger = setup_logger(args.log)
    logger.info("=" * 70)
    logger.info("CSV 合并与缺失值处理任务启动")
    logger.info(f"基础文件     : {args.base}")
    logger.info(f"合并文件     : {args.merge}")
    logger.info(f"输出文件     : {args.output}")
    logger.info(f"缺失值方法   : {args.method}")
    if args.method == "fill_periodic_mean":
        logger.info(f"周期邻居天数 : {args.periodic_neighbors} (总窗口 {2*args.periodic_neighbors+1} 天)")
    logger.info("=" * 70)

    # ========== 1. 加载文件 ==========
    try:
        base_df = load_file(args.base, "基础文件", logger)
    except Exception as e:
        logger.error(f"基础文件加载失败: {type(e).__name__}: {e}")
        sys.exit(1)

    try:
        merge_df = load_file(args.merge, "合并文件", logger)
    except Exception as e:
        logger.error(f"合并文件加载失败: {type(e).__name__}: {e}")
        sys.exit(1)

    # ========== 2. 合并 ==========
    try:
        merged_df = merge_by_time(base_df, merge_df, logger)
    except Exception as e:
        logger.error(f"合并失败: {type(e).__name__}: {e}", exc_info=True)
        sys.exit(1)

    # ========== 3. 处理前缺失值检测 ==========
    logger.info("-" * 60)
    logger.info("处理前缺失值统计")
    before_stats = detect_missing(merged_df, logger, label="处理前")

    logger.info("处理前数据预览:")
    preview_data(merged_df, n=5, logger=logger)

    # ========== 4. 缺失值处理 ==========
    logger.info("-" * 60)
    logger.info(f"开始缺失值处理 (方法: {args.method})")
    filled_df = merged_df.copy()
    filled_df = fill_missing(filled_df, args.method, args.periodic_neighbors, logger)

    # ========== 5. 处理后缺失值检测 ==========
    logger.info("-" * 60)
    logger.info("处理后缺失值统计")
    after_stats = detect_missing(filled_df, logger, label="处理后")

    logger.info("处理后数据预览:")
    preview_data(filled_df, n=5, logger=logger)

    # ========== 6. 对比报告 ==========
    logger.info("-" * 60)
    logger.info("处理前后对比报告")
    report = pd.DataFrame({
        "列名": before_stats["列名"],
        "处理前缺失数": before_stats["缺失数"],
        "处理前缺失比例(%)": before_stats["缺失比例(%)"],
        "处理后缺失数": after_stats["缺失数"].values,
        "处理后缺失比例(%)": after_stats["缺失比例(%)"].values,
    })
    report["减少缺失数"] = report["处理前缺失数"] - report["处理后缺失数"]
    total_before = report["处理前缺失数"].sum()
    total_after = report["处理后缺失数"].sum()
    reduced = total_before - total_after
    logger.info(f"  总缺失值: 处理前 {total_before} -> 处理后 {total_after} (减少 {reduced})")
    if total_before > 0:
        logger.info(f"  填补率: {reduced/total_before*100:.2f}%")

    # 保存报告
    report_path = os.path.splitext(args.output)[0] + "_missing_report.csv"
    try:
        report.to_csv(report_path, index=False, encoding="utf-8-sig")
        logger.info(f"  缺失值报告已保存: {report_path}")
    except Exception as e:
        logger.warning(f"  保存报告失败: {e}")

    # ========== 7. 保存结果 ==========
    logger.info("-" * 60)
    logger.info(f"保存处理结果: {args.output}")
    # 将 time 列统一格式化为 YYYYMMDDHHmm (与合并文件一致, 便于后续模型使用)
    if "_timestamp" in filled_df.columns:
        # 用 _timestamp 重新格式化 time 列, 确保格式统一
        filled_df[TIME_COL] = filled_df["_timestamp"].dt.strftime(OUTPUT_TIME_FORMAT)
        # 移除辅助列 _timestamp
        filled_df = filled_df.drop(columns=["_timestamp"])
    try:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    except OSError:
        pass
    try:
        filled_df.to_csv(args.output, index=False, encoding="utf-8")
        logger.info(
            f"保存成功: 行数={len(filled_df)}, 列数={len(filled_df.columns)}, "
            f"time格式={OUTPUT_TIME_FORMAT}"
        )
    except Exception as e:
        logger.error(f"保存失败: {type(e).__name__}: {e}")
        sys.exit(1)

    # ========== 8. 关键参数记录 ==========
    logger.info("-" * 60)
    logger.info("关键参数和处理步骤记录 (可复现):")
    logger.info(f"  base_file          : {args.base}")
    logger.info(f"  merge_file         : {args.merge}")
    logger.info(f"  output_file        : {args.output}")
    logger.info(f"  method             : {args.method}")
    logger.info(f"  periodic_neighbors : {args.periodic_neighbors}")
    logger.info(f"  points_per_day     : {POINTS_PER_DAY} (15分钟分辨率)")
    logger.info(f"  merge_strategy     : left-join on '_timestamp' (统一时间格式)")
    logger.info(f"  base_time_format   : {BASE_TIME_FORMAT}")
    logger.info(f"  merge_time_format  : {MERGE_TIME_FORMAT}")
    logger.info(f"  output_time_format : {OUTPUT_TIME_FORMAT}")
    logger.info(f"  merge_time         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    logger.info("任务结束")


if __name__ == "__main__":
    main()
