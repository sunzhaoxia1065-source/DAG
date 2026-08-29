#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集时间连续性检查工具

检查数据集是否满足业务评测对时间序列的 4 项要求:
  1. 索引必须是 DatetimeIndex
  2. 时间戳必须升序
  3. 时间戳必须无重复
  4. 所有相邻时间戳之差必须严格等于 15 分钟 (重点: 输出具体哪些位置不满足)

用法:
  python check_dataset_time.py                         # 交互式输入路径
  python check_dataset_time.py --input 数据集.csv
  python check_dataset_time.py -i 数据集.csv --time-format "%%Y%%m%%d%%H%%M"
"""

import argparse
import sys
import pandas as pd

KNOWN_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y%m%d%H%M%S",   # 14位
    "%Y%m%d%H%M",     # 12位
    "%Y-%m-%d",
    "%Y%m%d",
]


def parse_time(series, time_format=None):
    """解析时间列; 优先用户指定 format, 否则自动识别 (含纯数字串防误判为纳秒)"""
    s = series.astype(str).str.strip()
    if time_format:
        return pd.to_datetime(s, format=time_format, errors="coerce")
    # 全是纯数字: 按长度选 format, 避免 to_datetime 把 202604010000 误判为纳秒时间戳
    if s.str.isdigit().all():
        lens = set(s.str.len())
        if 14 in lens:
            return pd.to_datetime(s, format="%Y%m%d%H%M%S", errors="coerce")
        if 12 in lens:
            return pd.to_datetime(s, format="%Y%m%d%H%M", errors="coerce")
        if 8 in lens:
            return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    for fmt in KNOWN_FORMATS:
        res = pd.to_datetime(s, format=fmt, errors="coerce")
        if res.notna().mean() > 0.5:
            return res
    return pd.to_datetime(s, errors="coerce")


def check(csv_path, time_col="time", time_format=None):
    print("=" * 70)
    print(f"检查文件: {csv_path}")
    print("=" * 70)

    df = None
    for enc in ["utf-8", "utf-8-sig", "gbk", "latin1"]:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            print(f"读取成功 (编码 {enc}): {len(df)} 行, {len(df.columns)} 列")
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        print("[错误] 无法解码文件, 尝试了多种编码")
        return False

    if time_col not in df.columns:
        for cand in ["time", "datetime", "date", "timestamp"]:
            if cand in df.columns:
                time_col = cand
                break
        else:
            print(f"[错误] 找不到时间列 (尝试 'time'), 可用列: {list(df.columns)[:10]}")
            return False
    print(f"时间列: '{time_col}'")
    print(f"前5值: {df[time_col].head().tolist()}")

    ts = parse_time(df[time_col], time_format)
    nat = int(ts.isna().sum())
    if nat:
        print(f"[警告] {nat} 行时间解析失败 (NaT), 已丢弃")
    valid = ts.dropna()
    if valid.empty:
        print("[错误] 无有效时间戳, 无法检查")
        return False

    idx = pd.DatetimeIndex(valid)
    s = pd.Series(idx)

    results = {}

    # 1. DatetimeIndex
    print("\n--- 1. 索引必须是 DatetimeIndex ---")
    c1 = isinstance(idx, pd.DatetimeIndex)
    print(f"  [{'OK' if c1 else 'FAIL'}] 有效时间戳 {len(idx)} 个")
    results["c1"] = c1

    # 2. 升序
    print("\n--- 2. 时间戳必须升序 ---")
    c2 = idx.is_monotonic_increasing
    if c2:
        print(f"  [OK] 已升序")
    else:
        print(f"  [FAIL] 存在乱序时间戳")
        d = s.diff()
        desc = s[d.dt.total_seconds() < 0]
        print(f"  下降点(前20): {list(desc.head(20))}")
    results["c2"] = c2

    # 3. 无重复
    print("\n--- 3. 时间戳必须无重复 ---")
    dup_mask = idx.duplicated(keep=False)
    n_dup = int(dup_mask.sum())
    c3 = (n_dup == 0)
    if c3:
        print(f"  [OK] 无重复")
    else:
        print(f"  [FAIL] 有 {n_dup} 个重复时间戳")
        print(f"  重复值(前20): {list(pd.Series(idx[dup_mask]).drop_duplicates().head(20))}")
    results["c3"] = c3

    # 4. 15min 间隔 (重点)
    print("\n--- 4. 相邻时间戳之差必须严格 = 15分钟 (重点) ---")
    d = s.diff()
    non15 = d[(d != pd.Timedelta(minutes=15)) & d.notna()]
    c4 = (len(non15) == 0)
    n_total = int(d.notna().sum())
    if c4:
        print(f"  [OK] 全部 {n_total} 个相邻间隔均为 15 分钟")
    else:
        print(f"  [FAIL] {len(non15)} 处间隔 != 15分钟 (共 {n_total} 个间隔)")
        print(f"  间隔分布 (前10):")
        for v, c in d.value_counts().head(10).items():
            tag = "  <- 异常" if v != pd.Timedelta(minutes=15) else ""
            print(f"    {v} : {c} 次{tag}")
        print(f"\n  具体异常位置 (前50, 格式: 前一时刻 -> 当前时刻 = 间隔):")
        cnt = 0
        for j in non15.index:
            prev = s.iloc[j - 1]
            cur = s.iloc[j]
            print(f"    #{j-1}->#{j}: {prev} -> {cur} = {d.iloc[j]}")
            cnt += 1
            if cnt >= 50:
                print(f"    ... 共 {len(non15)} 处, 仅显示前50")
                break
    results["c4"] = c4

    # 附加: 缺失时间点
    print("\n--- 附加: 相对完整 15min 网格的缺失时间点 ---")
    full = pd.date_range(idx.min(), idx.max(), freq="15min")
    missing = full.difference(idx)
    print(f"  时间范围: {idx.min()} ~ {idx.max()}")
    print(f"  完整网格应有 {len(full)} 个点; 实际 {len(idx)} 个点; 缺失 {len(missing)} 个点")
    if len(missing) > 0:
        print(f"  缺失时间点 (前30):")
        for m in missing[:30]:
            print(f"    {m}")
        if len(missing) > 30:
            print(f"    ... 共 {len(missing)} 个")
        miss_dates = pd.Series([t.date() for t in missing])
        print(f"  缺失最多的日期 (前10):")
        for dt, c in miss_dates.value_counts().head(10).items():
            print(f"    {dt}: 缺 {c} 个点")

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print(f"  1. DatetimeIndex : {'OK' if results['c1'] else 'FAIL'}")
    print(f"  2. 升序         : {'OK' if results['c2'] else 'FAIL'}")
    print(f"  3. 无重复       : {'OK' if results['c3'] else 'FAIL'}")
    print(f"  4. 15min连续    : {'OK' if results['c4'] else 'FAIL'}")
    ok = all(results.values())
    print(f"\n  {'[通过] 满足业务评测要求' if ok else '[不通过] 需修复后才能用于业务评测'}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="数据集时间连续性检查工具")
    parser.add_argument("-i", "--input", default=None, help="数据集 CSV 文件路径")
    parser.add_argument("--time-col", default="time", help="时间列名 (默认 time)")
    parser.add_argument("--time-format", default=None, help="时间格式, 如 %%Y%%m%%d%%H%%M; 留空自动识别")
    args = parser.parse_args()

    csv_path = args.input
    if not csv_path:
        csv_path = input("请输入数据集文件路径: ").strip().strip('"').strip("'")
    if not csv_path:
        print("[错误] 未输入路径")
        sys.exit(1)
    check(csv_path, args.time_col, args.time_format)


if __name__ == "__main__":
    main()
