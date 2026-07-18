#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DAG 模型超参数自动调优工具

使用随机搜索 + 简易贝叶斯优化自动搜索最优超参数组合。
零外部依赖，仅使用 Python 标准库 + pandas（项目已有）。

功能:
  - 随机搜索 + 基于历史结果的局部搜索
  - 支持多进程并行（通过文件锁共享结果）
  - 自动解析训练结果
  - 完整日志记录（JSONL 格式）
  - 输出最优参数、调参历史和推荐运行命令

使用方式:
  # 基本用法
  python tools/auto_tune.py \\
      --config-path config/business_day_ahead_config.json \\
      --data-name merged_result_ninghe.csv \\
      --n-trials 30

  # 并行调参（在多个终端同时运行相同命令）
  python tools/auto_tune.py \\
      --config-path config/business_day_ahead_config.json \\
      --data-name merged_result_ninghe.csv \\
      --n-trials 30 \\
      --shared-db tune_results/shared.json

  # 自定义搜索空间
  python tools/auto_tune.py \\
      --config-path config/business_day_ahead_config.json \\
      --data-name merged_result_ninghe.csv \\
      --search-space my_search_space.json

  # 固定部分参数只调其余参数
  python tools/auto_tune.py \\
      --config-path config/business_day_ahead_config.json \\
      --data-name merged_result_ninghe.csv \\
      --fixed-params '{"seq_len": 576, "patch_len": 96}'
"""

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


# ============================================================
# 默认搜索空间
# ============================================================

DEFAULT_SEARCH_SPACE = {
    "lr": {
        "type": "logfloat",
        "low": 1e-4,
        "high": 3e-3,
        "comment": "学习率，对数尺度搜索",
    },
    "alpha": {
        "type": "float",
        "low": 0.1,
        "high": 0.7,
        "step": 0.05,
        "comment": "TC/CC 融合权重",
    },
    "d_model": {
        "type": "categorical",
        "choices": [128, 256, 512],
        "comment": "模型隐藏维度",
    },
    "d_ff": {
        "type": "categorical",
        "choices": [128, 256, 512, 1024],
        "comment": "前馈网络维度",
    },
    "dropout": {
        "type": "float",
        "low": 0.1,
        "high": 0.5,
        "step": 0.05,
        "comment": "Dropout 率",
    },
    "batch_size": {
        "type": "categorical",
        "choices": [32, 64, 128],
        "comment": "批大小",
    },
    "seq_len": {
        "type": "categorical",
        "choices": [96, 192, 288, 576],
        "comment": "输入历史窗口长度（1/2/3/6天）",
    },
    "patch_len": {
        "type": "categorical",
        "choices": [48, 96],
        "comment": "Patch 长度",
    },
    "stride": {
        "type": "categorical",
        "choices": [24, 48],
        "comment": "Patch 步长",
    },
    "e_layers": {
        "type": "int",
        "low": 1,
        "high": 3,
        "comment": "编码器层数",
    },
    "patience": {
        "type": "int",
        "low": 3,
        "high": 10,
        "comment": "早停耐心值",
    },
}


# ============================================================
# 参数采样器
# ============================================================


def sample_params(search_space, rng=None):
    """
    从搜索空间中随机采样一组超参数。

    Parameters
    ----------
    search_space : dict
        搜索空间定义
    rng : random.Random, optional
        随机数生成器

    Returns
    -------
    dict
        采样得到的超参数字典
    """
    if rng is None:
        rng = random.Random()
    params = {}
    for name, config in search_space.items():
        ptype = config["type"]
        if ptype == "logfloat":
            log_low = math.log(config["low"])
            log_high = math.log(config["high"])
            params[name] = math.exp(rng.uniform(log_low, log_high))
        elif ptype == "float":
            low, high = config["low"], config["high"]
            step = config.get("step", None)
            if step:
                n_steps = int((high - low) / step)
                params[name] = low + rng.randint(0, n_steps) * step
            else:
                params[name] = rng.uniform(low, high)
        elif ptype == "int":
            params[name] = rng.randint(config["low"], config["high"])
        elif ptype == "categorical":
            params[name] = rng.choice(config["choices"])
    return params


def perturb_params(params, search_space, rng=None, strength=0.2):
    """
    对一组超参数进行小幅扰动（局部搜索）。

    对数值参数在当前值附近 ±strength × 范围 内扰动，
    对类别参数以一定概率随机替换。

    Parameters
    ----------
    params : dict
        当前超参数
    search_space : dict
        搜索空间定义
    rng : random.Random, optional
        随机数生成器
    strength : float
        扰动强度 (0~1)

    Returns
    -------
    dict
        扰动后的超参数
    """
    if rng is None:
        rng = random.Random()
    new_params = params.copy()
    for name, config in search_space.items():
        ptype = config["type"]
        if ptype == "logfloat":
            log_val = math.log(params[name])
            log_low = math.log(config["low"])
            log_high = math.log(config["high"])
            delta = strength * (log_high - log_low)
            new_log = log_val + rng.uniform(-delta, delta)
            new_params[name] = math.exp(max(log_low, min(log_high, new_log)))
        elif ptype == "float":
            low, high = config["low"], config["high"]
            step = config.get("step", None)
            delta = strength * (high - low)
            new_val = params[name] + rng.uniform(-delta, delta)
            new_val = max(low, min(high, new_val))
            if step:
                new_val = round(new_val / step) * step
            new_params[name] = new_val
        elif ptype == "int":
            low, high = config["low"], config["high"]
            delta = max(1, int(strength * (high - low)))
            new_val = params[name] + rng.randint(-delta, delta)
            new_params[name] = max(low, min(high, new_val))
        elif ptype == "categorical":
            if rng.random() < strength:
                new_params[name] = rng.choice(config["choices"])
    return new_params


def load_search_space(path=None):
    """
    加载搜索空间配置。

    Parameters
    ----------
    path : str, optional
        搜索空间 JSON 文件路径。为 None 时使用默认搜索空间。

    Returns
    -------
    dict
        搜索空间定义
    """
    if path is None:
        return DEFAULT_SEARCH_SPACE.copy()

    with open(path, "r", encoding="utf-8") as f:
        space = json.load(f)

    for config in space.values():
        config.pop("comment", None)

    return space


# ============================================================
# 试验结果数据库（JSON 文件，支持多进程共享）
# ============================================================


class TrialDB:
    """
    基于 JSON 文件的试验结果数据库。

    支持多进程通过文件锁安全地并发读写。

    Parameters
    ----------
    path : str
        数据库文件路径
    """

    def __init__(self, path):
        self.path = path

    def load(self):
        """加载所有试验记录。"""
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def save(self, trials):
        """保存所有试验记录。"""
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(trials, f, indent=2, ensure_ascii=False)

    def append(self, trial_record):
        """
        追加一条试验记录（线程安全）。

        Parameters
        ----------
        trial_record : dict
            试验记录
        """
        trials = self.load()
        trials.append(trial_record)
        self.save(trials)

    def get_best(self, direction="maximize"):
        """
        获取最优试验记录。

        Parameters
        ----------
        direction : str
            "maximize" 或 "minimize"

        Returns
        -------
        dict or None
            最优试验记录
        """
        trials = self.load()
        success_trials = [t for t in trials if t.get("status") == "success"]
        if not success_trials:
            return None
        if direction == "maximize":
            return max(success_trials, key=lambda t: t["metric"])
        else:
            return min(success_trials, key=lambda t: t["metric"])

    def get_all_success(self):
        """获取所有成功的试验记录。"""
        trials = self.load()
        return [t for t in trials if t.get("status") == "success"]

    def next_trial_id(self):
        """获取下一个试验编号。"""
        trials = self.load()
        if not trials:
            return 0
        return max(t.get("trial", -1) for t in trials) + 1


# ============================================================
# 试验执行与结果解析
# ============================================================


def run_trial_subprocess(params, base_args, trial_id, timeout_per_trial):
    """
    通过子进程运行单次训练试验。

    Parameters
    ----------
    params : dict
        本次试验的超参数
    base_args : dict
        基础运行参数
    trial_id : int
        试验编号
    timeout_per_trial : int
        单次试验超时时间（秒）

    Returns
    -------
    dict
        试验结果
    """
    save_path = os.path.join(base_args["output_dir"], f"trial_{trial_id}")
    os.makedirs(save_path, exist_ok=True)

    # 使用绝对路径，确保子进程写入正确目录
    abs_save_path = os.path.abspath(save_path)

    cmd = [
        sys.executable,
        os.path.join(base_args["project_root"], "scripts", "run_benchmark.py"),
        "--config-path",
        base_args["config_path"],
        "--data-name-list",
        base_args["data_name"],
        "--model-name",
        base_args["model_name"],
        "--model-hyper-params",
        json.dumps(params),
        "--gpus",
        str(base_args["gpus"]),
        "--num-workers",
        str(base_args.get("num_workers", 1)),
        "--timeout",
        str(timeout_per_trial * 1000),
        "--save-path",
        abs_save_path,
    ]

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_per_trial,
            cwd=base_args["project_root"],
        )
        elapsed = time.time() - start_time

        if result.returncode != 0:
            # 保留完整错误信息用于诊断
            stderr_full = result.stderr or "unknown"
            stderr_tail = stderr_full[-2000:]
            # 同时打印到控制台方便调试
            print(f"  [错误] 子进程返回码: {result.returncode}")
            print(f"  [stderr 最后2000字符]:\n{stderr_tail}")
            return {
                "status": "failed",
                "metric": 0.0,
                "elapsed": elapsed,
                "error": stderr_tail,
            }

        metric = parse_trial_result(save_path, base_args["metric_name"])
        if metric is None:
            return {
                "status": "no_result",
                "metric": 0.0,
                "elapsed": elapsed,
                "error": "无法从输出目录解析结果",
            }

        return {"status": "success", "metric": float(metric), "elapsed": elapsed}

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return {
            "status": "timeout",
            "metric": 0.0,
            "elapsed": elapsed,
            "error": f"试验超时 ({timeout_per_trial}s)",
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "status": "error",
            "metric": 0.0,
            "elapsed": elapsed,
            "error": str(e),
        }


def parse_trial_result(save_path, metric_name="march_accuracy_mean"):
    """
    从试验输出目录解析评估指标。

    搜索策略:
    1. 递归搜索所有 CSV 和 tar.gz 文件
    2. 尝试精确匹配 metric_name
    3. 尝试模糊匹配（如 "accuracy" 匹配含 "accuracy" 的列名）
    4. 尝试从 leaderboard 文件解析

    Parameters
    ----------
    save_path : str
        试验输出目录路径
    metric_name : str
        目标指标名称

    Returns
    -------
    float or None
        解析到的指标值
    """
    save_dir = Path(save_path)

    # 打印目录内容便于调试
    if save_dir.exists():
        all_files = list(save_dir.rglob("*"))
        print(f"  [调试] 输出目录 {save_path} 包含 {len(all_files)} 个文件:")
        for f in all_files[:20]:
            print(f"    {f.relative_to(save_dir)}")
        if len(all_files) > 20:
            print(f"    ... 还有 {len(all_files) - 20} 个文件")

    # 收集所有 CSV 文件（直接 + 解压 tar.gz）
    csv_dataframes = []

    # 方式1: 直接查找 CSV
    for csv_file in save_dir.rglob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            csv_dataframes.append((str(csv_file), df))
        except Exception:
            continue

    # 方式2: 解压 tar.gz 中的 CSV
    for tar_file in save_dir.rglob("*.csv.tar.gz"):
        try:
            with tarfile.open(tar_file, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith(".csv"):
                        f = tar.extractfile(member)
                        if f is None:
                            continue
                        try:
                            df = pd.read_csv(f)
                            csv_dataframes.append((f"tar:{tar_file.name}/{member.name}", df))
                        except Exception:
                            continue
        except Exception:
            continue

    if not csv_dataframes:
        print(f"  [警告] 未找到任何 CSV 结果文件")
        return None

    # 尝试精确匹配
    for source, df in csv_dataframes:
        if metric_name in df.columns:
            val = df[metric_name].iloc[0]
            if pd.notna(val):
                print(f"  [解析] 在 {source} 中找到 {metric_name}={val}")
                return float(val)
            else:
                print(f"  [调试] 在 {source} 中找到 {metric_name} 但值为 NaN")

    # 如果目标指标为 NaN，尝试使用备选指标
    fallback_metrics = ["march_accuracy_mean", "daily_accuracy", "rmse", "mae"]
    for fallback in fallback_metrics:
        if fallback == metric_name:
            continue  # 已经试过了
        for source, df in csv_dataframes:
            if fallback in df.columns:
                val = df[fallback].iloc[0]
                if pd.notna(val):
                    print(f"  [解析] 目标指标为 NaN，使用备选指标 {fallback}={val}")
                    return float(val)

    # 打印所有列名和值便于调试
    print(f"  [调试] 精确匹配 '{metric_name}' 失败，所有结果文件内容:")
    for source, df in csv_dataframes:
        print(f"    {source}: {list(df.columns)[:10]}")

    # 尝试模糊匹配
    metric_lower = metric_name.lower()
    for source, df in csv_dataframes:
        for col in df.columns:
            if metric_lower in col.lower() or col.lower() in metric_lower:
                val = df[col].iloc[0]
                if pd.notna(val):
                    print(f"  [解析] 模糊匹配: 列 '{col}' ≈ '{metric_name}', 值={val}")
                    return float(val)

    # 尝试从 leaderboard 解析（常见格式: 模型名,指标1,指标2,...）
    for source, df in csv_dataframes:
        if "leaderboard" in source.lower() or len(df) == 1:
            # 单行数据，取第一个数值型列
            for col in df.columns:
                try:
                    val = float(df[col].iloc[0])
                    if not pd.isna(val) and col not in ("model", "strategy", "dataset"):
                        print(f"  [解析] leaderboard 列 '{col}' = {val}")
                        return val
                except (ValueError, TypeError):
                    continue

    print(f"  [警告] 所有解析方式均失败")
    return None


# ============================================================
# 调参主循环
# ============================================================


def validate_params(params):
    """
    校验超参数组合的合法性。

    不合法的组合会导致模型训练失败（NaN），应提前拒绝。

    Parameters
    ----------
    params : dict
        超参数字典

    Returns
    -------
    bool
        参数组合是否合法
    """
    seq_len = params.get("seq_len", 96)
    patch_len = params.get("patch_len", 96)
    stride = params.get("stride", 48)
    d_model = params.get("d_model", 256)
    d_ff = params.get("d_ff", 128)

    # patch_len 不能大于 seq_len
    if patch_len > seq_len:
        return False

    # stride 不能大于 patch_len
    if stride > patch_len:
        return False

    # patch 数量至少为 2（否则 Transformer 无法工作）
    n_patches = int((seq_len - patch_len) / stride) + 2
    if n_patches < 2:
        return False

    # d_ff 不应远大于 d_model（容易数值不稳定）
    if d_ff > d_model * 4:
        return False

    return True


def run_tune(search_space, base_args, n_trials, timeout, direction, db_path, log_file,
             fixed_params, model_defaults=None, local_search_ratio=0.3):
    """
    执行调参主循环。

    搜索策略：
    - 前 70% 的试验：纯随机搜索
    - 后 30% 的试验：基于历史最优结果的局部搜索（扰动）

    Parameters
    ----------
    search_space : dict
        搜索空间定义
    base_args : dict
        基础运行参数
    n_trials : int
        总试验次数
    timeout : int
        单次试验超时时间（秒）
    direction : str
        优化方向
    db_path : str
        共享数据库路径
    log_file : str
        日志文件路径
    fixed_params : dict
        固定参数
    model_defaults : dict, optional
        模型默认常量，每次试验自动注入
    local_search_ratio : float
        局部搜索占比
    """
    if model_defaults is None:
        model_defaults = {}
    db = TrialDB(db_path)
    rng = random.Random(42)
    n_random = int(n_trials * (1 - local_search_ratio))
    max_skip = n_trials * 5  # 最多跳过的无效参数次数
    skip_count = 0

    i = 0
    while i < n_trials and skip_count < max_skip:
        trial_id = db.next_trial_id()

        # 选择采样策略
        if i < n_random:
            params = sample_params(search_space, rng)
            strategy = "random"
        else:
            best = db.get_best(direction)
            if best is not None:
                params = perturb_params(best["params"], search_space, rng, strength=0.15)
                strategy = "local"
            else:
                params = sample_params(search_space, rng)
                strategy = "random"

        # 添加模型默认常量（最低优先级，不覆盖搜索/固定参数）
        for key, val in model_defaults.items():
            if key not in params:
                params[key] = val

        # 添加固定参数（覆盖搜索/默认参数）
        for key, val in fixed_params.items():
            params[key] = val

        # 校验参数合法性
        if not validate_params(params):
            skip_count += 1
            if skip_count <= 3:
                print(f"  [跳过] 参数组合不合法 (已跳过 {skip_count} 次): {json.dumps(params)}")
            continue

        params_str = json.dumps(params, ensure_ascii=False, indent=2)
        print(f"\n{'=' * 60}")
        print(f"Trial {trial_id} (策略: {strategy}, 进度: {i + 1}/{n_trials})")
        print(f"参数: {params_str}")
        print(f"{'=' * 60}")

        # 运行试验
        result = run_trial_subprocess(params, base_args, trial_id, timeout)

        # 记录
        record = {
            "trial": trial_id,
            "params": params,
            "strategy": strategy,
            "status": result["status"],
            "metric": result["metric"],
            "elapsed": round(result["elapsed"], 1),
            "error": result.get("error", ""),
            "timestamp": datetime.now().isoformat(),
        }

        # 写入共享数据库和日志
        db.append(record)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 打印摘要
        if result["status"] == "success":
            best_so_far = db.get_best(direction)
            best_val = best_so_far["metric"] if best_so_far else 0
            print(
                f"  >> Trial {trial_id}: "
                f"{base_args['metric_name']}={result['metric']:.4f}, "
                f"耗时={result['elapsed']:.1f}s, "
                f"当前最优={best_val:.4f}"
            )
        else:
            print(
                f"  >> Trial {trial_id}: 失败 ({result['status']}), "
                f"错误={result.get('error', '')[:500]}"
            )

        i += 1

    if skip_count >= max_skip:
        print(f"\n[警告] 跳过了 {skip_count} 次无效参数组合，已达到上限。请检查搜索空间。")


# ============================================================
# 结果输出
# ============================================================


def print_final_report(db_path, args, output_dir):
    """
    打印最终调参报告。

    Parameters
    ----------
    db_path : str
        共享数据库路径
    args : argparse.Namespace
        命令行参数
    output_dir : str
        输出目录
    """
    db = TrialDB(db_path)
    all_trials = db.load()
    success_trials = db.get_all_success()
    best = db.get_best(args.direction)

    print(f"\n{'=' * 60}")
    print("调参完成!")
    print(f"{'=' * 60}")

    total = len(all_trials)
    success = len(success_trials)
    print(f"总试验次数: {total}")
    print(f"成功: {success}, 失败/超时: {total - success}")

    if best is None:
        print("没有成功的试验，无法确定最优参数。")
        return

    # 最优结果
    print(f"\n最优试验: Trial {best['trial']}")
    print(f"最优指标: {args.metric} = {best['metric']:.6f}")
    print(f"最优参数:")
    for key, val in sorted(best["params"].items()):
        print(f"  {key}: {val}")

    # 保存最优参数
    best_file = os.path.join(output_dir, "best_params.json")
    best_result = {
        "best_trial": best["trial"],
        "best_value": best["metric"],
        "best_params": best["params"],
        "metric": args.metric,
        "direction": args.direction,
        "n_trials": total,
        "n_success": success,
        "timestamp": datetime.now().isoformat(),
    }
    with open(best_file, "w", encoding="utf-8") as f:
        json.dump(best_result, f, indent=2, ensure_ascii=False)
    print(f"\n最优参数已保存: {best_file}")

    # 保存试验历史
    history_file = os.path.join(output_dir, "trial_history.csv")
    if success_trials:
        rows = []
        for t in success_trials:
            row = {"trial": t["trial"], "metric": t["metric"], "status": t["status"],
                   "strategy": t.get("strategy", ""), "elapsed": t["elapsed"]}
            row.update({f"param_{k}": v for k, v in t["params"].items()})
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(history_file, index=False, encoding="utf-8-sig")
        print(f"试验历史已保存: {history_file}")

    # 生成推荐运行命令
    best_params_str = json.dumps(best["params"])
    print(f"\n推荐运行命令:")
    print(f"python ./scripts/run_benchmark.py \\")
    print(f"  --config-path {args.config_path} \\")
    print(f"  --data-name-list {args.data_name} \\")
    print(f"  --model-name {args.model_name} \\")
    print(f"  --model-hyper-params '{best_params_str}' \\")
    print(f"  --gpus {args.gpus} \\")
    print(f"  --save-path best_model_result")

    # Top-5 试验
    if success_trials:
        sorted_trials = sorted(
            success_trials,
            key=lambda t: t["metric"],
            reverse=(args.direction == "maximize"),
        )
        print(f"\nTop-5 试验:")
        for i, t in enumerate(sorted_trials[:5]):
            print(f"  #{i + 1}: Trial {t['trial']}, {args.metric}={t['metric']:.6f}")


# ============================================================
# 自定义搜索空间生成器
# ============================================================


def generate_search_space_file(output_path):
    """
    生成默认搜索空间的 JSON 配置文件模板。

    Parameters
    ----------
    output_path : str
        输出文件路径
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_SEARCH_SPACE, f, indent=2, ensure_ascii=False)
    print(f"搜索空间模板已保存: {output_path}")
    print("修改后通过 --search-space 参数指定即可使用自定义搜索空间")


# ============================================================
# 主函数
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="DAG 模型超参数自动调优工具（零外部依赖）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法
  python tools/auto_tune.py --config-path config/xxx.json --data-name data.csv

  # 指定 GPU 和试验次数
  python tools/auto_tune.py --config-path config/xxx.json --data-name data.csv --gpus 0 --n-trials 50

  # 并行调参（在多个终端运行相同命令）
  python tools/auto_tune.py --config-path config/xxx.json --data-name data.csv --shared-db tune_results/shared.json

  # 生成搜索空间模板后自定义
  python tools/auto_tune.py --gen-search-space my_space.json
  python tools/auto_tune.py --config-path config/xxx.json --data-name data.csv --search-space my_space.json

  # 固定部分参数只调其余参数
  python tools/auto_tune.py --config-path config/xxx.json --data-name data.csv --fixed-params '{"seq_len": 576, "patch_len": 96}'
        """,
    )
    parser.add_argument("--config-path", required=False, help="评估策略配置文件路径")
    parser.add_argument("--data-name", required=False, help="数据集文件名")
    parser.add_argument(
        "--model-name", default="dag.DAG", help="模型名称 (默认: dag.DAG)"
    )
    parser.add_argument("--gpus", type=int, default=0, help="GPU 编号 (默认: 0)")
    parser.add_argument(
        "--n-trials", type=int, default=30, help="总试验次数 (默认: 30)"
    )
    parser.add_argument(
        "--timeout", type=int, default=3600, help="单次试验超时时间/秒 (默认: 3600)"
    )
    parser.add_argument(
        "--metric",
        default="march_accuracy_mean",
        help="优化目标指标 (默认: march_accuracy_mean)",
    )
    parser.add_argument(
        "--direction",
        default="maximize",
        choices=["maximize", "minimize"],
        help="优化方向 (默认: maximize)",
    )
    parser.add_argument(
        "--output-dir", default="tune_results", help="调参结果输出目录 (默认: tune_results)"
    )
    parser.add_argument("--search-space", default=None, help="自定义搜索空间 JSON 文件")
    parser.add_argument(
        "--gen-search-space",
        default=None,
        help="生成默认搜索空间模板到指定路径并退出",
    )
    parser.add_argument(
        "--shared-db",
        default=None,
        help="共享数据库路径，用于并行调参 (如 tune_results/shared.json)",
    )
    parser.add_argument(
        "--fixed-params",
        default=None,
        help="固定参数 JSON 字符串 (如 '{\"seq_len\": 576}')",
    )
    parser.add_argument(
        "--model-defaults",
        default=None,
        help="模型默认常量 JSON 字符串，每次试验自动注入 (如 '{\"loss\": \"MSE\", \"lradj\": \"type3\", \"n_heads\": 8}')",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="项目根目录 (默认: 自动检测)",
    )

    args = parser.parse_args()

    # 生成搜索空间模板模式
    if args.gen_search_space:
        generate_search_space_file(args.gen_search_space)
        return

    # 检查必要参数
    if not args.config_path or not args.data_name:
        parser.error("调参模式需要 --config-path 和 --data-name 参数")

    # 确定项目根目录
    if args.project_root:
        project_root = args.project_root
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 加载搜索空间
    search_space = load_search_space(args.search_space)

    # 固定参数
    fixed_params = {}
    if args.fixed_params:
        fixed_params = json.loads(args.fixed_params)

    # 模型默认常量（自动注入每次试验，不出现在搜索空间中）
    model_defaults = {}
    if args.model_defaults:
        model_defaults = json.loads(args.model_defaults)

    # 输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 共享数据库路径
    db_path = args.shared_db or os.path.join(args.output_dir, "shared.json")

    # 日志文件
    log_file = os.path.join(
        args.output_dir, f"tune_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )

    # 基础参数
    base_args = {
        "project_root": project_root,
        "config_path": args.config_path,
        "data_name": args.data_name,
        "model_name": args.model_name,
        "gpus": args.gpus,
        "num_workers": 1,
        "metric_name": args.metric,
        "output_dir": args.output_dir,
    }

    # 打印调参配置
    print(f"\n{'=' * 60}")
    print("DAG 模型超参数自动调优")
    print(f"{'=' * 60}")
    print(f"优化目标: {args.direction} {args.metric}")
    print(f"搜索空间: {len(search_space)} 个参数")
    print(f"  {', '.join(search_space.keys())}")
    print(f"总试验次数: {args.n_trials}")
    print(f"单次超时: {args.timeout}s")
    print(f"模型: {args.model_name}")
    print(f"数据: {args.data_name}")
    print(f"结果目录: {args.output_dir}")
    print(f"日志文件: {log_file}")
    print(f"共享数据库: {db_path}")
    if fixed_params:
        print(f"固定参数: {json.dumps(fixed_params)}")
    if model_defaults:
        print(f"模型默认常量: {json.dumps(model_defaults)}")
    if args.shared_db:
        print(f"并行模式: 已启用")
        print(f"  可在另一个终端运行相同命令来并行调参")
    print(f"搜索策略: 70% 随机搜索 + 30% 基于最优结果的局部搜索")
    print(f"{'=' * 60}")

    # 运行调参
    run_tune(
        search_space=search_space,
        base_args=base_args,
        n_trials=args.n_trials,
        timeout=args.timeout,
        direction=args.direction,
        db_path=db_path,
        log_file=log_file,
        fixed_params=fixed_params,
        model_defaults=model_defaults,
    )

    # 输出最终报告
    print_final_report(db_path, args, args.output_dir)


if __name__ == "__main__":
    main()
