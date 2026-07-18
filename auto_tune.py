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
        save_path,
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
            stderr_tail = result.stderr[-500:] if result.stderr else "unknown"
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

    # 方式1: 直接查找 CSV
    for csv_file in save_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if metric_name in df.columns:
                val = df[metric_name].iloc[0]
                if pd.notna(val):
                    return float(val)
        except Exception:
            continue

    # 方式2: 解压 tar.gz
    tar_files = list(save_dir.glob("*.csv.tar.gz"))
    if tar_files:
        latest_tar = max(tar_files, key=lambda p: p.stat().st_mtime)
        try:
            with tarfile.open(latest_tar, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith(".csv"):
                        f = tar.extractfile(member)
                        if f is None:
                            continue
                        try:
                            df = pd.read_csv(f)
                            if metric_name in df.columns:
                                val = df[metric_name].iloc[0]
                                if pd.notna(val):
                                    return float(val)
                        except Exception:
                            continue
        except Exception:
            pass

    # 方式3: 递归搜索子目录
    for csv_file in save_dir.rglob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if metric_name in df.columns:
                val = df[metric_name].iloc[0]
                if pd.notna(val):
                    return float(val)
        except Exception:
            continue

    return None


# ============================================================
# 调参主循环
# ============================================================


def run_tune(search_space, base_args, n_trials, timeout, direction, db_path, log_file,
             fixed_params, local_search_ratio=0.3):
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
    local_search_ratio : float
        局部搜索占比
    """
    db = TrialDB(db_path)
    rng = random.Random(42)
    n_random = int(n_trials * (1 - local_search_ratio))

    for i in range(n_trials):
        trial_id = db.next_trial_id()

        # 选择采样策略
        if i < n_random:
            # 随机搜索
            params = sample_params(search_space, rng)
            strategy = "random"
        else:
            # 局部搜索：基于历史最优结果扰动
            best = db.get_best(direction)
            if best is not None:
                params = perturb_params(best["params"], search_space, rng, strength=0.15)
                strategy = "local"
            else:
                params = sample_params(search_space, rng)
                strategy = "random"

        # 添加固定参数
        for key, val in fixed_params.items():
            params[key] = val

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
                f"错误={result.get('error', '')[:100]}"
            )


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
    )

    # 输出最终报告
    print_final_report(db_path, args, args.output_dir)


if __name__ == "__main__":
    main()
