# -*- coding: utf-8 -*-
"""
GCGNet数据预处理脚本

功能：
1. 读取原始数据文件
2. 删除label列（不参与建模）
3. 重新排列列顺序：[power, sr/ws, 其他协变量]
4. 保存处理后的数据

数据格式要求：
- guang数据集：time, power, sr, label, [其他协变量...]
- feng数据集：time, power, ws, label, [其他协变量...]

输出格式：
- 第0列：power（主预测目标）
- 第1列：sr/ws（辅助内生变量）
- 第2~N列：其他外生协变量
"""

import pandas as pd
import os
from pathlib import Path

# 配置路径
INPUT_DIR = Path(__file__).parent / "dataset" / "forecasting"
OUTPUT_DIR = Path(__file__).parent / "dataset" / "forecasting" / "processed"

# 内生变量（参与图构建，但不全部作为预测目标）
ENDO_VARIABLES = ["power", "sr", "ws"]

# 主预测目标（只预测这个）
TARGET_VARIABLE = "power"

# 需要删除的列
DROP_COLUMNS = ["label"]


def preprocess_single_dataset(
    input_path: str,
    output_path: str,
    aux_endo_variable: str = None
) -> None:
    """
    预处理单个数据集

    参数:
        input_path: 输入文件路径
        output_path: 输出文件路径
        aux_endo_variable: 辅助内生变量名称（如 "sr" 或 "ws"）
    """
    print(f"\n{'='*60}")
    print(f"处理文件: {input_path}")
    print(f"{'='*60}")

    # 读取数据
    df = pd.read_csv(input_path)
    print(f"原始列: {df.columns.tolist()}")
    print(f"原始形状: {df.shape}")

    # 检查是否包含 time 列，如果有则设为索引
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.set_index("time")
        print("已将 'time' 列设为索引")

    # 删除不需要的列
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"已删除列: {cols_to_drop}")

    # 重新排列列顺序
    new_columns = []

    # 1. 主目标变量（power）
    if TARGET_VARIABLE in df.columns:
        new_columns.append(TARGET_VARIABLE)
        print(f"主目标: {TARGET_VARIABLE}")

    # 2. 辅助内生变量（sr 或 ws）
    if aux_endo_variable and aux_endo_variable in df.columns:
        new_columns.append(aux_endo_variable)
        print(f"辅助内生变量: {aux_endo_variable}")

    # 3. 其他所有列（作为外生协变量）
    other_cols = [c for c in df.columns if c not in new_columns]
    new_columns.extend(other_cols)
    print(f"外生协变量: {other_cols}")

    # 重新排列
    df = df[new_columns]
    print(f"\n新列顺序: {df.columns.tolist()}")

    # 验证列顺序
    assert df.columns[0] == TARGET_VARIABLE, "power 必须在第0列"
    print(f"✓ 验证通过: power 在第0列")

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    print(f"✓ 已保存至: {output_path}")
    print(f"最终形状: {df.shape}")

    return df


def preprocess_guang_dataset():
    """预处理guang数据集（包含sr作为辅助内生变量）"""
    input_path = INPUT_DIR / "guang.csv"
    output_path = OUTPUT_DIR / "guang_processed.csv"

    # guang数据集包含 power, sr 作为内生变量
    return preprocess_single_dataset(
        input_path=str(input_path),
        output_path=str(output_path),
        aux_endo_variable="sr"
    )


def preprocess_feng_dataset():
    """预处理feng数据集（包含ws作为辅助内生变量）"""
    input_path = INPUT_DIR / "feng.csv"
    output_path = OUTPUT_DIR / "feng_processed.csv"

    # feng数据集包含 power, ws 作为内生变量
    return preprocess_single_dataset(
        input_path=str(input_path),
        output_path=str(output_path),
        aux_endo_variable="ws"
    )


def main():
    """主函数"""
    print("="*60)
    print("GCGNet 数据预处理脚本")
    print("="*60)
    print(f"\n输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")

    # 预处理两个数据集
    preprocess_guang_dataset()
    preprocess_feng_dataset()

    print("\n" + "="*60)
    print("预处理完成!")
    print("="*60)
    print("\n处理后的数据格式:")
    print("  guang_processed.csv: [power, sr, 其他外生协变量...]")
    print("  feng_processed.csv:   [power, ws, 其他外生协变量...]")
    print("\n变量说明:")
    print("  - power: 主预测目标（用于损失计算）")
    print("  - sr/ws: 辅助内生变量（参与图构建，用于辅助损失）")
    print("  - 其他列: 外生协变量（历史+未来输入）")


if __name__ == "__main__":
    main()
