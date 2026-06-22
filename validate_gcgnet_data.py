# -*- coding: utf-8 -*-
"""
数据验证脚本

验证预处理后的数据是否符合 GCGNet 模型的要求：
1. 列顺序正确：[power, sr/ws, 外生协变量...]
2. time 已设为索引
3. 无缺失值
4. 数据类型正确
"""

import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "dataset" / "forecasting" / "processed"


def validate_dataset(file_path: str, dataset_name: str, aux_col: str = None):
    """验证数据集是否符合要求"""
    print(f"\n{'='*60}")
    print(f"验证数据集: {dataset_name}")
    print(f"{'='*60}")

    df = pd.read_csv(file_path, index_col=0, parse_index=True)

    # 检查索引是否为时间类型
    print(f"\n1. 索引检查:")
    print(f"   索引类型: {type(df.index)}")
    print(f"   索引示例: {df.index[0]}")
    if pd.api.types.is_datetime64_any_dtype(df.index):
        print("   ✓ 索引是时间类型")
    else:
        print("   ✗ 警告: 索引不是时间类型")

    # 检查列顺序
    print(f"\n2. 列顺序检查:")
    print(f"   列名: {df.columns.tolist()}")
    if df.columns[0] == "power":
        print("   ✓ power 在第0列（主预测目标）")
    else:
        print(f"   ✗ 错误: 第0列应该是 power，实际是 {df.columns[0]}")

    if aux_col and aux_col in df.columns:
        aux_idx = df.columns.get_loc(aux_col)
        if aux_idx == 1:
            print(f"   ✓ {aux_col} 在第1列（辅助内生变量）")
        else:
            print(f"   ✗ 警告: {aux_col} 在第{aux_idx}列，应该在第1列")

    # 检查缺失值
    print(f"\n3. 缺失值检查:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   ✓ 无缺失值")
    else:
        print(f"   ✗ 存在缺失值:")
        for col, count in missing[missing > 0].items():
            print(f"     - {col}: {count} 个缺失值")

    # 检查数值列
    print(f"\n4. 数据类型检查:")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    print(f"   数值列: {numeric_cols}")
    if non_numeric_cols:
        print(f"   非数值列: {non_numeric_cols}")
        print("   ✓ 外生协变量可以包含非数值列（将自动处理）")

    # 数据形状
    print(f"\n5. 数据形状:")
    print(f"   样本数: {len(df)}")
    print(f"   特征数: {len(df.columns)}")
    print(f"   内生变量: 1 (power)")
    print(f"   辅助内生变量: 1 ({aux_col})")
    print(f"   外生协变量: {len(df.columns) - 2}")

    print(f"\n6. 数据预览:")
    print(df.head(3))

    return df


def main():
    print("="*60)
    print("数据验证脚本")
    print("="*60)

    # 验证 guang 数据集
    guang_path = OUTPUT_DIR / "guang_processed.csv"
    if guang_path.exists():
        validate_dataset(str(guang_path), "guang", aux_col="sr")
    else:
        print(f"✗ 文件不存在: {guang_path}")
        print("  请先运行 preprocess_gcgnet_data.py")

    # 验证 feng 数据集
    feng_path = OUTPUT_DIR / "feng_processed.csv"
    if feng_path.exists():
        validate_dataset(str(feng_path), "feng", aux_col="ws")
    else:
        print(f"✗ 文件不存在: {feng_path}")
        print("  请先运行 preprocess_gcgnet_data.py")

    print("\n" + "="*60)
    print("验证完成!")
    print("="*60)


if __name__ == "__main__":
    main()
