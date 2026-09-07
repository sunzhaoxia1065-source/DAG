# DAG 模型在线评测系统 — 完整使用文档

> 本文档面向气象公司评测人员与开发维护人员，详细说明 DAG 模型在线评测系统的架构、使用方法、配置规范、边界条件及故障排查。阅读完本文档后，您应能独立完成新数据集的准备、配置文件的修改以及评测任务的执行。

---

## 目录

1. [系统架构与工作原理](#一系统架构与工作原理)
2. [环境准备与安装](#二环境准备与安装)
3. [数据集格式要求与新数据集配置](#三数据集格式要求与新数据集配置)
4. [评测执行完整流程](#四评测执行完整流程)
5. [命令行参数详细说明](#五命令行参数详细说明)
6. [配置文件结构与字段说明](#六配置文件结构与字段说明)
7. [模型超参数详解与自定义](#七模型超参数详解与自定义)
8. [输出文件结构与解读](#八输出文件结构与解读)
9. [准确率计算口径](#九准确率计算口径)
10. [边界条件与限制说明](#十边界条件与限制说明)
11. [常见问题排查指南](#十一常见问题排查指南)
12. [附：核心代码文件清单](#十二附核心代码文件清单)

---

## 一、系统架构与工作原理

### 1.1 整体架构

DAG 评测系统基于 PyTorch + ts_benchmark 框架构建，采用"训练 + 推理 + 评测"一体化设计：

```
┌──────────────────────────────────────────────────────────────┐
│                    评测主入口 inference.py                     │
└──────────────────────────┬───────────────────────────────────┘
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
┌─────────────┐    ┌──────────────┐    ┌────────────────┐
│DataPreprocessor│  │ModelLoader  │    │InferenceEngine │
│ 数据预处理     │    │ 模型实例化   │    │ 训练+逐日推理  │
└──────┬──────┘    └──────┬───────┘    └────────┬───────┘
       │                  │                      │
       │                  ▼                      ▼
       │          ┌──────────────┐     ┌──────────────────┐
       │          │  DAG 模型    │     │  AccuracyCalculator │
       │          │ (CC/TC 双编码)│     │   准确率计算       │
       │          └──────────────┘     └──────────────────┘
       │                                      │
       └──────────────────────────────────────▼
                         ┌──────────────────┐
                         │   ResultSaver    │
                         │  输出 CSV 文件    │
                         └──────────────────┘
```

### 1.2 工作原理

**评测流程：**

1. **数据预处理**：读取 CSV 数据，校验时间连续性，分离内生变量（如 power, ws）与外生变量（如温度、气压等）
2. **模型实例化**：根据超参 JSON 创建 DAG 模型（CC 编码器 + TC 编码器 + 协变量融合模块）
3. **训练阶段**：在评测月份之前的全部历史数据上训练模型，使用 EarlyStopping 机制避免过拟合
4. **逐日推理**：遍历评测月份每一天，每日 09:00 发布预测窗口，模型生成未来 156 个点（39 小时）的预测
5. **指标计算**：取后 96 个点（次日 00:00–24:00）与真实值对比，计算每日 weighted_accuracy
6. **月度汇总**：对所有评测日求均值得到月度准确率
7. **结果输出**：生成 4 个 CSV 文件保存预测结果与统计指标

### 1.3 DAG 模型核心组件

| 组件 | 作用 | 控制参数 |
|------|------|---------|
| **CC 编码器** (Channel Correlation) | 建模历史外生变量对内生变量的影响 | `use_c`, `use_c_exog` |
| **TC 编码器** (Temporal Correlation) | 建模历史外生变量对未来外生变量的影响 | `use_t`, `use_t_exog` |
| **协变量融合模块** | 融合 CC/TC 编码结果 | `fusion_method` |
| **Patch Embedding** | 时间序列分块嵌入 | `patch_len`, `stride` |
| **Transformer 主干** | 注意力机制建模 | `d_model`, `n_heads`, `e_layers`, `d_ff` |

### 1.4 关键设计要点

- **无数据泄露**：每日推理时，模型只能看到当日 09:00 之前的历史数据，未来外生变量作为"已知未来协变量"输入（业务日 ahead 评测允许）
- **随机种子固定**：自动从 config 读取 `seed` 和 `deterministic` 字段，确保结果可复现
- **反归一化精度**：训练时对内生/外生变量分别标准化（scaler1/scaler2），预测后反归一化回原始尺度

### 1.5 变量分类与数据泄露防护（**重要**）

DAG 模型对变量有严格分类，不同类型变量的未来值使用规则不同：

#### 变量类型定义

| 类型 | 说明 | 典型例子 | 是否需要预测 |
|------|------|---------|------------|
| **目标变量** (target) | 评测的核心指标 | `power`（功率） | ✅ 是，且只评估此列 |
| **内生变量** (endogenous) | 模型需要预测的变量集合，包含目标变量 | `power`, `ws`（风速） | ✅ 是（但仅评估 target） |
| **外生变量** (exogenous) | 辅助输入变量，模型不预测它们 | 气温、气压、湿度等 | ❌ 否，作为协变量 |

#### 关系图

```
所有列 = {time} ∪ {内生变量} ∪ {外生变量}
                                    ↑
                        endogenous_columns 之外的列自动归入外生变量
```

#### 未来值使用规则（**核心约束**）

| 变量类型 | 历史值（09:00 前） | 未来值（09:00 后至次日 24:00） | 原因 |
|---------|-------------------|--------------------------------|------|
| **内生变量** | ✅ 使用 | ❌ **禁止使用** | 防止数据泄露，未来值需要预测 |
| **外生变量** | ✅ 使用 | ✅ **允许使用** | 气象预报可获取，业务日 ahead 评测允许 |

#### 数据泄露防护机制

代码层面已严格隔离：

1. **内生变量**：推理时只传入 `target_history`（评测日 09:00 之前的历史值），不传入未来值
2. **外生变量**：推理时传入 `exog_history`（历史值）+ `exog_future`（未来值），拼接后作为协变量
3. **代码位置**：[inference.py L479-490](file:///f:/华为科研/DAG(start)/tools/inference.py#L479-490) 的协变量构造逻辑

#### 变量配置随数据集改变（**灵活配置**）

内生变量和外生变量的归属**可以随数据集动态调整**，无需改代码，只需修改配置：

**方式 1：命令行临时指定**

```bash
# 数据集 A: 内生变量 = power, ws
python tools/inference.py --data dataset_A.csv \
    --endogenous-columns power,ws --target-column power ...

# 数据集 B: 内生变量 = power, sr（无 ws 列）
python tools/inference.py --data dataset_B.csv \
    --endogenous-columns power,sr --target-column power ...
```

**方式 2：config 持久配置**

```json
"strategy_args": {
    "target_column": "power",
    "endogenous_columns": ["power", "ws"]
}
```

**自动分离规则**：`endogenous_columns` 中列出的变量视为内生，CSV 中其他数值列自动归为外生变量。

#### 配置示例

假设 CSV 列为 `time, power, ws, temperature, pressure, humidity`：

| 配置 | 内生变量 | 外生变量（自动识别） |
|------|---------|---------------------|
| `--endogenous-columns power,ws` | power, ws | temperature, pressure, humidity |
| `--endogenous-columns power` | power | ws, temperature, pressure, humidity |
| `--endogenous-columns power,ws,temperature` | power, ws, temperature | pressure, humidity |

> **注意**：`ws` 设为内生时只能用历史值；设为外生时可用未来值。需根据业务语义选择。风速通常是预测目标之一，应设为内生；气温通常有预报，可设为外生。

---

## 二、环境准备与安装

### 2.1 系统要求

- **操作系统**：Linux / Windows（推荐 Linux，性能更佳）
- **Python 版本**：3.10+（推荐 3.11）
- **硬件**：建议 GPU（CUDA 11.8+），CPU 也可运行但训练较慢
- **内存**：≥ 16 GB
- **磁盘**：≥ 10 GB（含数据与依赖）

### 2.2 创建环境

```bash
# 创建 conda 环境
conda create -n dag_eval python=3.11 -y
conda activate dag_eval

# 安装 PyTorch (GPU 版本, 按 CUDA 版本调整)
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖
cd /path/to/DAG-start
pip install -r requirements.txt
```

### 2.3 验证安装

```bash
python -c "import torch; import einops; import pandas; print('环境验证通过')"
python -c "import sys; sys.path.insert(0,'.'); from ts_benchmark.baselines.dag.dag import DAG; print('模型加载成功')"
```

### 2.4 依赖清单（requirements.txt）

| 包名 | 版本 | 用途 |
|------|------|------|
| torch | 2.4.1 | 深度学习框架 |
| numpy | 1.24.4 | 数值计算 |
| pandas | 1.5.3 | 数据处理 |
| scikit-learn | 1.3.2 | 数据标准化 |
| scipy | 1.10.1 | 科学计算 |
| statsmodels | 0.14.1 | 统计工具 |
| tqdm | 4.66.4 | 进度条 |
| matplotlib | 3.7.5 | 可视化（可选） |
| reformer-pytorch | 1.4.4 | 注意力机制组件 |
| einops | 0.7.0 | **关键依赖**，DAG 网络结构必需 |

### 2.5 可选依赖

| 包名 | 用途 |
|------|------|
| ray==2.10.0 | 仅 `run_benchmark.py` 批量评测需要 |
| dash==2.17.0 | 仅可视化面板需要 |

---

## 三、数据集格式要求与新数据集配置

### 3.1 CSV 数据集格式

| 列名 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `time` | 字符串 | 是 | 时间戳，格式 `YYYY-MM-DD HH:MM:SS`，15 分钟间隔 |
| `power` | 数值 | 是 | 功率（预测目标，内生变量） |
| `ws` / `sr` | 数值 | 推荐 | 风速/辐照度（内生变量，作为模型辅助输入） |
| 其他列 | 数值 | 可选 | 外生变量（气温、湿度等，自动识别为协变量） |

### 3.2 数据集要求（硬性约束）

1. **时间间隔**：必须严格 15 分钟连续，无缺失时间点
2. **时间顺序**：必须按时间升序排列（代码会自动排序）
3. **无重复时间戳**：如有重复，自动保留最后一条并告警
4. **数值完整**：power 列必须有值（NaN 会触发异常）
5. **数据范围**：必须覆盖评测月份之前至少 `seq_len` 个时间点（默认 96 个点 = 24 小时）用于训练

### 3.3 新数据集接入步骤

#### 步骤 1：准备数据集文件

将 CSV 文件放入 `dataset/forecasting/` 目录，命名为 `<station_name>.csv`（如 `ninghe_sr.csv`）。

#### 步骤 2：在 config 中配置容量

编辑 [config/business_day_ahead_config.json](file:///f:/华为科研/DAG(start)/config/business_day_ahead_config.json)，在 `capacity` 字典中添加该数据集的装机容量：

```json
"capacity": {
    "ninghe_sr_0323_clean.csv": 80.4,
    "ninghe_sr_0323_clean": 80.4,
    "your_new_station.csv": 100.0,
    "your_new_station": 100.0
}
```

> **命名规范**：同时添加带 `.csv` 和不带 `.csv` 两种键名，兼容代码内部不同查找逻辑。

#### 步骤 3：配置内生变量（可选）

如果数据集使用了不同的内生变量列名（例如用 `wind_speed` 替代 `ws`），在 config 中修改：

```json
"endogenous_columns": ["power", "wind_speed"],
"target_column": "power"
```

或在命令行直接指定：
```bash
--endogenous-columns power,wind_speed --target-column power
```

#### 步骤 4：执行评测

```bash
python tools/inference.py \
    --data dataset/forecasting/your_new_station.csv \
    --config-path config/business_day_ahead_config.json \
    --model-hyper-params '{"seq_len":96,...}' \
    --endogenous-columns power,ws \
    --target-column power \
    --evaluation-year 2026 \
    --evaluation-month 3 \
    --capacity 100.0 \
    --output-dir results/your_station_march
```

> **--capacity 命令行参数优先级最高**，指定后可跳过 config 字典匹配，免去步骤 2。

### 3.4 数据集校验工具

运行以下命令快速校验数据集格式：

```bash
python -c "
import pandas as pd
df = pd.read_csv('dataset/forecasting/your_data.csv')
df['time'] = pd.to_datetime(df['time'])
diffs = df['time'].diff().dropna()
print(f'行数: {len(df)}, 列: {list(df.columns)}')
print(f'时间范围: {df.time.iloc[0]} ~ {df.time.iloc[-1]}')
print(f'间隔一致 (15min): {diffs.eq(pd.Timedelta(minutes=15)).all()}')
print(f'缺失值: {df.isnull().sum().to_dict()}')
"
```

---

## 四、评测执行完整流程

### 4.1 流程概览

```
[1] 环境准备  →  [2] 数据配置  →  [3] 执行评测  →  [4] 结果分析
     ↑              ↑               ↑               ↑
   conda/pip     CSV+config      inference.py    CSV 输出
```

### 4.2 场景 A：标准评测流程（气象公司推荐）

```bash
python tools/inference.py \
    --data dataset/forecasting/your_data.csv \
    --config-path config/business_day_ahead_config.json \
    --model-hyper-params '{"seq_len":96,"patch_len":48,"stride":48,"d_model":128,"d_ff":128,"n_heads":8,"e_layers":1,"horizon":156,"lr":0.002,"lradj":"type3","loss":"MSE","norm":true,"batch_size":32,"patience":4,"use_c":true,"use_t":true,"use_c_exog":true,"use_t_exog":true,"fusion_method":"mlp","alpha":0.5,"dropout":0.5,"mlp_hidden_dims":64}' \
    --endogenous-columns power,ws \
    --target-column power \
    --evaluation-year 2026 \
    --evaluation-month 3 \
    --output-dir results/march_2026
```

### 4.3 场景 B：排除停机检修日

```bash
python tools/inference.py \
    --data dataset/forecasting/your_data.csv \
    --config-path config/business_day_ahead_config.json \
    --model-hyper-params '{"seq_len":96,...}' \
    --endogenous-columns power,ws \
    --target-column power \
    --evaluation-year 2026 --evaluation-month 3 \
    --exclude-days 2026-03-15,2026-03-16,2026-03-20 \
    --output-dir results/march_2026
```

排除的日期不参与评测，不计入月度准确率均值，且会记录在 monthly_summary CSV 中便于审核。

### 4.4 场景 C：使用命令行覆盖容量

```bash
python tools/inference.py \
    --data dataset/forecasting/your_data.csv \
    --config-path config/business_day_ahead_config.json \
    --model-hyper-params '{"seq_len":96,...}' \
    --evaluation-year 2026 --evaluation-month 3 \
    --capacity 50.0
```

适用于临时评测新数据集，无需修改 config 文件。

### 4.5 执行结果分析

评测完成后，控制台输出汇总信息：

```
============================================================
推理结果汇总
============================================================
数据集:         dataset/forecasting/your_data.csv
模型:           dag.DAG
评测月份:       2026-03
评测天数:       30 天
排除天数:       1 天
装机容量:       50.0 MW
march_accuracy_mean:  0.7890
最高日精度:     0.9234 (2026-03-08)
最低日精度:     0.6123 (2026-03-15)
============================================================
```

详细结果在 `--output-dir` 指定目录下的 4 个 CSV 文件中（见第八节）。

---

## 五、命令行参数详细说明

### 5.1 必需参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `--data` / `-d` | str | 数据集 CSV 文件路径（必填） |
| `--config-path` / `-c` | str | 评测配置 JSON 文件路径（必填） |

### 5.2 数据参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--target-column` | `power` | 预测目标列名 |
| `--endogenous-columns` | `[power]` | 内生变量列名，逗号分隔（如 `power,ws`） |
| `--capacity` | config 匹配 | 装机容量 MW，覆盖 config 中的字典查找 |

### 5.3 评测参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--evaluation-year` | config | 评测年份 |
| `--evaluation-month` | config | 评测月份（1-12） |
| `--exclude-days` | 无 | 排除日期，逗号分隔（如 `2026-03-15,2026-03-16`） |
| `--horizon` | config (156) | 预测长度 |
| `--issue-hour` | config (9) | 发布时刻（小时，整数） |
| `--output-dir` | 数据集同目录/inference_results | 结果输出目录 |

### 5.4 模型超参

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-name` | `dag.DAG` | 模型名称 |
| `--model-hyper-params` | 无 | 模型超参 JSON 字符串（必填，包含架构参数） |
| `--gpus` | `0` | GPU 编号 |

### 5.5 架构开关

| 参数 | 说明 |
|------|------|
| `--use-c` | 是否启用 CC 编码器（`true`/`false`） |
| `--use-t` | 是否启用 TC 编码器（`true`/`false`） |
| `--use-c-exog` | CC 是否使用外生变量 |
| `--use-t-exog` | TC 是否使用外生变量 |
| `--fusion-method` | 协变量融合方法（`mlp` / `conv` / `cross_attention` / 空） |
| `--loss` | 损失函数（`MSE` / `MAE` / `Huber`） |
| `--norm` | 是否标准化数据 |

### 5.6 其他

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--log-level` | `INFO` | 日志级别（DEBUG/INFO/WARNING/ERROR） |

---

## 六、配置文件结构与字段说明

### 6.1 配置文件总览

[config/business_day_ahead_config.json](file:///f:/华为科研/DAG(start)/config/business_day_ahead_config.json) 采用分层结构：

```json
{
    "data_config": {...},
    "model_config": {...},
    "evaluation_config": {
        "metrics": [...],
        "strategy_args": {...}      ← inference.py 实际使用的部分
    },
    "report_config": {...}
}
```

`inference.py` 通过 `load_config()` 读取 `evaluation_config.strategy_args` 部分。

### 6.2 strategy_args 字段详解

```json
"strategy_args": {
    "strategy_name": "business_day_ahead",
    "horizon": 156,                    // 预测长度 (单位: 15分钟点数)
    "evaluation_horizon": 96,          // 评测窗口 (次日 00:00-24:00, 96 点)
    "issue_hour": 9,                   // 每日发布时刻 (9 表示 09:00)
    "evaluation_year": 2026,           // 评测年份
    "evaluation_month": 3,            // 评测月份
    "capacity": {                      // 装机容量字典 (MW)
        "ninghe_sr_0323_clean.csv": 80.4,
        "ninghe_sr_0323_clean": 80.4
    },
    "train_ratio_in_tv": 0.875,        // 训练/验证集比例 (87.5% 训练)
    "seed": 2021,                      // 随机种子 (保证可复现)
    "deterministic": "efficient",      // 确定性模式: full/efficient/none
    "save_true_pred": true,
    "target_column": "power",          // 预测目标列
    "endogenous_columns": ["power", "sr"],  // 内生变量
    "exclude_days": ["2026-03-01"]     // 默认排除日期 (可被 --exclude-days 覆盖)
}
```

### 6.3 字段含义与修改规范

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `horizon` | int | 156 | 预测长度。计算公式: `(24 - issue_hour) * 4 + evaluation_horizon`。09:00 发布 → 次日 00:00 共 60 点 + 96 = 156 |
| `evaluation_horizon` | int | 96 | 实际参与准确率计算的点数（次日全天） |
| `issue_hour` | int | 9 | 发布时刻，整数 0-23 |
| `train_ratio_in_tv` | float | 0.875 | 训练/验证切分比例。0.875 = 87.5% 训练 + 12.5% 验证 |
| `seed` | int | 2021 | 随机种子，控制权重初始化、DataLoader 顺序 |
| `deterministic` | str | `"efficient"` | 确定性模式：`full`（cudnn 确定性）/`efficient`（基础）/`none`（不设置） |
| `capacity` | dict | - | 数据集名 → 装机容量 MW 映射。必须同时配 `.csv` 和不带后缀两种键 |
| `target_column` | str | `"power"` | 预测目标列 |
| `endogenous_columns` | list | `["power","sr"]` | 内生变量列名列表 |
| `exclude_days` | list | `[]` | 默认排除日期，可被 `--exclude-days` 覆盖 |

### 6.4 修改规范

1. **JSON 语法严格**：所有字符串用双引号，最后一个字段不能有逗号
2. **修改前备份**：建议保留原版 config，复制为新文件再修改
3. **键名大小写敏感**：`capacity` 与 `Capacity` 视为不同键
4. **capacity 字典键名**：必须同时添加 `<name>.csv` 和 `<name>` 两种写法
5. **命令行优先级**：CLI 参数覆盖 config，无需改 config 即可调整 `evaluation_year`/`month`/`capacity`/`exclude_days`

---

## 七、模型超参数详解与自定义

### 7.1 架构参数（影响模型容量与精度）

| 参数 | 类型 | 推荐值 | 说明 |
|------|------|--------|------|
| `seq_len` | int | 96 | 历史窗口长度（单位: 15分钟点数）。96 = 24 小时历史 |
| `patch_len` | int | 48 | Patch 长度，将历史序列切块嵌入 |
| `stride` | int | 48 | Patch 步长 |
| `d_model` | int | 128 | 模型隐藏维度 |
| `d_ff` | int | 128 | FFN 中间维度 |
| `n_heads` | int | 8 | 多头注意力头数 |
| `e_layers` | int | 1 | 编码器层数 |
| `dropout` | float | 0.5 | Dropout 比例 |
| `alpha` | float | 0.5 | 损失函数混合系数（DBLoss） |

### 7.2 训练参数（影响收敛速度与稳定性）

| 参数 | 类型 | 推荐值 | 说明 |
|------|------|--------|------|
| `lr` | float | 0.002 | 学习率（Adam 优化器） |
| `lradj` | str | `"type3"` | 学习率调整策略 |
| `batch_size` | int | 32 | 批大小 |
| `num_epochs` | int | 100 | 最大训练轮数 |
| `patience` | int | 4 | EarlyStopping 耐心值，验证集 4 轮无提升即停止 |
| `loss` | str | `"MSE"` | 损失函数，支持 `MSE`/`MAE`/`Huber`/`SmoothL1` |
| `norm` | bool | `true` | 是否标准化数据 |
| `use_amp` | int | 0 | 是否使用混合精度训练（0=否, 1=是） |
| `clip_grad_norm` | float | 1.0 | 梯度裁剪阈值 |

### 7.3 DAG 架构开关

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_c` | `true` | 启用 CC 编码器（建模外生→内生的影响） |
| `use_t` | `true` | 启用 TC 编码器（建模外生→外生的时序依赖） |
| `use_c_exog` | `true` | CC 编码器是否使用外生变量 |
| `use_t_exog` | `true` | TC 编码器是否使用外生变量 |
| `fusion_method` | `""` | 协变量融合方法：`mlp` / `conv` / `cross_attention` |
| `mlp_hidden_dims` | 64 | MLP 融合层维度 |
| `alpha_cov` | 1.0 | 协变量融合权重（仅 `fusion_method=conv/cross_attention` 时生效） |

### 7.4 推理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pred_dim` | `None` | 预测内生变量数。默认 `None`=全部；设为 `1` 则只预测 power |
| `horizon` | 156 | 预测长度（与 strategy_args.horizon 一致） |

### 7.5 调参建议

- **训练不稳（loss 波动大）**：降低 `lr` 至 0.0005-0.001，增大 `batch_size`
- **过拟合（训练 loss 低，验证 loss 高）**：增大 `dropout` 至 0.6，减小 `e_layers` 至 1
- **欠拟合（loss 不下降）**：增大 `d_model` 至 256，延长 `patience` 至 8
- **内存不足**：减小 `batch_size` 至 16，或启用 `use_amp=1`

---

## 八、输出文件结构与解读

### 8.1 输出目录

默认输出到 `<数据集目录>/inference_results/`，可通过 `--output-dir` 自定义。

### 8.2 输出文件清单

| 文件名 | 内容 | 用途 |
|--------|------|------|
| `monthly_prediction_{series}_{YYYYMM}.csv` | 完整月度 power 预测（15 分钟间隔） | 自动化处理、二次分析 |
| `daily_accuracy_{series}_{YYYYMM}.csv` | 每日准确率 + 月度均值行 | 快速查看日度表现 |
| `monthly_summary_{series}_{YYYYMM}.csv` | 月度统计汇总 | 气象公司考核报告 |
| `prediction_detail_{series}.csv` | 逐点预测对比（含所有内生变量） | 详细技术分析 |

### 8.3 monthly_prediction 列说明

| 列名 | 类型 | 说明 |
|------|------|------|
| `date` | str | 日期（`YYYY-MM-DD`） |
| `time` | str | 时间戳（`YYYY-MM-DD HH:MM:SS`，15 分钟间隔） |
| `predicted_power` | float | 预测功率 |
| `actual_power` | float | 实际功率 |
| `error` | float | 误差（`actual - predicted`） |
| `daily_accuracy` | float | 当日准确率 |

### 8.4 daily_accuracy 列说明

| 列名 | 说明 |
|------|------|
| `date` | 日期，最后一行为 `MONTHLY_MEAN ({metric_name})` |
| `daily_accuracy` | 当日准确率，最后一行为月度均值 |

### 8.5 monthly_summary 列说明

| metric | value 含义 |
|--------|----------|
| `series_name` | 数据集名 |
| `evaluation_year` | 评测年份 |
| `evaluation_month` | 评测月份 |
| `eval_day_count` | 参与评测的天数 |
| `exclude_day_count` | 被排除的天数 |
| `exclude_days` | 排除的日期列表（逗号分隔） |
| `capacity_mw` | 装机容量（MW） |
| `{month}_accuracy_mean` | 月度准确率均值 |
| `max_daily_accuracy` | 最高日准确率 |
| `min_daily_accuracy` | 最低日准确率 |
| `std_daily_accuracy` | 日准确率标准差 |

---

## 九、准确率计算口径

与业务评测口径完全一致（参考 [business_day_ahead.py](file:///f:/华为科研/DAG(start)/ts_benchmark/evaluation/strategy/business_day_ahead.py)）：

### 9.1 每日准确率

```
accuracy_daily = 1 - weighted_RMSE / capacity
```

其中：
- `weighted_RMSE = sqrt( sum( error² × weights ) )`
- `weights = |error| / sum(|error|)`（按绝对误差加权）
- `error = actual_power - predicted_power`

### 9.2 月度准确率均值

```
accuracy_monthly_mean = mean( accuracy_daily for all eval days )
```

### 9.3 示例

某日 96 个点，capacity = 50 MW：
- 绝对误差总和 = 100
- 第一个点误差 = 2 → 权重 = 2/100 = 0.02
- 该点加权误差² = 4 × 0.02 = 0.08
- 所有 96 点加权误差²之和取平方根 = weighted_RMSE
- daily_accuracy = 1 - weighted_RMSE / 50

---

## 十、边界条件与限制说明

### 10.1 数据要求

| 约束 | 说明 |
|------|------|
| 时间间隔 | 必须严格 15 分钟，不连续会报错 |
| 评测月前数据量 | 至少 `seq_len` 个点（默认 96 = 24 小时），否则无法训练 |
| 评测月数据完整 | 评测月每一天都必须有完整 96 个点（00:00-24:00），否则该日跳过 |
| 目标列无 NaN | `power` 列必须全部有值 |

### 10.2 资源限制

| 项目 | 默认上限 | 说明 |
|------|---------|------|
| GPU 显存 | 取决于 `batch_size` | 显存不足时减小 batch_size 或启用 `use_amp` |
| 训练时长 | 取决于 `num_epochs` | 通常 5-15 分钟（GPU），CPU 可能数小时 |
| 内存 | ~8 GB | 大数据集需更多 |

### 10.3 架构限制

- **仅支持 15 分钟间隔**：不支持其他时间分辨率
- **仅支持回归任务**：不支持分类
- **仅支持单步预测**：每日独立预测，不做迭代多步预测
- **外生变量未来值必须已知**：评测日的外生变量必须预先提供（气象预报数据）

### 10.4 不支持的场景

- 多目标同时预测（`pred_dim > 1` 时只能预测前 N 个内生变量）
- 实时流式推理（仅支持离线批处理）
- 自定义评测窗口（固定次日 00:00-24:00 共 96 点）

---

## 十一、常见问题排查指南

### 11.1 数据格式错误

**Q1: 报错 "时间序列不连续, 要求 15 分钟间隔"**

A: 数据集存在缺失时间点。解决方法：
1. 用 11.4 节校验工具定位缺失位置
2. 用插值填补：`df = df.resample('15min').interpolate()`
3. 或用 `--exclude-days` 跳过有缺失的日期

**Q2: 报错 "时间列 'time' 不在数据集中"**

A: CSV 第一列必须叫 `time`。如列名不同，用 `--target-column` 指定目标列，但 time 列名是硬约束，需重命名列。

**Q3: 报错 "内生变量列 'ws' 不在数据集中"**

A: 命令行 `--endogenous-columns` 指定的列名在 CSV 中不存在。检查列名拼写，或修改 `--endogenous-columns` 参数。

### 11.2 参数配置不当

**Q4: 报错 "config object has no attribute 'seq_len'"**

A: 训练模式必须传 `--model-hyper-params` 且 JSON 中必须包含 `seq_len`。完整示例：

```bash
--model-hyper-params '{"seq_len":96,"patch_len":48,"stride":48,"d_model":128,"d_ff":128,"n_heads":8,"e_layers":1,"horizon":156,"lr":0.002,"lradj":"type3","loss":"MSE","norm":true,"batch_size":32,"patience":4,"use_c":true,"use_t":true,"use_c_exog":true,"use_t_exog":true,"fusion_method":"mlp","alpha":0.5,"dropout":0.5}'
```

**Q5: 报错 "config capacity 字典中找不到 series_name='...'"**

A: 三种解决方案（任选一）：
1. 命令行加 `--capacity 50.0` 直接指定
2. 在 config 的 `capacity` 字典添加该数据集条目（同时加 `.csv` 和不带后缀两种键）
3. 在 capacity 字典添加 `"__default__": 50.0` 作为兜底

**Q6: 报错 "评测月 2026-03 之前没有训练数据"**

A: 数据集时间范围未覆盖评测月之前。检查 CSV 时间范围，确保包含 2026-02 及更早的数据。

### 11.3 任务执行失败

**Q7: 报错 "依赖缺失, ts_benchmark 未安装"**

A: 项目根目录未加入 sys.path。`inference.py` 已自动注入，但仍需在项目根目录运行：
```bash
cd /path/to/DAG-start
python tools/inference.py ...
```

**Q8: 报错 "ImportError: einops"**

A: 隐藏依赖未安装：`pip install einops==0.7.0`

**Q9: 训练时 GPU 显存不足**

A: 减小 batch_size：`--model-hyper-params '{"batch_size":16,...}'`，或启用混合精度：`"use_amp":1`

**Q10: 训练准确率异常高（如 95%+）**

A: 可能数据泄露。检查：
1. `endogenous_columns` 是否包含未来值（不应包含）
2. `capacity` 是否正确（错误的容量会导致准确率虚高）
3. 是否有内生变量在评测月使用了真实未来值

**Q11: 同样参数每次运行结果不同**

A: 随机种子未固定。检查 config 是否包含 `"seed": 2021, "deterministic": "efficient"`。inference.py 会自动读取这两个字段。

### 11.4 数据校验脚本

```python
import pandas as pd
df = pd.read_csv('your_data.csv')
df['time'] = pd.to_datetime(df['time'])
diffs = df['time'].diff().dropna()
print(f'行数: {len(df)}')
print(f'列: {list(df.columns)}')
print(f'时间范围: {df.time.iloc[0]} ~ {df.time.iloc[-1]}')
print(f'15min 间隔一致: {diffs.eq(pd.Timedelta(minutes=15)).all()}')
if not diffs.eq(pd.Timedelta(minutes=15)).all():
    bad = diffs[~diffs.eq(pd.Timedelta(minutes=15))]
    print(f'异常间隔位置: {bad.head().to_dict()}')
print(f'缺失值统计: {df.isnull().sum().to_dict()}')
```

---

## 十二、附：核心代码文件清单

| 文件路径 | 功能 |
|---------|------|
| `tools/inference.py` | 评测主脚本（训练+推理+输出） |
| `config/business_day_ahead_config.json` | 评测策略配置 |
| `ts_benchmark/baselines/dag/dag.py` | DAG 模型适配器 |
| `ts_benchmark/baselines/dag/models/dag_model.py` | DAGModel 网络定义 |
| `ts_benchmark/baselines/dag/layers/CC_EncDec.py` | CC 编码器 |
| `ts_benchmark/baselines/dag/layers/TC_EncDec.py` | TC 编码器 |
| `ts_benchmark/baselines/deep_forecasting_model_base.py` | 模型基类（forecast_fit/forecast） |
| `ts_benchmark/baselines/utils.py` | 共享工具（数据加载、归一化、DBLoss） |
| `ts_benchmark/evaluation/strategy/business_day_ahead.py` | 业务日 ahead 评测策略 |
| `ts_benchmark/utils/random_utils.py` | 随机种子工具 |
| `requirements.txt` | 依赖清单 |
