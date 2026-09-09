# DAG 模型推理与数据处理使用指南

本指南详细说明如何从原始气象数据生成模型可用的 CSV 文件，并使用 `inference.py` 完成训练/推理全流程。

---

## 目录

1. [环境准备](#1-环境准备)
2. [原始数据处理流程 (data_processor.py)](#2-原始数据处理流程-data_processorpy)
3. [模型推理流程 (inference.py)](#3-模型推理流程-inferencepy)
4. [输入输出格式定义](#4-输入输出格式定义)
5. [完整使用示例](#5-完整使用示例)

---

## 1. 环境准备

```bash
# 确保安装了 ts_benchmark 本地包和依赖
pip install torch pandas numpy scikit-learn tqdm
```

项目根目录下的 `ts_benchmark` 为本地包，脚本会自动将其加入 `sys.path`，无需额外安装。

---

## 2. 原始数据处理流程 (data_processor.py)

### 2.1 概述

`data_processor.py` 接收原始 EC 气象预报数据 + 功率实测数据，经过 6 步流水线处理，输出符合 DAG 模型训练/推理要求的格式化 CSV。

**6 步流水线：**

| 步骤 | 说明 | 风电 | 光伏 |
|------|------|------|------|
| 1. 按经纬度拆分 | 将原始 EC 数据按 `point_id` 拆分为多个网格点 CSV | ✓ | ✓ |
| 2. 清理+时间过滤 | 清理无用列，按时间区间过滤 | ✓ | ✓ |
| 3. 计算梯度/风切变 | 计算水平梯度、垂直风切变、幂指数 | ✓ | 跳过 |
| 4. 合并主文件+网格点 | 合并中心网格点与周边网格点风速(风电)/简单合并(光伏) | ✓ | ✓ |
| 5. 合并功率+缺失值处理 | 合并功率数据，处理异常0值和缺失值 | ✓ | ✓ |
| 6. 特征选择 | 选择最终特征列 | ✓ (~130列) | ✓ (7列) |

### 2.2 命令行参数

#### 必需参数

| 参数 | 说明 |
|------|------|
| `--plant-type` | 发电类型：`wind`(风电) / `solar`(光伏) |
| `--ec-data` | 原始 EC 气象数据 CSV 路径 |
| `--power-data` | 功率实测数据路径（CSV 格式） |
| `--work-dir` | 工作目录（存放中间和最终结果） |

#### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--time-start` | `202407030000` | 起始时间（格式 YYYYMMDDHHmm） |
| `--time-end` | `202607312345` | 结束时间（格式 YYYYMMDDHHmm） |
| `--center-grid` | 第一个点 | 中心网格点坐标（如 `41_400_114_900`），不指定则用第一个点 |
| `--fill-method` | `fill_periodic_mean` | 缺失值填充方法：`fill_periodic_mean`(周期均值插补) / `interpolate`(线性插值) / `mean`(均值填充) / `keep`(保留不处理) |
| `--periodic-neighbors` | `3` | 周期插补前后天数（`fill_periodic_mean` 策略使用前后 N 天数据计算均值） |
| `--test-cutoff` | 无 | 光伏测试期起始日期（如 `2026-03-01`），内生变量(power/sr)不检测此日期之后的异常0值，保护评估数据 |
| `--steps` | 全部 | 执行步骤（逗号分隔，如 `1,2,5,6`；默认全部） |
| `--features` | 无 | 自定义特征列表（逗号分隔，如 `time,power,ws,u_wind_component_surface_10_metre`）。**自定义特征优先于默认特征集** |
| `--features-file` | 无 | 从文件加载自定义特征列表（每行一个或逗号分隔） |
| `--lat-range` | 风电默认5点 | 纬度范围（逗号分隔，如 `41.2,41.3,41.4,41.5,41.6`） |
| `--lon-range` | 风电默认5点 | 经度范围（逗号分隔，如 `114.7,114.8,114.9,115.0,115.1`） |

#### 关键参数解释

- **`--plant-type`**: 决定内生变量（风电=`power,ws`；光伏=`power,sr`）和特征选择规则。风电执行全部6步，光伏跳过步骤3（梯度计算）。
- **`--fill-method`**: `fill_periodic_mean` 是推荐方法，使用测试截止日期前后 N 天同时段的均值替换异常0值，适用于具有日周期规律的数据。`keep` 仅用于光伏，保留异常0值不处理。
- **`--periodic-neighbors`**: 控制周期插补的数据范围。值越大，插补越平滑但可能丢失局部特征；值越小，越贴近局部数据但可能受噪声影响。
- **`--test-cutoff`**: 重要参数。保护评测期数据不被修改。光伏内生变量(power/sr)的异常0值检测仅在此日期之前执行。例如 `--test-cutoff 2026-03-01` 表示2026年3月1日之后的数据不会被修改。
- **`--center-grid`**: 风电场/光伏电站所在中心网格点坐标。风电需匹配5x5网格中的中心点。光伏仅使用中心点数据。
- **`--features` / `--features-file`**: 双模式特征选择。**自定义特征优先于默认特征集**。若指定了自定义特征，则使用自定义列表；否则使用默认特征集（光伏7列，风电约130列）。

### 2.3 数据处理命令示例

```bash
# 风电数据处理（完整6步）
python tools/data_processor.py --plant-type wind \
    --ec-data 玖天数据集处理/haojiaying.csv \
    --power-data 玖天数据集处理/郝家营风电场二期power-ws.csv \
    --work-dir 玖天数据集处理/processed_wind \
    --time-start 202407030000 --time-end 202607312345 \
    --center-grid 41_400_114_900

# 光伏数据处理（跳过梯度计算）
python tools/data_processor.py --plant-type solar \
    --ec-data 玖天数据集处理/ninghe.csv \
    --power-data 玖天数据集处理/宁河光伏电站power-sr.csv \
    --work-dir 玖天数据集处理/processed_solar \
    --time-start 202407030000 --time-end 202607312345 \
    --center-grid 39_200_117_400 \
    --test-cutoff 2026-03-01

# 仅执行指定步骤（调试用）
python tools/data_processor.py --plant-type wind \
    --ec-data ... --power-data ... --work-dir ... --steps 1,2,5

# 使用自定义特征列表
python tools/data_processor.py --plant-type solar \
    --ec-data ... --power-data ... --work-dir ... \
    --features time,power,sr,total_sky_direct_solar_radiation_at_surface_surface
```

### 2.4 数据处理输出

处理完成后，最终 CSV 文件位于 `{work-dir}/final/` 目录，列顺序为：

```
time, power, sr(光伏)/ws(风电), [外生变量...]
```

- `time` 列：ISO 格式时间字符串（`2026-04-01 00:00:00`），15分钟间隔
- `power` 列：功率实测值（目标预测变量）
- `sr`/`ws` 列：历史输入变量（内生变量，仅使用历史值）
- 外生变量列：气象预报数据（可使用未来值）

---

## 3. 模型推理流程 (inference.py)

### 3.1 概述

`inference.py` 支持两种推理模式：

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| **模式 A（训练后推理）** | 从数据训练模型，再逐日推理 | 有训练数据和 GPU 的用户 |
| **模式 B（加载预训练模型推理）** | 直接加载 `.ckpt` 模型文件推理，跳过训练 | 无法自行训练（缺 GPU/训练数据/时间）的用户 |

### 3.2 命令行参数

#### 数据参数

| 参数 | 必需 | 说明 |
|------|------|------|
| `--data` / `-d` | 是 | 数据集 CSV 文件路径（经 data_processor.py 处理后的格式化 CSV） |
| `--config-path` / `-c` | 是 | 策略配置 JSON 文件路径（如 `config/business_day_ahead_config.json`） |
| `--target-column` | 否 | 预测目标列名（默认: `power`） |
| `--endogenous-columns` | 否 | 内生变量列名，逗号分隔（如 `power,sr`）。不指定则从 config 读取 |

#### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-name` | `dag.DAG` | 模型名称 |
| `--model-hyper-params` | 无 | 模型超参 JSON 字符串（如 `'{"seq_len":96,"d_model":128}'`） |
| `--gpus` | `0` | GPU 编号 |

#### 架构开关

| 参数 | 说明 |
|------|------|
| `--use-c` | 是否使用 CC encoder（`true`/`false`） |
| `--use-t` | 是否使用 TC encoder（`true`/`false`） |
| `--use-c-exog` | CC 是否使用外生变量（`true`/`false`） |
| `--use-t-exog` | TC 是否使用外生变量（`true`/`false`） |
| `--fusion-method` | 协变量融合方法（`mlp`/`conv`/`cross_attention`/空字符串） |
| `--loss` | 损失函数（`MSE`/`MAE`/`Huber`） |
| `--norm` | 是否标准化数据（`true`/`false`） |

#### 评测参数

| 参数 | 说明 |
|------|------|
| `--evaluation-year` | 评测年份（覆盖 config） |
| `--evaluation-month` | 评测月份（覆盖 config） |
| `--capacity` | 装机容量 MW（覆盖 config 中的 capacity 字典） |
| `--horizon` | 预测长度（覆盖 config，默认 156 = 60点今日剩余 + 96点次日） |
| `--issue-hour` | 发布时刻（覆盖 config，默认 9，即每日9:00发布预测） |
| `--exclude-days` | 排除日期（逗号分隔，如 `2026-03-15,2026-03-16`），这些日期不参与评测 |

#### 模型保存/加载

| 参数 | 说明 |
|------|------|
| `--save-model` | 训练后将模型保存到此路径（如 `checkpoints/dag_model.ckpt`），供后续 `--load-model` 直接加载 |
| `--load-model` | 加载预训练模型路径（如 `checkpoints/dag_model.ckpt`），指定后跳过训练直接推理 |

#### 其他

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output-dir` | 数据集同目录/inference_results | 结果输出目录 |
| `--log-level` | `INFO` | 日志级别（`DEBUG`/`INFO`/`WARNING`/`ERROR`） |

#### 关键参数解释

- **`--load-model`**: 模式切换开关。未指定时走模式A（训练后推理）；指定后走模式B（加载模型直接推理，跳过训练）。
- **`--save-model`**: 仅在模式A下生效。训练完成后将模型权重+scaler+超参序列化为单个 `.ckpt` 文件，可分发给其他用户。模式B下被忽略。
- **`--evaluation-year` / `--evaluation-month`**: 指定评测月份。模型在评测月之前的数据上训练，对评测月逐日推理。例如 `--evaluation-year 2026 --evaluation-month 3` 表示用3月之前的数据训练，对3月逐日推理。
- **`--capacity`**: 装机容量（MW），用于准确率计算。若不指定，则按数据集名匹配 config 中的 capacity 字典（如 `ninghe_sr_0323_clean` → 80.4 MW）。
- **`--horizon`**: 预测长度。默认156点 = (24 - issue_hour) * 4 + evaluation_horizon。例如 issue_hour=9 时：60点(今日9:00~24:00) + 96点(次日全天) = 156点。
- **`--issue-hour`**: 发布时刻。每日该时刻发布预测，构造 history 时使用该时刻之前的全部数据。
- **`--exclude-days`**: 排除停机检修日。被排除的日期不参与评测，准确率均值也不包含这些日期。
- **`--endogenous-columns`**: 内生变量列。光伏默认 `power,sr`；风电默认 `power,ws`。内生变量只能使用历史值（防止数据泄露）。
- **`--model-hyper-params`**: 模型超参 JSON。关键超参包括 `seq_len`(输入长度，默认96)、`d_model`(隐藏维度)、`pred_dim`(输出维度，固定为1，只预测power)。

### 3.3 推理命令示例

```bash
# 模式A: 训练后推理（默认）
python tools/inference.py --data dataset.csv \
    --config-path config/business_day_ahead_config.json \
    --model-hyper-params '{"seq_len":96,"d_model":128}' \
    --endogenous-columns power,sr --target-column power \
    --evaluation-year 2026 --evaluation-month 3

# 模式A + 保存模型
python tools/inference.py --data dataset.csv \
    --config-path config/business_day_ahead_config.json \
    --save-model checkpoints/dag_model.ckpt \
    --evaluation-year 2026 --evaluation-month 2

# 模式B: 加载预训练模型直接推理
python tools/inference.py --data dataset.csv \
    --config-path config/business_day_ahead_config.json \
    --load-model checkpoints/dag_model.ckpt \
    --evaluation-year 2026 --evaluation-month 3

# 排除停机检修日
python tools/inference.py --data dataset.csv --config-path config.json \
    --model-hyper-params '{...}' \
    --evaluation-month 3 \
    --exclude-days 2026-03-15,2026-03-16

# 自定义模型架构开关
python tools/inference.py --data dataset.csv --config-path config.json \
    --use-c true --use-t true --use-c-exog true --use-t-exog true \
    --fusion-method mlp --loss Huber --norm true
```

---

## 4. 输入输出格式定义

### 4.1 输入 CSV 格式

`inference.py` 的输入 CSV（由 `data_processor.py` 生成）必须满足：

**列顺序：** `time, power, sr(光伏)/ws(风电), [外生变量...]`

**格式要求：**
- `time` 列：ISO 格式字符串（`YYYY-MM-DD HH:MM:SS`），必须为连续 15 分钟间隔
- `power` 列：功率值，目标预测变量
- `sr`/`ws` 列：内生变量，仅使用历史值
- 外生变量列：气象预报数据，可使用未来值

**示例（光伏）：**
```
time,power,sr,relative_humidity_isobaric_950,total_sky_direct_solar_radiation_at_surface_surface,...
2026-03-01 00:00:00,0.0,0.0,85.2,0.0,...
2026-03-01 00:15:00,0.0,0.0,85.5,0.0,...
...
```

**示例（风电）：**
```
time,power,ws,u_wind_component_surface_10_metre,v_wind_component_surface_10_metre,...
2026-03-01 00:00:00,12.5,6.8,3.2,-5.9,...
2026-03-01 00:15:00,12.8,7.1,3.5,-6.1,...
...
```

### 4.2 配置文件格式

配置文件（如 `config/business_day_ahead_config.json`）的关键字段：

```json
{
    "evaluation_config": {
        "strategy_args": {
            "horizon": 156,
            "evaluation_horizon": 96,
            "issue_hour": 9,
            "evaluation_year": 2026,
            "evaluation_month": 3,
            "capacity": {
                "ninghe_sr_0323_clean.csv": 80.4,
                "haojiaying_ws_0323_clean.csv": 50.0
            },
            "train_ratio_in_tv": 0.875,
            "seed": 2021,
            "deterministic": "efficient",
            "target_column": "power",
            "endogenous_columns": ["power", "sr"],
            "exclude_days": ["2026-03-01"]
        }
    }
}
```

**关键字段说明：**

| 字段 | 说明 |
|------|------|
| `horizon` | 预测长度，默认156点（=今日剩余 + 次日全天） |
| `evaluation_horizon` | 评测长度，默认96点（只取后96点即次日全天参与准确率计算） |
| `issue_hour` | 发布时刻，默认9（每日9:00发布预测） |
| `capacity` | 装机容量字典，按数据集名匹配（支持带/不带 .csv 后缀） |
| `train_ratio_in_tv` | 训练验证集划分比例，默认0.875（87.5%训练，12.5%验证） |
| `seed` | 随机种子，确保可复现 |
| `deterministic` | 随机确定性级别：`efficient`(轻量) / `full`(完全确定性) |
| `endogenous_columns` | 内生变量列（光伏: `["power","sr"]`，风电: `["power","ws"]`） |
| `exclude_days` | 排除日期列表，不参与评测 |

### 4.3 输出文件格式

推理完成后，在输出目录生成 4 类 CSV 文件：

#### 4.3.1 月度预测文件

**文件名：** `monthly_prediction_{series}_{YYYYMM}.csv`

**列：** `date, time, predicted_power, actual_power, error, daily_accuracy`

| 列 | 说明 |
|----|------|
| `date` | 日期（`YYYY-MM-DD`） |
| `time` | 时间戳（`YYYY-MM-DD HH:MM:SS`，15分钟间隔） |
| `predicted_power` | 预测功率值 |
| `actual_power` | 实际功率值 |
| `error` | 误差（actual - predicted） |
| `daily_accuracy` | 当日准确率 |

#### 4.3.2 日度准确率文件

**文件名：** `daily_accuracy_{series}_{YYYYMM}.csv`

**列：** `date, daily_accuracy`

末行：`MONTHLY_MEAN ({month}_accuracy_mean), {月度均值}`

#### 4.3.3 月度统计文件

**文件名：** `monthly_summary_{series}_{YYYYMM}.csv`

**列：** `metric, value`

包含：数据集名、评测年月、评测天数、排除天数、排除日期列表、装机容量、月度准确率均值、最高/最低/标准差日准确率。

#### 4.3.4 逐点预测详情文件

**文件名：** `prediction_detail_{series}.csv`

**列：** `date, time, actual_{col}, predicted_{col}, error, daily_accuracy`

含所有内生变量的逐点预测对比。

### 4.4 准确率计算口径

准确率计算与 `business_day_ahead.py` 完全一致：

1. **每日准确率** = `1 - weighted_rmse / capacity`
   - `weighted_rmse = sqrt(sum(error² × weights))`
   - `weights = |error| / sum(|error|)`（误差绝对值占比作为权重）
2. **月度均值** = 所有日准确率的算术平均

---

## 5. 完整使用示例

### 5.1 从原始数据到推理的完整流程

```bash
# ===== 步骤1: 数据处理 =====
# 光伏
python tools/data_processor.py --plant-type solar \
    --ec-data 玖天数据集处理/ninghe.csv \
    --power-data 玖天数据集处理/宁河光伏电站power-sr.csv \
    --work-dir 玖天数据集处理/processed_solar \
    --time-start 202407030000 --time-end 202607312345 \
    --center-grid 39_200_117_400 \
    --test-cutoff 2026-03-01

# 风电
python tools/data_processor.py --plant-type wind \
    --ec-data 玖天数据集处理/haojiaying.csv \
    --power-data 玖天数据集处理/郝家营风电场二期power-ws.csv \
    --work-dir 玖天数据集处理/processed_wind \
    --time-start 202407030000 --time-end 202607312345 \
    --center-grid 41_400_114_900

# ===== 步骤2: 模型推理 =====
# 模式A: 训练后推理
python tools/inference.py \
    --data 玖天数据集处理/processed_solar/final/ninghe_sr_0323_clean.csv \
    --config-path config/business_day_ahead_config.json \
    --model-hyper-params '{"seq_len":96,"d_model":128,"pred_dim":1}' \
    --endogenous-columns power,sr \
    --evaluation-year 2026 --evaluation-month 3

# 模式B: 加载预训练模型推理（无法自行训练时）
python tools/inference.py \
    --data 玖天数据集处理/processed_solar/final/ninghe_sr_0323_clean.csv \
    --config-path config/business_day_ahead_config.json \
    --load-model checkpoints/dag_model.ckpt \
    --evaluation-year 2026 --evaluation-month 3
```

### 5.2 两步流程：训练保存 + 加载推理

```bash
# 步骤1: 用户A训练并保存模型
python tools/inference.py --data train_data.csv \
    --config-path config/business_day_ahead_config.json \
    --save-model checkpoints/dag_ninghe.ckpt \
    --evaluation-year 2026 --evaluation-month 2

# 步骤2: 用户B加载模型对3月数据推理
python tools/inference.py --data eval_data.csv \
    --config-path config/business_day_ahead_config.json \
    --load-model checkpoints/dag_ninghe.ckpt \
    --evaluation-year 2026 --evaluation-month 3
```

### 5.3 编程式 API 调用

`inference.py` 文件末尾提供了两个样例函数，可在 Python 代码中直接调用：

- `example_train_save_load_inference()`: 训练 → 保存模型 → 加载模型 → 推理（完整两步流程）
- `example_load_pretrained_only()`: 直接加载预训练模型推理（无法自行训练的场景）

调用方式：

```python
from tools.inference import example_train_save_load_inference, example_load_pretrained_only

# 完整流程
example_train_save_load_inference()

# 仅加载推理
example_load_pretrained_only()
```

---

## 附：光伏与风电差异对照

| 维度 | 光伏 (solar) | 风电 (wind) |
|------|-------------|------------|
| 内生变量 | `power, sr` | `power, ws` |
| 默认特征数 | 7列 | ~130列 |
| 数据处理步骤3 | 跳过（无梯度计算） | 执行（水平梯度+垂直风切变） |
| 数据处理步骤4 | 简单合并 | 合并中心点+周边网格点风速 |
| 异常0值检测 | power/sr: 全天为0才算异常；太阳辐射列: 白天(07:00~19:00)为0才算异常 | 无特殊检测 |
| capacity | ninghe: 80.4 MW | haojiaying: 50.0 MW |
