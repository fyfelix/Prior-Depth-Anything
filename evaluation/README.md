# Prior-Depth-Anything 的 HAMMER 评估

该目录是为当前外部项目适配后的 HAMMER 评估入口。整体结构复用原导出 pipeline，但 `infer.py` 固定加载本仓库的 `prior_depth_anything.PriorDepthAnything` 模型。

## 适配模型

- 模型：`PriorDepthAnything`
- 默认架构：`version=1.1`，`frozen_model_size=vitb`，`conditioned_model_size=vitb`
- 输入类型：RGB-D / depth completion
- HAMMER raw depth 来源：由 `--raw-type` / `camera_type` 选择，可选 `d435`、`l515`、`tof`
- 输出：每个 sample 保存一个 `HxW float32` 的 metric depth `.npy`，单位为 meter

脚本读取 HAMMER raw depth PNG 时按毫米处理，并通过 `depth_scale=1000` 转换为米。`.npy` 格式的 raw depth 默认已经是米。当前模型输出的是 metric depth，因此默认不启用 eval-time alignment。

## 数据布局

默认数据集路径：

```bash
data/HAMMER/test.jsonl
```

JSONL 仍由 `HAMMERDataset` 读取，并需要包含以下字段：

```text
rgb
d435_depth / l515_depth / tof_depth
depth
depth-range
```

## 运行方式

从仓库根目录运行：

```bash
pip install -r evaluation/requirements.txt
./evaluation/run_eval.sh
```

默认情况下，脚本会从 `ckpts/` 读取以下两个本地权重文件：

```text
ckpts/depth_anything_v2_vitb.pth
ckpts/prior_depth_anything_vitb_1_1.pth
```

默认数据集为 `data/HAMMER/test.jsonl`，默认输出到 `evaluation/output`，默认保存 `*_pred_depth.png` 可视化，并默认保留生成的 `.npy` 预测文件。

若 `depth_anything_v2_vitb.pth` 存在其他位置，请设置 `MDE_DIR`；若需要使用 Hugging Face 自动下载权重，可把第一个参数和 `MDE_DIR` 都设为 `auto`：

```bash
MDE_DIR=/path/to/mde_dir \
DATASET_PATH=/path/to/HAMMER/test.jsonl \
OUTPUT_DIR=/tmp/priorda_hammer \
./evaluation/run_eval.sh /path/to/priorda_ckpt_dir d435 vitb vitb 1.1 false

MDE_DIR=auto ./evaluation/run_eval.sh auto
```

参数格式：

```text
./evaluation/run_eval.sh [ckpt_dir=ckpts] [camera_type=d435] [frozen_size=vitb] [conditioned_size=vitb] [version=1.1] [cleanup_npy=false]
```

常用环境变量：`DATASET_PATH`、`OUTPUT_DIR`、`MDE_DIR`、`BATCH_SIZE`、`NUM_WORKERS`、`DEVICE`、`PATTERN`、`SAVE_VIS`、`COARSE_ONLY`、`PRIOR_COVER`、`DOWN_FILL_MODE`、`CLAMP_TO_DEPTH_RANGE`、`PYTHON_BIN`。如果未设置 `PYTHON_BIN`，脚本会先尝试 `python`，再回退到 `python3`。

## 注意事项与限制

`evaluation/eval.py`、`dataset.py` 和 `utils/metric.py` 保留原 HAMMER 指标链路。由于当前项目官方推理接口以单图为主，即使 `BATCH_SIZE > 1`，`infer.py` 也会逐样本循环推理；建议使用 `BATCH_SIZE=1`。completion 阶段依赖 `torch_cluster` KNN，强烈建议使用 CUDA 环境。若未设置 `PATTERN`，脚本会直接把所选 raw depth 作为 prior；若设置稀疏采样 pattern，则沿用 Prior-Depth-Anything 自身的采样规则。
