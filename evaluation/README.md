# Prior-Depth-Anything 的 HAMMER / ClearPose / DREDS 评估

该目录是当前项目的轻量评估导出入口，结构与 `eval_pipeline_cdm` 对齐，但模型加载固定保留本仓库的 `prior_depth_anything.PriorDepthAnything`。旧的单入口 `run_eval.sh` 已拆分为三个数据集专属 wrapper。

## 文件结构

```text
evaluation/
├── dataset.py
├── infer.py
├── eval.py
├── run_hammer.sh
├── run_clearpose.sh
├── run_dreds.sh
├── requirements.txt
└── utils/
    ├── img_utils.py
    └── metric.py
```

链路边界：

1. `infer.py` 读取 JSONL 和 raw depth，调用 `PriorDepthAnything`，逐样本写出 `predictions/*.npy`。
2. `eval.py` 从 `predictions/` 读取预测并计算指标；为了兼容旧结果，也会 fallback 到输出根目录查找 `.npy`。
3. `run_*.sh` 负责选择数据集、组织输出目录、顺序调用推理和评估，并可选清理逐样本 `.npy`。

## 数据格式

HAMMER JSONL 每行是一个样本，字段为：

```text
rgb
d435_depth / l515_depth / tof_depth
depth
depth-range
```

ClearPose JSONL 每行是一个序列 manifest，字段为：

```text
rgb
rgb-suffix
raw_depth-suffix
depth-suffix
depth-range
```

DREDS 使用与 CDM 模板一致的 sequence JSONL，字段同 ClearPose。DREDS 的 raw / GT depth 是 EXR 浮点深度，单位已经是 meter，因此 `DREDSDataset.depth_scale=1.0`；HAMMER 和 ClearPose 使用 `depth_scale=1000.0`。

样本命名约定：

```text
HAMMER:    scene#frame-stem.npy
ClearPose: dir1#dir2#frame-stem.npy
DREDS:     dir1#dir2#frame-stem.npy
```

## 运行方式

先安装依赖：

```bash
pip install -r evaluation/requirements.txt
```

默认 checkpoint：

```text
ckpts/depth_anything_v2_vitl.pth
ckpts/prior_depth_anything_vitb_1_1.pth
```

也可以把 checkpoint 参数设为 `auto`、`none` 或 `null`，让 `PriorDepthAnything` 从 Hugging Face 下载权重。

### HAMMER

```bash
DATASET_PATH=/path/to/HAMMER/test_filled_d435.jsonl \
OUTPUT_DIR=/tmp/priorda_hammer_eval \
BATCH_SIZE=1 \
NUM_WORKERS=0 \
bash evaluation/run_hammer.sh /path/to/prior_depth_anything_vitb_1_1.pth /path/to/depth_anything_v2_vitl.pth d435 vitl vitb 1.1 false
```

参数：

```text
bash evaluation/run_hammer.sh [priorda_ckpt] [mde_ckpt] [camera_type=d435] [frozen_size=vitl] [conditioned_size=vitb] [version=1.1] [cleanup_npy=false]
```

`camera_type` 支持 `d435`、`l515`、`tof`。

### ClearPose

ClearPose 固定按 `raw-type=d435`：

```bash
DATASET_PATH=/path/to/clearpose/test.jsonl \
OUTPUT_DIR=/tmp/priorda_clearpose_eval \
bash evaluation/run_clearpose.sh /path/to/prior_depth_anything_vitb_1_1.pth /path/to/depth_anything_v2_vitl.pth vitl vitb 1.1 false
```

参数：

```text
bash evaluation/run_clearpose.sh [priorda_ckpt] [mde_ckpt] [frozen_size=vitl] [conditioned_size=vitb] [version=1.1] [cleanup_npy=false]
```

### DREDS

DREDS route 支持 `catknown`、`catnovel`、`all`：

```bash
DREDS_KNOWN_JSONL=/path/to/DREDS/test_std_catknown.jsonl \
DREDS_NOVEL_JSONL=/path/to/DREDS/test_std_catnovel.jsonl \
OUTPUT_ROOT=/tmp/priorda_dreds_eval \
SAVE_VIS=false \
bash evaluation/run_dreds.sh /path/to/prior_depth_anything_vitb_1_1.pth /path/to/depth_anything_v2_vitl.pth all vitl vitb 1.1 false
```

参数：

```text
bash evaluation/run_dreds.sh [priorda_ckpt] [mde_ckpt] [variant=all] [frozen_size=vitl] [conditioned_size=vitb] [version=1.1] [cleanup_npy=false]
```

说明：

- `variant=catknown` 使用 `DREDS_KNOWN_JSONL`。
- `variant=catnovel` 使用 `DREDS_NOVEL_JSONL`。
- `variant=all` 会顺序运行 catknown 和 catnovel；此时请使用 `OUTPUT_ROOT`，不要使用单目录 `OUTPUT_DIR`。
- `run_dreds.sh` 会在 Python 启动前设置 `OPENCV_IO_ENABLE_OPENEXR=1`。
- DREDS 的 `raw-type=d435` 仅用于满足共享 CLI 参数，dataset loader 不按 raw type 分支。

## 常用环境变量

```text
PRIORDA_CKPT          Prior-Depth-Anything checkpoint
MDE_CKPT              Depth Anything V2 checkpoint
DATASET_PATH          HAMMER / ClearPose JSONL 路径
DREDS_KNOWN_JSONL     DREDS catknown JSONL 路径
DREDS_NOVEL_JSONL     DREDS catnovel JSONL 路径
OUTPUT_DIR            单数据集或单 DREDS variant 输出目录
OUTPUT_ROOT           DREDS all 模式输出根目录
BATCH_SIZE            推理 batch size，当前模型仍逐样本调用
NUM_WORKERS           推理 DataLoader worker 数
MAX_SAMPLES           最大样本数，0 表示全部
DEVICE                Torch device，例如 cuda:0
PATTERN               Prior-Depth-Anything sparse sampling pattern
SAVE_VIS              true 时保存可视化，HAMMER/ClearPose 默认 true，DREDS 默认 false
COARSE_ONLY           true 时只使用 coarse stage
DOUBLE_GLOBAL         true 时启用 double-global conditioning
PRIOR_COVER           sparse pattern 下保留所有 prior 像素
DOWN_FILL_MODE        downscale_* pattern 的填充方式：linear/global/knn
CLAMP_TO_DEPTH_RANGE  true 时保存前裁剪到 dataset depth-range
PYTHON_BIN            Python 可执行文件，默认 python，找不到时回退 python3
```

## 输出结构

未设置 `OUTPUT_DIR` 时，HAMMER / ClearPose 默认写到 PriorDepthAnything checkpoint 同级目录；DREDS `all` 默认在 `OUTPUT_ROOT` 下为两个 variant 分别创建目录。

```text
<output_dir>/
├── args.json
├── eval_args.json
├── predictions/
│   └── *.npy
├── visualizations/
│   └── *_promptda_vis.jpg
├── all_metrics_<timestamp>_False.csv
└── mean_metrics_<timestamp>_False.json
```

如果 `cleanup_npy=true`，评估结束后会删除 `predictions/*.npy`，指标文件和参数 JSON 会保留。

## 关键约定

- 模型类、checkpoint 格式、Depth Anything V2 transform 和 `PriorDepthAnything.infer_one_sample()` 不随 CDM 模板改动。
- `infer.py` 默认把 `.npy` 写入 `predictions/`，可视化写入 `visualizations/`；`--prediction-dir` 和 `--visualization-dir` 可覆盖默认路径。
- `eval.py` 对 HAMMER / ClearPose 要求 prediction 与 GT shape 一致；DREDS 如果 shape 不一致，会用 nearest resize 将 prediction 对齐到 GT。
- 当前项目官方推理接口以单图为主，即使 `BATCH_SIZE > 1`，`infer.py` 也会逐样本循环推理。
- completion 阶段依赖 `torch_cluster` KNN，完整推理建议使用 CUDA 环境。
