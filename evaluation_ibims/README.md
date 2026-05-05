# Prior-Depth-Anything iBims 官方评估

`evaluation_ibims/` 是当前项目专用的 iBims 官方评估适配目录。它只消费已有 synthetic raw depth manifest，不包含也不调用 raw depth 生成/校验脚本。

## 前置条件

默认 iBims 数据集目录：

```text
data/ibims1
```

运行前需要已经存在 synthetic manifest：

```text
data/ibims1/ibims1_synthetic_raw_depth/manifests/ibims_easy.jsonl
data/ibims1/ibims1_synthetic_raw_depth/manifests/ibims_medium.jsonl
data/ibims1/ibims1_synthetic_raw_depth/manifests/ibims_hard.jsonl
data/ibims1/ibims1_synthetic_raw_depth/manifests/ibims_extreme.jsonl
```

完整官方评估还需要数据集自带文件：

```text
data/ibims1/imagelist.txt
data/ibims1/ibims1_core_mat/
data/ibims1/evaluation_scripts/evaluate_ibims.py
```

项目推理依赖沿用当前仓库环境。官方 evaluator 额外依赖 `scipy`、`scikit-image`、`scikit-learn`。

## 一站式运行

本地默认优先使用 `.venv/bin/python`，服务器可通过 `PYTHON_BIN` 指向 conda 环境中的 Python：

```bash
./evaluation_ibims/run_all.sh
```

默认权重与 `evaluation/run_eval.sh` 保持一致：

```text
ckpts/prior_depth_anything_vitb_1_1.pth
ckpts/depth_anything_v2_vitl.pth
```

也可以显式传入两个 checkpoint，位置参数顺序为 Prior-Depth-Anything checkpoint、Depth Anything V2 checkpoint：

```bash
./evaluation_ibims/run_all.sh \
  /path/to/prior_depth_anything_vitb_1_1.pth \
  /path/to/depth_anything_v2_vitl.pth
```

小样本 smoke：

```bash
MAX_SAMPLES=1 ./evaluation_ibims/run_all.sh \
  /path/to/prior_depth_anything_vitb_1_1.pth \
  /path/to/depth_anything_v2_vitl.pth \
  --levels easy \
  --skip-eval
```

使用 Hugging Face 自动下载权重：

```bash
./evaluation_ibims/run_all.sh auto auto --levels easy --max-samples 1 --skip-eval
```

## Python 入口

```bash
.venv/bin/python evaluation_ibims/run_all.py \
  --priorda-ckpt ckpts/prior_depth_anything_vitb_1_1.pth \
  --mde-ckpt ckpts/depth_anything_v2_vitl.pth \
  --ibims-root data/ibims1 \
  --levels easy medium hard extreme \
  --batch-size 1 \
  --device cuda:0
```

常用参数：

```text
--run-dir <dir>              指定完整输出目录
--output-dir <dir>           指定输出根目录，默认 evaluation_ibims/output
--max-samples <N>            每个 difficulty 只跑前 N 个样本；0 表示全量
--skip-infer                 跳过推理，使用 --run-dir 下已有 predictions
--skip-eval                  跳过官方评估，只生成 MAT prediction
--pattern <pattern>          使用 Prior-Depth-Anything 稀疏采样 pattern
--coarse-only <true|false>   只运行 coarse stage
--clamp-to-depth-range true  保存前裁剪到 manifest depth-range
```

## 输出结构

默认输出目录：

```text
evaluation_ibims/output/ibims_<mde_stem>__<priorda_stem>_<YYYYMMDD_HHMMSS>/
```

主要内容：

```text
predictions/<level>/<sample>_results.mat
predictions/<level>/infer_args.json
official_eval/<level>/workspace/
official_eval/<level>/official_eval_stdout.txt
eval_summary.csv
eval_summary.json
```

每个 prediction MAT 包含变量 `pred_depths`：

```text
shape: 480x640
dtype: float32
unit: meter
invalid prediction: NaN
```

## 推理处理约定

- RGB 直接以路径传给 `PriorDepthAnything.infer_one_sample`，读取逻辑沿用项目自身 sampler。
- raw depth 使用 manifest 中的 `depth_scale`，默认 `65535 / 50`，转换为 meter。
- raw depth 中非有限值、`<= 0`、超过 manifest `depth-range` 上限的点会置为 `0`。
- 模型固定为 `prior_depth_anything.PriorDepthAnything`，输出视为 metric depth，不做 disparity/inverse-depth 转换，也不做 eval-time alignment。
- 即使 `--batch-size > 1`，当前模型接口仍会逐样本调用 `infer_one_sample`；建议服务器上使用 CUDA。
