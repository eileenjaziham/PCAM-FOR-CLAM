# CLAM on PCam Reproduction

这个仓库基于原始 CLAM 项目，新增了 PCam 复现流程。当前实现采用最小侵入改法：不改 `Generic_MIL_Dataset` 的核心读取逻辑，继续使用 CLAM 原生的 `csv + pt_files` 训练接口。

## What Changed

- 新增 `task_pcam`，接入训练/评估/切分入口。
- 新增 `utils/task_utils.py`，统一管理 task 配置。
- 新增 `preprocess_pcam.py`，把 PCam 的 `meta + x/y.h5(.gz)` 转成 bag 级 `.pt` 与 CSV。
- `preprocess_pcam.py` 支持 sub-bag 切分：每个 WSI 可拆成多个固定 patch 数量的 bag。
- 默认特征提取 encoder 已切到 `uni_v1`（相关脚本见 `extract_features.py`、`extract_features_fp.py`、`preprocess_pcam.py`）。

## Environment

- Python 3.10
- PyTorch + torchvision + timm
- h5py, pandas, numpy, tqdm, pillow

如果出现 `NumPy 2.x` 与 torch/torchvision ABI 报错，降级到：

```bash
pip install --force-reinstall numpy==1.26.4
```

如果 `torch.version.cuda` 与驱动不匹配，PyTorch 会退回 CPU，UNI 预处理会非常慢。

## Data Preparation

目录结构示例：

- 原始数据：`pcam/train`, `pcam/valid`, `pcam/test`
- 输出特征：`pcam_features_subbag/pt_files/*.pt`
- 输出 CSV：`dataset_csv/pcam_subbag_clean.csv`

只跑 `valid` 快速验证：

```bash
PYTHONPATH=. python preprocess_pcam.py \
  --pcam_dir pcam \
  --feat_dir pcam_features_subbag \
  --csv_out dataset_csv/pcam_subbag_clean.csv \
  --model_name uni_v1 \
  --weights_path /root/yy/CLAM-master/pytorch_model.bin \
  --batch_size 32 \
  --splits valid \
  --patches_per_bag 256 \
  --min_patches_per_bag 128
```

全量生成：

```bash
PYTHONPATH=. python preprocess_pcam.py \
  --pcam_dir pcam \
  --feat_dir pcam_features_subbag \
  --csv_out dataset_csv/pcam_subbag_clean.csv \
  --model_name uni_v1 \
  --weights_path /root/yy/CLAM-master/pytorch_model.bin \
  --batch_size 32 \
  --splits train valid test \
  --patches_per_bag 256 \
  --min_patches_per_bag 128
```

数据完整性检查：

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv('dataset_csv/pcam_subbag_clean.csv')
print(df['slide_id'].str.split('__').str[0].value_counts())
print(df['label'].value_counts())
print('total:', len(df))
PY
```

## Task Config

`task_pcam` 默认读取：

- `dataset_csv/pcam_clean.csv`
- `pcam_features/pt_files`

如果你要跑 sub-bag 结果，请先修改 `utils/task_utils.py` 中 `task_pcam` 的两项配置：

- `csv_path` -> `dataset_csv/pcam_subbag_clean.csv`
- `features_subdir` -> `pcam_features_subbag`

## Train & Eval

生成切分：

```bash
PYTHONPATH=. python create_splits_seq.py \
  --task task_pcam \
  --label_frac 1.0 \
  --k 5 \
  --val_frac 0.1 \
  --test_frac 0.1
```

训练 MIL：

```bash
PYTHONPATH=. python main.py \
  --task task_pcam \
  --data_root_dir . \
  --exp_code pcam_subbag_mil \
  --k 5 \
  --max_epochs 50 \
  --lr 1e-4 \
  --model_type mil \
  --embed_dim 1024 \
  --label_frac 1.0 \
  --drop_out 0.25 \
  --weighted_sample \
  --seed 1
```

评估 MIL：

```bash
PYTHONPATH=. python eval.py \
  --task task_pcam \
  --data_root_dir . \
  --results_dir results \
  --models_exp_code pcam_subbag_mil_s1 \
  --save_exp_code pcam_subbag_mil_eval \
  --k 5 \
  --model_type mil \
  --embed_dim 1024
```

训练 CLAM-SB：

```bash
PYTHONPATH=. python main.py \
  --task task_pcam \
  --data_root_dir . \
  --exp_code pcam_subbag_clam \
  --k 5 \
  --max_epochs 50 \
  --lr 1e-4 \
  --model_type clam_sb \
  --embed_dim 1024 \
  --label_frac 1.0 \
  --drop_out 0.25 \
  --bag_loss ce \
  --inst_loss ce \
  --bag_weight 0.7 \
  --B 8 \
  --weighted_sample \
  --seed 1
```

评估 CLAM-SB：

```bash
PYTHONPATH=. python eval.py \
  --task task_pcam \
  --data_root_dir . \
  --results_dir results \
  --models_exp_code pcam_subbag_clam_s1 \
  --save_exp_code pcam_subbag_clam_eval \
  --k 5 \
  --model_type clam_sb \
  --embed_dim 1024
```

## Notes

- `k-fold` 训练时，epoch 会在每个 fold 内从 0 重新计数。
- 当前早停逻辑默认 `stop_epoch=50`，`max_epochs=50` 时通常不会触发早停。
- 若出现 `FileNotFoundError`，通常是 `task_utils.py` 的 CSV/特征目录与实际生成路径不一致，或 split 文件仍是旧 slide_id。

## Before Pushing to GitHub
