# Two-Stage Fusion for PET-Based Continual Learning

Code to train and evaluate the two-stage fusion method described in our paper.
The method augments an HRM-PET learner with a frozen random-projection ridge
head. At inference, its scores contribute to task routing and then to gated
class-score fusion. The pretrained backbone remains frozen; each task learns
parameter-efficient components, and the ridge head uses aggregate statistics
rather than stored training images.

This repository contains **code and reproduction instructions only**. It does
not contain the manuscript source, datasets, pretrained weights, training
checkpoints, or private run logs.

## Code map

| Path | Role |
| --- | --- |
| `main.py`, `configs/` | Experiment entry point and dataset configurations |
| `trainers/`, `engines/` | PET training, evaluation, task/class fusion, RP ridge head |
| `vits/`, `peft/` | Frozen ViT and parameter-efficient modules |
| `continual_datasets/`, `datasets.py` | Dataset and task-order construction |
| `training_scripts/` | Training and evaluation commands |
| `tools/` | Data preparation, result verification, RanPAC comparisons |
| `tests/` | Unit and regression checks |

The fusion implementation is in
`engines/hrm_lora_wtp_and_tap_engine.py`; the analytic classifier is in
`engines/random_projection_head.py`.

## Environment and data

Use Python 3.12, a CUDA-capable GPU, and install `requirements.txt` in a
dedicated environment. The reported runs used an RTX 4090. Install a PyTorch
build compatible with your CUDA driver before the remaining requirements.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python tools/prepare_datasets.py --which all --root /path/to/datasets
```

The data-preparation command may download data and move ImageNet-A images
into a fixed train/test split. Inspect an existing dataset directory before
running it; use `--which imagenet-a` or `--which fivedatasets` when only
one benchmark needs preparation.

The evaluated benchmarks are ImageNet-R, CIFAR-100, ImageNet-A, and
5-Datasets (SVHN, MNIST, CIFAR-10, NotMNIST, Fashion-MNIST). CUB-200 is used
in the component analysis. Review the dataset licenses and obtain the source
data separately. `tools/imagenet_a_from_parquet.py` supports the
ImageNet-A/ImageNet-R preparation path.

The main matched runs use an ImageNet-21K-pretrained ViT-B/16. The backbone
grid additionally uses the reported MoCo-v3, iBOT-1K, iBOT-21K, and DINO
checkpoints; place them under `checkpoints/` using the filenames expected by
`vits/hrm_lora_vision_transformer.py`. Pretrained weights are not included
here. Keep the backbone, task order, seed, and training budget fixed when
comparing methods.

## Train and evaluate

Train the task-identity module and LoRA pool for one dataset/seed:

```bash
DATASETS_ROOT=/path/to/datasets OUTPUT_ROOT=/path/to/output \
DATASET=imr SEED=42 bash training_scripts/train_any_4090.sh
```

`DATASET` accepts `imr`, `cifar100`, `cub200`, `ima`, or
`fivedatasets`. The main matched study uses seeds 42–45. To evaluate the
reported full method (routing weight `w=0.7`, class-fusion weight
`beta=0.5`) after both training stages finish:

```bash
DATASETS_ROOT=/path/to/datasets OUTPUT_ROOT=/path/to/output \
DATASET=Split-Imagenet-R CONFIG=imr_lora SEED=42 NUM_TASKS=10 \
TII_DIR=/path/to/output/imr_tii_original_10tasks_seed42 \
RP_FUSE=1 RP_FUSE_DRM=1 RP_CLS_MIN=1 CALIBRATE=0 \
RP_FUSE_W=0.7 RP_CLS_GATE=margin RP_CLS_W=0.5 \
bash training_scripts/eval_rp_head_any_4090.sh \
  /path/to/output/imr_lora_rank8_baseline_10tasks_seed42
```

The conventional evaluation scripts cover ImageNet-R, CIFAR-100, and
CUB-200 without the RP fusion. Do not compare these matched
Sup-21K results with the separately trained, dataset-specific RanPAC
checkpoints as though they shared a backbone or training budget.

The scripts under `tools/verify_paper_*.py` contain the checks used for
the reported log sets, paired statistics, backbone grid, and ablations.
Several historical checks pin the evaluator source hash from the original
run and therefore reject this cleaned source snapshot; do not bypass those
checks or present an unverified rerun as the reported result. The
`tools/run_ranpac_original*.py` and `tools/run_ranpac_matched_extra.py`
scripts support the distinct RanPAC protocol and its matched comparisons.
Use `--help` on each tool for its arguments. Raw experiment outputs remain
outside the repository.

## Checks

```bash
.venv/bin/python -m pytest tests/ -q
```

The method builds on [HRM-PET](https://github.com/wei-cheng777/HRM-PET).
The random-projection ridge head follows
[RanPAC](https://github.com/McDonnell-Research-Lab/RanPAC). Please cite
those works alongside our paper when using their respective components.
