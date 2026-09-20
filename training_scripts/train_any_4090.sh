#!/usr/bin/env bash
set -euo pipefail

# Train an HRM-PET baseline (TII, then LoRA) for any dataset / backbone / seed.
#
# The repo shipped one training script, hard-wired to ImageNet-R with Sup-21K
# and the original author's absolute paths. Matching the original paper's
# coverage means 4 datasets x 5 pre-trained settings (README.md carries their
# abstract), so the script has to be parameterised before any of that is
# possible. Their seed count is not stated anywhere we can check; the only
# training script they shipped loops over seed 42 alone.
#
# With no arguments beyond DATASET and SEED it reproduces the directory names
# the evaluation scripts already expect, so existing checkpoints stay valid.
#
# Usage:
#   DATASET=imr SEED=42 bash training_scripts/train_any_4090.sh
#   DATASET=imr SEED=42 BACKBONE=vit_base_patch16_224_dino BTAG=dino \
#       bash training_scripts/train_any_4090.sh
#
# Knobs: DATASET BACKBONE BTAG SEED LORA_RANK TII_EPOCHS LORA_EPOCHS STAGE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
DATA_PATH="${DATA_PATH:-${DATASETS_ROOT}}"

DATASET="${DATASET:-}"
SEED="${SEED:-42}"
BACKBONE="${BACKBONE:-vit_base_patch16_224}"
# Empty BTAG keeps the historical directory names for the Sup-21K runs. Any
# other backbone must set it, or its checkpoints would overwrite those.
BTAG="${BTAG:-}"
LORA_RANK="${LORA_RANK:-8}"
TII_EPOCHS="${TII_EPOCHS:-20}"
LORA_EPOCHS="${LORA_EPOCHS:-50}"
CRCT_EPOCHS="${CRCT_EPOCHS:-30}"
NUM_TASKS="${NUM_TASKS:-10}"
STAGE="${STAGE:-both}"          # tii | lora | both

case "${DATASET}" in
  imr)          DS=Split-Imagenet-R; CFG_TII=imr_hideprompt_5e;           CFG_LORA=imr_lora ;;
  cifar100)     DS=Split-CIFAR100;   CFG_TII=cifar100_hideprompt_5e;      CFG_LORA=cifar100_lora ;;
  cub200)       DS=Split-CUB200;     CFG_TII=imr_hideprompt_5e;           CFG_LORA=imr_lora ;;
  ima)          DS=Split-Imagenet-A; CFG_TII=ima_hideprompt_5e;           CFG_LORA=ima_lora ;;
  fivedatasets) DS=5-datasets;       CFG_TII=five_datasets_hideprompt_5e; CFG_LORA=five_datasets_lora
                NUM_TASKS="${NUM_TASKS_OVERRIDE:-5}" ;;
  *) echo "Set DATASET to one of: imr cifar100 cub200 ima fivedatasets" >&2; exit 64 ;;
esac

SUFFIX=""; [[ -n "${BTAG}" ]] && SUFFIX="_${BTAG}"
TII_DIR="${OUTPUT_ROOT}/${DATASET}${SUFFIX}_tii_original_${NUM_TASKS}tasks_seed${SEED}"
LORA_DIR="${OUTPUT_ROOT}/${DATASET}${SUFFIX}_lora_rank${LORA_RANK}_baseline_${NUM_TASKS}tasks_seed${SEED}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Python not found: ${PYTHON_BIN}" >&2; exit 1; }
cd "${REPO_ROOT}"
echo "dataset=${DS} backbone=${BACKBONE} seed=${SEED} rank=${LORA_RANK} tasks=${NUM_TASKS}"
echo "  TII  -> ${TII_DIR}"
echo "  LoRA -> ${LORA_DIR}"

complete() { local d="$1" t; for t in $(seq 1 "${NUM_TASKS}"); do
  [[ -s "${d}/checkpoint/task${t}_checkpoint.pth" ]] || return 1; done; }

validate_saved_run() {
  "${PYTHON_BIN}" - "$1" "${DS}" "${SEED}" "${NUM_TASKS}" "${BACKBONE}" <<'PY'
import sys
from pathlib import Path
import torch
root, dataset, seed, tasks, model = sys.argv[1:]
expected = dict(dataset=dataset, seed=int(seed), num_tasks=int(tasks), model=model)
for stage in range(1, int(tasks) + 1):
    path = Path(root) / 'checkpoint' / f'task{stage}_checkpoint.pth'
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    saved = checkpoint.get('args')
    for key, value in expected.items():
        actual = saved.get(key) if isinstance(saved, dict) else getattr(saved, key, None)
        if actual != value:
            raise SystemExit(f'CHECKPOINT_MISMATCH: {path}: {key}={actual!r}, expected {value!r}. '
                             'No checkpoint was overwritten. Investigate the existing run.')
print('Existing checkpoint configuration: PASS', root)
PY
}

if [[ "${STAGE}" == "tii" || "${STAGE}" == "both" ]]; then
  if complete "${TII_DIR}"; then
    validate_saved_run "${TII_DIR}"
    echo "TII already complete, skipping."
  else
    START="$(date +%s)"
    PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
      --nproc_per_node=1 --master_port="${MASTER_PORT:-29500}" \
      main.py "${CFG_TII}" \
      --model "${BACKBONE}" --original_model "${BACKBONE}" \
      --batch-size "${TII_BATCH:-128}" --epochs "${TII_EPOCHS}" \
      --ca_storage_efficient_method covariance \
      --data-path "${DATA_PATH}" --lr 0.0005 --ca_lr 0.005 \
      --crct_epochs "${CRCT_EPOCHS}" --seed "${SEED}" \
      --num_tasks "${NUM_TASKS}" --dataset "${DS}" \
      --train_inference_task_only --output_dir "${TII_DIR}"
    printf 'TII wall time seconds: %s\n' "$(( $(date +%s) - START ))"
  fi
fi

if [[ "${STAGE}" == "lora" || "${STAGE}" == "both" ]]; then
  complete "${TII_DIR}" || { echo "TII incomplete; LoRA needs it" >&2; exit 2; }
  validate_saved_run "${TII_DIR}"
  if complete "${LORA_DIR}"; then
    validate_saved_run "${LORA_DIR}"
    echo "LoRA already complete, skipping."
  else
    START="$(date +%s)"
    PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
      --nproc_per_node=1 --master_port="${MASTER_PORT_LORA:-29513}" \
      main.py "${CFG_LORA}" \
      --model "${BACKBONE}" --original_model "${BACKBONE}" \
      --batch-size "${LORA_BATCH:-24}" --epochs "${LORA_EPOCHS}" \
      --data-path "${DATA_PATH}" --ca_lr 0.005 \
      --crct_epochs "${CRCT_EPOCHS}" --seed "${SEED}" \
      --lr 0.03 --con 0.2 --lora_rank "${LORA_RANK}" \
      --En gen --tau -10 --K 5 --sched cosine --dataset "${DS}" \
      --lora_momentum 0.4 --lora_type hide \
      --num_tasks "${NUM_TASKS}" \
      --trained_original_model "${TII_DIR}" --output_dir "${LORA_DIR}"
    printf 'LoRA wall time seconds: %s\n' "$(( $(date +%s) - START ))"
  fi
fi

echo "done: ${DATASET}${SUFFIX} seed ${SEED}"
