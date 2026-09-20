#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
RUN_DIR="${1:-}"
LORA_RANK="${LORA_RANK:-}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: $0 RUN_DIR" >&2
  exit 64
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 1
fi

RUN_BASENAME="$(basename "${RUN_DIR}")"
SEED="${SEED:-}"
if [[ -z "${SEED}" && "${RUN_BASENAME}" =~ seed([0-9]+)$ ]]; then
  SEED="${BASH_REMATCH[1]}"
fi
SEED="${SEED:-42}"
TII_DIR="${TII_DIR:-${OUTPUT_ROOT}/imr_tii_original_10tasks_seed${SEED}}"
TRAIN_LOG="${TRAIN_LOG:-${OUTPUT_ROOT}/${RUN_BASENAME}.log}"
LOG_TAG="${LOG_TAG:-maskfix_v1}"
[[ "${LOG_TAG}" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid LOG_TAG" >&2; exit 64; }
LOG_PATH="${OUTPUT_ROOT}/${RUN_BASENAME}_eval_conventional__${LOG_TAG}.log"

IMR_DATA_PATH="$(PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" -c '
import sys
from tools.verify_paper_results import resolve_imr_data_path
print(resolve_imr_data_path(sys.argv[1], sys.argv[2] or None))
' "${DATASETS_ROOT}" "${DATA_PATH:-}")"
echo "EVALUATION_DATA_PATH=${IMR_DATA_PATH}"

for task_id in $(seq 1 10); do
  [[ -s "${RUN_DIR}/checkpoint/task${task_id}_checkpoint.pth" ]] || {
    echo "Missing LoRA checkpoint: ${RUN_DIR}/checkpoint/task${task_id}_checkpoint.pth" >&2
    exit 2
  }
  [[ -s "${TII_DIR}/checkpoint/task${task_id}_checkpoint.pth" ]] || {
    echo "Missing TII checkpoint: ${TII_DIR}/checkpoint/task${task_id}_checkpoint.pth" >&2
    exit 2
  }
done
if [[ ! -s "${TRAIN_LOG}" ]]; then
  echo "Training log required for protocol audit: ${TRAIN_LOG}" >&2
  exit 2
fi
if [[ -s "${LOG_PATH}" ]]; then
  echo "Refusing to overwrite completed evaluation log: ${LOG_PATH}" >&2
  exit 3
fi

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" -c '
import sys
from protocols import validate_exemplar_free_training_log
validate_exemplar_free_training_log(sys.argv[1])
' "${TRAIN_LOG}"

CHECKPOINT_RANK="$("${PYTHON_BIN}" -c '
import sys
import torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if checkpoint.get("real_feature_memory"):
    raise SystemExit(
        "Checkpoint protocol audit failed: real_feature_memory is present")
state = checkpoint.get("model", checkpoint)
matches = [value for key, value in state.items()
           if key.endswith("lora_layer.k_lora_A")]
if len(matches) != 1:
    raise SystemExit(
        f"Expected one lora_layer.k_lora_A tensor, found {len(matches)}")
print(int(matches[0].shape[-1]))
' "${RUN_DIR}/checkpoint/task1_checkpoint.pth")"
if [[ -z "${LORA_RANK}" ]]; then
  LORA_RANK="${CHECKPOINT_RANK}"
elif [[ "${LORA_RANK}" != "${CHECKPOINT_RANK}" ]]; then
  echo "Requested LoRA rank ${LORA_RANK}, but checkpoint rank is ${CHECKPOINT_RANK}" >&2
  exit 2
fi

echo "Conventional HRM-PET evaluation"
echo "Run=${RUN_DIR}; seed=${SEED}; rank=${LORA_RANK}"
echo "Protocol source log: ${TRAIN_LOG}"

START_TIME="$(date +%s)"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT:-29590}" \
  main.py \
  imr_lora \
  --model vit_base_patch16_224 \
  --original_model vit_base_patch16_224 \
  --batch-size "${EVAL_BATCH_SIZE:-24}" \
  --epochs 1 \
  --data-path "${IMR_DATA_PATH}" \
  --seed "${SEED}" \
  --lr 0.03 \
  --con 0.2 \
  --lora_rank "${LORA_RANK}" \
  --En gen \
  --tau -10 \
  --K 5 \
  --sched cosine \
  --dataset Split-Imagenet-R \
  --lora_momentum 0.4 \
  --lora_type hide \
  --trained_original_model "${TII_DIR}" \
  --num_tasks 10 \
  --strict_exemplar_free \
  --eval \
  --output_dir "${RUN_DIR}" \
  2>&1 | tee "${LOG_PATH}"

END_TIME="$(date +%s)"
printf 'Conventional evaluation wall time seconds: %s\n' "$((END_TIME - START_TIME))" | tee -a "${LOG_PATH}"
FINAL_LINE="$(grep "Average accuracy till task10" "${LOG_PATH}" | tail -n 1 || true)"
REFERENCE_LINE="$(grep "Average accuracy till task10" "${TRAIN_LOG}" | tail -n 1 || true)"
echo "Final conventional metrics:"
echo "${FINAL_LINE}"

"${PYTHON_BIN}" - "${FINAL_LINE}" "${REFERENCE_LINE}" <<'PY'
import re
import sys

candidate, reference = sys.argv[1:]
metrics = ('Acc@task', 'Acc@1', 'Acc@5', 'Loss')
print('Historical comparison: final-stage metrics only. Retention needs corrected intermediate stages.')
limits = {'Loss': 0.001}


def metric(row, name):
    match = re.search(re.escape(name) + r':\s*([-+0-9.]+)', row)
    if match is None:
        raise SystemExit(f'Missing {name}: {row}')
    return float(match.group(1))


passed = True
for name in metrics:
    delta = metric(candidate, name) - metric(reference, name)
    tolerance = limits.get(name, 0.01)
    ok = abs(delta) <= tolerance
    passed = passed and ok
    print(f'{name} delta={delta:+.6f}: {"PASS" if ok else "FAIL"}')
print('CONVENTIONAL_REPRODUCTION_GATE=' + ('PASS' if passed else 'FAIL'))
PY
