#!/usr/bin/env bash
set -euo pipefail

# Conventional HRM-PET (TII + DRM + CRM) evaluation on Split-CUB200 with cost.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
DATA_PATH="${DATA_PATH:-${DATASETS_ROOT}}"
RUN_DIR="${1:-}"
LORA_RANK="${LORA_RANK:-}"

if [[ -z "${RUN_DIR}" ]]; then echo "Usage: $0 RUN_DIR" >&2; exit 64; fi
if [[ ! -x "${PYTHON_BIN}" ]]; then echo "Python not found" >&2; exit 1; fi
RUN_BASENAME="$(basename "${RUN_DIR}")"
SEED="${SEED:-}"; if [[ -z "${SEED}" && "${RUN_BASENAME}" =~ seed([0-9]+)$ ]]; then SEED="${BASH_REMATCH[1]}"; fi; SEED="${SEED:-42}"
TII_DIR="${TII_DIR:-${OUTPUT_ROOT}/cub200_tii_original_10tasks_seed${SEED}}"
LOG_TAG="${LOG_TAG:-maskfix_v1}"
[[ "${LOG_TAG}" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid LOG_TAG" >&2; exit 64; }
LOG_PATH="${OUTPUT_ROOT}/${RUN_BASENAME}_eval_conventional__${LOG_TAG}.log"

for task_id in $(seq 1 10); do
  [[ -s "${RUN_DIR}/checkpoint/task${task_id}_checkpoint.pth" ]] || { echo "Missing LoRA checkpoint task${task_id}" >&2; exit 2; }
  [[ -s "${TII_DIR}/checkpoint/task${task_id}_checkpoint.pth" ]] || { echo "Missing TII checkpoint task${task_id}: ${TII_DIR}" >&2; exit 2; }
done
[[ -s "${LOG_PATH}" ]] && { echo "Refusing to overwrite: ${LOG_PATH}" >&2; exit 3; } || true

cd "${REPO_ROOT}"
CHECKPOINT_RANK="$("${PYTHON_BIN}" -c '
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
s = c.get("model", c)
m = [v for k, v in s.items() if k.endswith("lora_layer.k_lora_A")]
if len(m) != 1: raise SystemExit(f"Expected one k_lora_A, found {len(m)}")
print(int(m[0].shape[-1]))
' "${RUN_DIR}/checkpoint/task1_checkpoint.pth")"
[[ -z "${LORA_RANK}" ]] && LORA_RANK="${CHECKPOINT_RANK}"

echo "Conventional HRM-PET (DRM+CRM) on Split-CUB200; seed=${SEED}; rank=${LORA_RANK}"
START_TIME="$(date +%s)"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=1 --master_port="${MASTER_PORT:-29552}" \
  main.py imr_lora \
  --model vit_base_patch16_224 --original_model vit_base_patch16_224 \
  --batch-size "${EVAL_BATCH_SIZE:-24}" --epochs 1 --data-path "${DATA_PATH}" \
  --seed "${SEED}" --lr 0.03 --con 0.2 --lora_rank "${LORA_RANK}" \
  --En gen --tau -10 --K 5 --sched cosine --dataset Split-CUB200 \
  --lora_momentum 0.4 --lora_type hide --trained_original_model "${TII_DIR}" \
  --num_tasks 10 --report_conventional_cost --strict_exemplar_free --eval \
  --output_dir "${RUN_DIR}" 2>&1 | tee "${LOG_PATH}"
printf 'Conventional evaluation wall time seconds: %s\n' "$(( $(date +%s) - START_TIME ))" | tee -a "${LOG_PATH}"
echo "Final conventional metrics:"; grep "Average accuracy till task10" "${LOG_PATH}" | tail -n 1 || true
