#!/usr/bin/env bash
set -euo pipefail

# Hybrid evaluation on any dataset: HRM-PET's own LoRA supplies first-session
# adaptation, and a routing-free RanPAC-style second-order head classifies.
#
# On Split-CUB200 this beat conventional HRM-PET where every routing-based
# variant had failed: Acc@1 87.50 vs 86.53, Acc@task 94.02 vs 93.21, better
# Forgetting/Backward, at 1.0 LoRA/sample instead of 1.60.
#
# Usage:  DATASET=Split-CUB200 TII_DIR=... $0 RUN_DIR
# Knobs:  RP_DIM RP_LAMBDA RP_ACT RP_NORM RP_SOURCE RP_LORA_TASK CALIBRATE
#         RP_INORM RP_BARE RP_LSEARCH RP_LCRIT RP_LSCOPE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATASETS_ROOT="${DATASETS_ROOT:-${WORK_ROOT}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
DATA_PATH="${DATA_PATH:-${DATASETS_ROOT}}"
RUN_DIR="${1:-}"
DATASET="${DATASET:-}"
RP_DIM="${RP_DIM:-10000}"
RP_ACT="${RP_ACT:-relu}"
RP_LAMBDA="${RP_LAMBDA:-10000}"
RP_SOURCE="${RP_SOURCE:-lora}"
RP_NORM="${RP_NORM:-none}"
RP_LORA_TASK="${RP_LORA_TASK:-0}"
CALIBRATE="${CALIBRATE:-1}"
NUM_TASKS="${NUM_TASKS:-10}"
CONFIG="${CONFIG:-imr_lora}"
RP_BLEND="${RP_BLEND:-0.0}"
RP_PIN="${RP_PIN:-1}"
RP_BARE="${RP_BARE:-0}"
RP_INORM="${RP_INORM:-none}"
RP_ROUTE_AUDIT="${RP_ROUTE_AUDIT:-0}"
RP_LAYER_STAT="${RP_LAYER_STAT:-0}"
RP_LS_W="${RP_LS_W:-0.0}"
RP_COST="${RP_COST:-0}"
RP_CLS_AUDIT="${RP_CLS_AUDIT:-0}"
RP_CLS_W="${RP_CLS_W:-0.0}"
RP_CLS_SHARP="${RP_CLS_SHARP:-1.0}"
RP_CLS_MIN="${RP_CLS_MIN:-1}"
RP_CLS_GATE="${RP_CLS_GATE:-none}"
RP_RAMP="${RP_RAMP:-0.0}"
RP_RAMP_SCOPE="${RP_RAMP_SCOPE:-both}"
RP_FUSE="${RP_FUSE:-0}"
RP_FUSE_W="${RP_FUSE_W:-0.5}"
RP_FUSE_DRM="${RP_FUSE_DRM:-0}"
RP_LSEARCH="${RP_LSEARCH:-0}"
# RP_LCRIT: mse | cosine | accuracy
RP_LCRIT="${RP_LCRIT:-mse}"
RP_LSCOPE="${RP_LSCOPE:-task}"
RP_BARE_MODEL="${RP_BARE_MODEL:-}"
# Must match the backbone the checkpoints were trained with.
BACKBONE="${BACKBONE:-vit_base_patch16_224}"

if [[ -z "${RUN_DIR}" ]]; then echo "Usage: DATASET=... TII_DIR=... $0 RUN_DIR" >&2; exit 64; fi
if [[ -z "${DATASET}" ]]; then echo "Set DATASET (e.g. Split-CUB200)" >&2; exit 64; fi
if [[ -z "${TII_DIR:-}" ]]; then echo "Set TII_DIR" >&2; exit 64; fi
if [[ ! -x "${PYTHON_BIN}" ]]; then echo "Python not found" >&2; exit 1; fi
if [[ "${RP_LAYER_STAT}" != "0" || "${RP_LS_W}" != "0" && "${RP_LS_W}" != "0.0" ]]; then
  echo "Layer-stat routing was removed; use RP_LAYER_STAT=0 and RP_LS_W=0.0" >&2
  exit 64
fi
RUN_BASENAME="$(basename "${RUN_DIR}")"
SEED="${SEED:-}"; if [[ -z "${SEED}" && "${RUN_BASENAME}" =~ seed([0-9]+)$ ]]; then SEED="${BASH_REMATCH[1]}"; fi; SEED="${SEED:-42}"
LOG_TAG="${LOG_TAG:-maskfix_v1}"
[[ "${LOG_TAG}" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid LOG_TAG" >&2; exit 64; }
CONVENTIONAL_LOG="${OUTPUT_ROOT}/${RUN_BASENAME}_eval_conventional__${LOG_TAG}.log"

for task_id in $(seq 1 "${NUM_TASKS}"); do
  [[ -s "${RUN_DIR}/checkpoint/task${task_id}_checkpoint.pth" ]] || { echo "Missing LoRA checkpoint task${task_id}" >&2; exit 2; }
  [[ -s "${TII_DIR}/checkpoint/task${task_id}_checkpoint.pth" ]] || { echo "Missing TII checkpoint task${task_id}" >&2; exit 2; }
done

CHECKPOINT_RANK="$("${PYTHON_BIN}" -c '
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if c.get("real_feature_memory"): raise SystemExit("STRICT_CHECKPOINT_AUDIT=FAIL (real_feature_memory present)")
s = c.get("model", c); m = [v for k, v in s.items() if k.endswith("lora_layer.k_lora_A")]
if len(m) != 1: raise SystemExit(f"Expected one k_lora_A, found {len(m)}")
print(int(m[0].shape[-1]))
' "${RUN_DIR}/checkpoint/task1_checkpoint.pth")"

tag() { printf '%s' "$1" | tr '.' 'p'; }
GATE_TAG=""; [[ "${RP_CLS_GATE}" != "none" ]] && GATE_TAG="g${RP_CLS_GATE}"
RAMP_TAG=""; [[ "${RP_RAMP}" != "0.0" ]] && RAMP_TAG="r$(tag "${RP_RAMP}")"
[[ "${RP_RAMP}" != "0.0" && "${RP_RAMP_SCOPE}" != "both" ]] && RAMP_TAG="${RAMP_TAG}s${RP_RAMP_SCOPE}"
CAL_FLAG=""; [[ "${CALIBRATE}" == "1" ]] && CAL_FLAG="--rp_calibrate"
LSEARCH_FLAG=""; LSEARCH_TAG=""
[[ "${RP_LSEARCH}" == "1" ]] && LSEARCH_FLAG="--rp_lambda_search --rp_lambda_criterion ${RP_LCRIT} --rp_lambda_scope ${RP_LSCOPE}" && LSEARCH_TAG="lsearch${RP_LCRIT}${RP_LSCOPE}"
PIN_FLAG=""; [[ "${RP_PIN}" == "1" ]] && PIN_FLAG="--rp_pin_extractor"
[[ "${RP_BARE}" == "1" ]] && PIN_FLAG="--rp_pin_extractor --rp_bare_extractor"
BARE_TAG=""; [[ "${RP_BARE}" == "1" ]] && BARE_TAG="bare"
BACKBONE_TAG=""
[[ "${BACKBONE}" != "vit_base_patch16_224" ]] && BACKBONE_TAG="b$(printf %s "${BACKBONE#vit_base_patch16_224_}" | tr -d _)"
BAREM_FLAG=""
# Tag by the last chunk of the model name so two bare runs on different
# checkpoints cannot land on the same log path.
[[ -n "${RP_BARE_MODEL}" ]] && BAREM_FLAG="--rp_bare_model ${RP_BARE_MODEL}" && BARE_TAG="${BARE_TAG}m${RP_BARE_MODEL##*_}"
AUDIT_FLAG=""; [[ "${RP_ROUTE_AUDIT}" == "1" ]] && AUDIT_FLAG="--rp_route_audit"
[[ "${RP_COST}" == "1" ]] && AUDIT_FLAG="${AUDIT_FLAG} --report_conventional_cost"
[[ "${RP_CLS_AUDIT}" == "1" ]] && AUDIT_FLAG="${AUDIT_FLAG} --classifier_union_audit"
FUSE_FLAG=""; [[ "${RP_FUSE}" == "1" ]] && FUSE_FLAG="--rp_route_fusion --rp_route_fusion_weight ${RP_FUSE_W}"
[[ "${RP_FUSE_DRM}" == "1" ]] && FUSE_FLAG="--rp_route_fusion_drm --rp_route_fusion_weight ${RP_FUSE_W} --rp_route_fusion_ls_weight ${RP_LS_W} --rp_class_fusion_weight ${RP_CLS_W} --rp_class_fusion_sharpen ${RP_CLS_SHARP} --rp_class_fusion_min_tasks ${RP_CLS_MIN} --rp_class_fusion_gate ${RP_CLS_GATE} --rp_fusion_ramp ${RP_RAMP} --rp_fusion_ramp_scope ${RP_RAMP_SCOPE}"
LOG_PATH="${OUTPUT_ROOT}/${RUN_BASENAME}_eval_rp_${RP_SOURCE}_d${RP_DIM}_${RP_ACT}_l$(tag "${RP_LAMBDA}")_n${RP_NORM}_t${RP_LORA_TASK}_b$(tag "${RP_BLEND}")_p${RP_PIN}_i${RP_INORM}_c${CALIBRATE}_ra${RP_ROUTE_AUDIT}ls${RP_LAYER_STAT}_f${RP_FUSE}d${RP_FUSE_DRM}w$(tag "${RP_FUSE_W}")lsw$(tag "${RP_LS_W}")c${RP_COST}ca${RP_CLS_AUDIT}cw$(tag "${RP_CLS_W}")sh$(tag "${RP_CLS_SHARP}")m${RP_CLS_MIN}${GATE_TAG}${RAMP_TAG}${BARE_TAG}${LSEARCH_TAG}${BACKBONE_TAG}.log"
LOG_PATH="${LOG_PATH%.log}__${LOG_TAG}.log"
[[ -s "${LOG_PATH}" ]] && { echo "Refusing to overwrite: ${LOG_PATH}" >&2; exit 3; } || true

cd "${REPO_ROOT}"
echo "Hybrid RP head on ${DATASET}; dim=${RP_DIM} lambda=${RP_LAMBDA} source=${RP_SOURCE} adapter=${RP_LORA_TASK} rank=${CHECKPOINT_RANK} calibrate=${CALIBRATE}"
START_TIME="$(date +%s)"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node=1 --master_port="${MASTER_PORT:-29558}" \
  main.py "${CONFIG}" \
  --model "${BACKBONE}" --original_model "${BACKBONE}" \
  --batch-size "${EVAL_BATCH_SIZE:-24}" --epochs 1 --data-path "${DATA_PATH}" \
  --seed "${SEED}" --lr 0.03 --con 0.2 --lora_rank "${CHECKPOINT_RANK}" \
  --En gen --tau -10 --K 5 --sched cosine --dataset "${DATASET}" \
  --lora_momentum 0.4 --lora_type hide --trained_original_model "${TII_DIR}" \
  --num_tasks "${NUM_TASKS}" --rp_head --rp_dim "${RP_DIM}" \
  --rp_activation "${RP_ACT}" --rp_lambda "${RP_LAMBDA}" \
  --rp_feature_source "${RP_SOURCE}" --rp_normalize "${RP_NORM}" \
  --rp_lora_task "${RP_LORA_TASK}" --rp_logit_blend "${RP_BLEND}" --rp_input_norm "${RP_INORM}" ${CAL_FLAG} ${PIN_FLAG} ${AUDIT_FLAG} ${FUSE_FLAG} ${LSEARCH_FLAG} ${BAREM_FLAG} \
  --strict_exemplar_free --eval --output_dir "${RUN_DIR}" 2>&1 | tee "${LOG_PATH}"
printf 'Hybrid RP head wall time seconds: %s\n' "$(( $(date +%s) - START_TIME ))" | tee -a "${LOG_PATH}"
echo "Final hybrid metrics:"; grep "Average accuracy till task${NUM_TASKS}" "${LOG_PATH}" | tail -n 1 || true

if [[ -s "${CONVENTIONAL_LOG}" ]]; then
  NUM_TASKS="${NUM_TASKS}" "${PYTHON_BIN}" - "${CONVENTIONAL_LOG}" "${LOG_PATH}" <<'PY' | tee -a "${LOG_PATH}"
import os, re, sys
tasks = os.environ.get('NUM_TASKS', '10')
def final_row(p):
    with open(p, 'r', encoding='utf-8') as h:
        r = [l.strip() for l in h if f'Average accuracy till task{tasks}' in l]
    if not r: raise SystemExit('HYBRID_GATE=FAIL (missing final row)')
    return r[-1]
def metric(row, name):
    mm = re.search(rf'{re.escape(name)}:\s*(-?[0-9]+(?:\.[0-9]+)?)', row)
    return None if not mm else float(mm.group(1))
base, prop = final_row(sys.argv[1]), final_row(sys.argv[2])
checks = {}
print('Comparison with conventional HRM-PET baseline:')
for name in ('Acc@task', 'Acc@1', 'Acc@5', 'Backward'):
    a, b = metric(prop, name), metric(base, name)
    if a is None or b is None: continue
    checks[name] = (a - b) > 0.0
    print(f'  {name} delta={a-b:+.4f}: {"PASS" if checks[name] else "FAIL"}')
for name in ('Loss', 'Forgetting'):
    a, b = metric(prop, name), metric(base, name)
    if a is None or b is None: continue
    checks[name] = (a - b) < 0.0
    print(f'  {name} delta={a-b:+.4f}: {"PASS" if checks[name] else "FAIL"}')
a, b = metric(prop, 'LoRA/sample'), metric(base, 'LoRA/sample')
if a is not None and b is not None:
    print(f'  LoRA/sample {a:.4f} vs {b:.4f} (lower is cheaper)')
print('HYBRID_GATE=' + ('PASS' if all(checks.values()) else 'FAIL'))
PY
fi
