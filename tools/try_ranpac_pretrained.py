#!/usr/bin/env python3
"""Fresh HRM-PET + fixed fusion on the exact pretrained/order of native RanPAC.

No old checkpoint, paper, config or engine is edited. Training is opt-in.
Run via the existing paper virtualenv, not RanPAC's private Python environment.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import run_ranpac_baseline as common
from tools import run_ranpac_original as native
from tools import verify_paper_results as verify

PHASES = ("tii", "lora", "baseline", "full")
NPZ_SHA = "ac104c0df8158c46754510e50495536fa6de9647e572e38b5659dc54f260b124"
MODEL = "vit_base_patch16_224"


def checked_summary(path):
    data = json.loads(path.read_text())
    meta = data["metadata"]
    if meta["config"] != native.EXPECTED or meta["revision"] != common.REVISION:
        raise ValueError("Expected completed original RanPAC ImageNet-R ID7")
    order = meta["class_order"]
    if sorted(order) != list(range(200)) or len(data["stages"]) != 10:
        raise ValueError("Invalid class order or incomplete original run")
    if meta["pretrained_sha256"] != NPZ_SHA or data["final"]["stage"] != 10:
        raise ValueError("Unexpected pretrained or incomplete final stage")
    return data


def cli_arguments(phase, folder, data_parent):
    if phase not in PHASES:
        raise ValueError("Unknown phase")
    tii = phase == "tii"
    args = ["imr_hideprompt_5e" if tii else "imr_lora",
            "--model", MODEL, "--original_model", MODEL,
            "--data-path", str(data_parent), "--dataset", "Split-Imagenet-R",
            "--seed", "1", "--num_tasks", "10", "--world_size", "1",
            "--strict_exemplar_free", "--num_workers", "4",
            "--output_dir", str(folder / ("tii" if tii else "lora")),
            "--batch-size", "128" if tii else "24",
            "--epochs", "20" if tii else "50",
            "--ca_lr", "0.005", "--crct_epochs", "30"]
    if tii:
        args += ["--lr", "0.0005", "--ca_storage_efficient_method", "covariance",
                 "--train_inference_task_only"]
    else:
        args += ["--trained_original_model", str(folder / "tii"),
                 "--lr", "0.03", "--con", "0.2", "--lora_rank", "8",
                 "--En", "gen", "--tau", "-10", "--K", "5", "--sched", "cosine",
                 "--lora_momentum", "0.4", "--lora_type", "hide"]
    if phase in ("baseline", "full"):
        args += ["--eval"]
    if phase == "full":
        args += ["--rp_head", "--rp_dim", "10000", "--rp_activation", "relu",
                 "--rp_lambda", "10000", "--rp_feature_source", "lora",
                 "--rp_normalize", "none", "--rp_lora_task", "0",
                 "--rp_logit_blend", "0.0", "--rp_input_norm", "none",
                 "--rp_pin_extractor", "--rp_route_fusion", "--rp_route_fusion_drm",
                 "--rp_route_fusion_weight", "0.7", "--rp_route_fusion_ls_weight", "0.0",
                 "--rp_class_fusion_weight", "0.5", "--rp_class_fusion_sharpen", "1.0",
                 "--rp_class_fusion_min_tasks", "1", "--rp_class_fusion_gate", "margin",
                 "--rp_fusion_ramp", "0.0", "--rp_fusion_ramp_scope", "both"]
    return args


def ordered_split(module, original, order):
    """Use the existing split algorithm; change only its local permutation."""
    def split(train, test, args):
        old_random, old_shuffle = module.random, args.shuffle
        def exact_shuffle(labels):
            if labels != list(range(200)):
                raise ValueError("Unexpected label list for the fixed ImageNet-R order")
            labels[:] = order
        try:
            module.random = SimpleNamespace(shuffle=exact_shuffle)
            args.shuffle = True
            result = original(train, test, args)
        finally:
            module.random, args.shuffle = old_random, old_shuffle
        if result[1] != [order[i:i + 20] for i in range(0, 200, 20)]:
            raise ValueError("Class masks differ from original RanPAC")
        print("RANPAC_CLASS_MASKS=PASS", flush=True)
        return result
    return split


def backbone_digest(state):
    keys = sorted(k for k in state if
                  (k in ("cls_token", "pos_embed") or
                   k.startswith(("patch_embed.", "blocks.", "norm.")))
                  and "lora" not in k and "ssf" not in k)
    if not keys:
        raise ValueError("No frozen backbone tensors")
    digest = hashlib.sha256()
    for key in keys:
        tensor = state[key].detach().cpu().contiguous()
        digest.update((key + str(tensor.dtype) + str(tuple(tensor.shape))).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def save_json_exclusive(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def atomic_checkpoint(original_save, folder):
    def save(state, path, *pos, **kw):
        path = Path(path).resolve()
        if folder.resolve() not in path.parents or path.exists():
            raise ValueError("Checkpoint overwrite/out-of-scope save refused: " + str(path))
        def tensor_bytes(obj):
            if hasattr(obj, "numel") and hasattr(obj, "element_size"):
                return obj.numel() * obj.element_size()
            if isinstance(obj, dict):
                return sum(tensor_bytes(v) for v in obj.values())
            if isinstance(obj, (list, tuple)):
                return sum(tensor_bytes(v) for v in obj)
            return 0
        if shutil.disk_usage(folder).free < max(1024**3, tensor_bytes(state) + 512 * 1024**2):
            raise RuntimeError("Insufficient disk for atomic checkpoint plus safety reserve")
        partial = path.with_suffix(path.suffix + ".partial")
        if partial.exists():
            raise FileExistsError("Partial checkpoint preserved: " + str(partial))
        original_save(state, partial, *pos, **kw)
        partial.replace(path)
    return save


def checkpoint_audit(folder, role, meta, torch):
    records = {}
    for i in range(1, 11):
        path = folder / role / "checkpoint" / ("task%d_checkpoint.pth" % i)
        state = torch.load(path, map_location="cpu", weights_only=False)
        args = vars(state["args"]) if not isinstance(state["args"], dict) else state["args"]
        if (args.get("experiment_pretrained_sha256") != NPZ_SHA
                or args.get("experiment_class_order") != meta["class_order"]
                or args.get("seed") != 1 or args.get("num_tasks") != 10
                or args.get("dataset") != "Split-Imagenet-R"
                or args.get("model") != MODEL):
            raise ValueError("Wrong checkpoint provenance: " + str(path))
        if backbone_digest(state["model"]) != meta["backbone_sha256"]:
            raise ValueError("Frozen backbone changed: " + str(path))
        records[str(path.relative_to(folder))] = common.sha(path)
        del state
    return records


def worker(args):
    import torch
    import numpy as np
    import torchvision
    import timm
    import datasets
    from torchvision.datasets import ImageFolder
    from vits import hide_prompt_vision_transformer as tii_vit
    from vits import hrm_lora_vision_transformer as lora_vit
    folder = args.output_root / args.tag
    if folder.exists() and not (folder / "preflight.json").exists():
        if any(p.name != ".trial.lock" for p in folder.iterdir()):
            raise ValueError("Existing folder is not an initialized trial; preserved")
    summary_path = args.output_root / "_ranpac_support/original/ranpac_original_imr_id7_v1/summary.json"
    ranpac = checked_summary(summary_path)
    source = ranpac["metadata"]
    npz = args.output_root / "_ranpac_support/original/torch-cache/hub/checkpoints" / source["pretrained_url"].split("/")[-1]
    if common.sha(npz) != NPZ_SHA:
        raise ValueError("RanPAC NPZ changed/missing")
    data_parent = verify.resolve_imr_data_path(args.data_root)
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    if args.worker == "preflight":
        data_root = data_parent / "imagenet-r"
        digest = hashlib.sha256()
        counts = {}
        for split in ("train", "test"):
            ds = ImageFolder(str(data_root / split))
            count = [0] * 200
            for path, label in ds.samples:
                path = Path(path)
                count[label] += 1
                digest.update((path.relative_to(data_root).as_posix() + ":" + common.sha(path)).encode())
            counts[split] = count
        if counts != source["counts"] or digest.hexdigest() != source["dataset_sha256"]:
            raise ValueError("Dataset differs from completed RanPAC run")
        hashes = []
        for module, extra in ((tii_vit, {"mlp_structure": [2]}),
                              (lora_vit, {"lora": True, "lora_type": "hide", "rank": 8, "lora_pool_size": 10})):
            model = module.vit_base_patch16_224(pretrained=False, num_classes=200, **extra)
            model.load_pretrained(str(npz))
            hashes.append(backbone_digest(model.state_dict()))
            del model
        if hashes[0] != hashes[1]:
            raise ValueError("TII and LoRA NPZ loaders yield different backbone tensors")
        meta = {"pretrained_sha256": NPZ_SHA, "pretrained_path": str(npz),
                "backbone_sha256": hashes[0], "class_order": source["class_order"],
                "dataset_sha256": source["dataset_sha256"], "counts": counts,
                "data_parent": str(data_parent), "ranpac_summary_sha256": common.sha(summary_path),
                "source_sha256": verify.source_digest(), "driver_sha256": common.sha(__file__),
                "seed": 1, "runtime": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                                     "numpy": np.__version__, "timm": timm.__version__},
                "commands": {p: cli_arguments(p, folder, data_parent) for p in PHASES},
                "limitations": ["Single exploratory run; no test-selected hyperparameters",
                                "Same pretrained/images/task order as RanPAC, not identical training or runtime",
                                "HRM-PET trains adapters across tasks; RanPAC trains SSF only on task 1"]}
        path = folder / "preflight.json"
        if path.exists():
            if json.loads(path.read_text()) != meta:
                raise ValueError("Experiment inputs/source changed; existing run preserved")
        else:
            save_json_exclusive(path, meta)
        print("PRETRAINED_AND_DATA_AND_LOADER_TENSORS=PASS", flush=True)
        return

    meta = json.loads((folder / "preflight.json").read_text())
    if verify.source_digest() != meta["source_sha256"] or common.sha(__file__) != meta["driver_sha256"]:
        raise ValueError("Source changed since preflight")
    phase = args.worker
    if phase != "tii":
        checkpoint_audit(folder, "tii", meta, torch)
    if phase in ("baseline", "full"):
        checkpoint_audit(folder, "lora", meta, torch)
    datasets.split_single_dataset = ordered_split(datasets, datasets.split_single_dataset, meta["class_order"])
    import main as entry
    sys.argv = ["main.py"] + cli_arguments(phase, folder, data_parent)
    training = entry.get_args()
    training.shuffle = True
    training.experiment_class_order = meta["class_order"]
    training.experiment_pretrained_sha256 = NPZ_SHA
    training.experiment_tag = args.tag
    module = __import__("trainers.tii_trainer" if phase == "tii" else "trainers.lora_trainer", fromlist=["train"])
    def create(model_name, **kwargs):
        if model_name != MODEL:
            raise ValueError("Unexpected model name")
        # Resolve the intended module explicitly: importing both custom ViTs
        # registers the same timm name, so global registry order is unsafe.
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        kwargs["pretrained"] = False
        constructor = tii_vit if phase == "tii" else lora_vit
        model = constructor.vit_base_patch16_224(**kwargs)
        model.load_pretrained(str(npz))
        if backbone_digest(model.state_dict()) != meta["backbone_sha256"]:
            raise ValueError("Factory did not load the pinned backbone exactly")
        print("EXACT_RANPAC_PRETRAINED_TENSORS=PASS", flush=True)
        return model
    module.create_model = create
    import utils
    utils.save_on_master = atomic_checkpoint(utils.save_on_master, folder)
    try:
        entry.main(training)
    finally:
        utils.cleanup_distributed()
    checkpoint_records = {}
    for role in (("tii",) if phase == "tii" else ("tii", "lora")):
        checkpoint_records.update(checkpoint_audit(folder, role, meta, torch))
    save_json_exclusive(folder / (phase + "_audit.json"), checkpoint_records)
    print("PRETRAINED_TRIAL_PHASE_COMPLETE=" + phase, flush=True)


def validate_done(folder, phase):
    done = folder / (phase + "_done.json")
    if not done.exists():
        return False
    record = json.loads(done.read_text())
    for relative, sha in record["files"].items():
        path = (folder / relative).resolve()
        if folder.resolve() not in path.parents or common.sha(path) != sha:
            raise ValueError("Completed phase artifact changed: " + str(path))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tag", default="ranpac_pretrained_trial_v1")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-minutes", type=float, default=360)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker", choices=("preflight",) + PHASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,80}", args.tag):
        parser.error("Unsafe tag")
    if not 1 <= args.cpu_threads <= 16 or not math.isfinite(args.max_minutes) or not 1 <= args.max_minutes <= 1440:
        parser.error("Invalid resource limits")
    args.output_root, args.data_root = args.output_root.resolve(), args.data_root.resolve()
    if args.worker:
        worker(args)
        return
    if sys.platform != "linux":
        raise RuntimeError("Use the existing paper Python environment on the Linux 4090 host")
    if not args.output_root.is_dir():
        raise ValueError("Existing output-root required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "0"):
        raise ValueError("CUDA_VISIBLE_DEVICES must be unset or 0")
    folder = args.output_root / args.tag
    if folder.exists() and not (folder / "preflight.json").exists():
        if any(p.name != ".trial.lock" for p in folder.iterdir()):
            raise ValueError("Existing folder is not an initialized trial; preserved")
    if not (folder / "tii_done.json").exists() and args.run and shutil.disk_usage(args.output_root).free < 12 * 1024**3:
        raise RuntimeError("Fresh training requires 12 GiB free disk for new checkpoints; old data untouched")
    folder.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               PYTHONPYCACHEPREFIX=str(folder / "unused-bytecode"),
               OMP_NUM_THREADS=str(args.cpu_threads), MKL_NUM_THREADS=str(args.cpu_threads),
               OPENBLAS_NUM_THREADS=str(args.cpu_threads))
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    base = [str(Path(__file__).resolve()), "--output-root", str(args.output_root),
            "--data-root", str(args.data_root), "--tag", args.tag,
            "--cpu-threads", str(args.cpu_threads)]
    with common.acquire_shared_lock(folder / ".trial.lock"):
        subprocess.run([sys.executable, "-B", "-u"] + base + ["--worker", "preflight"],
                       cwd=ROOT, env=env, check=True, timeout=1800)
        print("PLAN: TII 20 epochs/task; LoRA 50 epochs/task; baseline + fixed full; 10 tasks", flush=True)
        if not args.run:
            return
        start = time.monotonic()
        for phase in PHASES:
            if validate_done(folder, phase):
                print("REUSE_COMPLETED_PHASE=" + phase, flush=True)
                continue
            log = folder / (phase + ".log")
            if log.exists() or (phase in ("tii", "lora") and (folder / phase).exists()):
                raise RuntimeError("Incomplete phase preserved; no automatic restart/overwrite: " + phase)
            remaining = args.max_minutes - (time.monotonic() - start) / 60
            if remaining <= 0:
                raise RuntimeError("Time budget exhausted; completed phases reusable")
            common.idle_gpu_preflight()
            with common.acquire_shared_lock(args.output_root / ".paper_backbone_verifier.lock") as lock:
                common.idle_gpu_preflight()
                cmd = [sys.executable, "-B", "-m", "torch.distributed.run", "--standalone",
                       "--nproc_per_node=1"] + base + ["--worker", phase]
                print("RUN_PHASE=" + phase + " LOG=" + str(log), flush=True)
                native.run_bounded(cmd, ROOT, env, lock, log, remaining)
            text = log.read_text(errors="replace")
            if "PRETRAINED_TRIAL_PHASE_COMPLETE=" + phase not in text:
                raise RuntimeError("Phase missing completion marker: " + phase)
            audit_path = folder / (phase + "_audit.json")
            files = json.loads(audit_path.read_text())
            files[log.name], files[audit_path.name] = common.sha(log), common.sha(audit_path)
            save_json_exclusive(folder / (phase + "_done.json"), {"files": files})
        baseline, full = verify.read_stages(folder / "baseline.log"), verify.read_stages(folder / "full.log")
        meta = json.loads((folder / "preflight.json").read_text())
        result = {"metadata": meta, "baseline": baseline, "full": full,
                  "full_minus_baseline": {k: full[10][k] - baseline[10][k] for k in verify.CORE + verify.RETENTION},
                  "note": "Single exploratory matched-pretrained/order run; no claim of multi-seed significance"}
        summary = folder / "summary.json"
        if not summary.exists():
            save_json_exclusive(summary, result)
        elif json.loads(summary.read_text()) != json.loads(json.dumps(result)):
            raise ValueError("Existing summary differs; preserved")
        common.emit("BASELINE_FINAL", baseline[10])
        common.emit("FULL_FINAL", full[10])
        common.emit("FULL_MINUS_BASELINE", result["full_minus_baseline"])
        print("RANPAC_PRETRAINED_TRIAL_COMPLETE=10/10", flush=True)
        print("SUMMARY=" + str(summary), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("PRETRAINED_TRIAL_STOP: " + str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1)
