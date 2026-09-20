#!/usr/bin/env python3
"""Train/evaluate HRM-PET and Full on CIFAR-100/ImageNet-A with RanPAC's
AugReg ImageNet-21K -> ImageNet-1K ViT-B/16, class order 1993, train seed 1.

This is an isolated exploratory runner.  It never reuses or overwrites the
paper's Sup-21K checkpoints.  Without --run it performs CPU-only provenance
checks and prints the exact plan.
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
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_ranpac_baseline as common
from tools import run_ranpac_original as native
from tools import run_ranpac_original_extra as extra
from tools import try_ranpac_pretrained as imr_trial
from tools import verify_paper_results as verify

PHASES = ("tii", "lora", "baseline", "full")
TAG = "ranpac_matched_extra_v1"
MODEL = "vit_base_patch16_224"
SPECS = {
    "cifar100": {
        "classes": 100, "tasks": 10, "dataset": "Split-CIFAR100",
        "tii": "cifar100_hideprompt_5e", "lora": "cifar100_lora",
        "tii_batch": "128", "lora_batch": "24",
    },
    "ima": {
        "classes": 200, "tasks": 10, "dataset": "Split-Imagenet-A",
        "tii": "ima_hideprompt_5e", "lora": "ima_lora",
        "tii_batch": "128", "lora_batch": "24",
    },
}


def native_backbone(name, metadata):
    """Return the exact checkpoint used by the verified native RanPAC run."""
    if name not in SPECS:
        raise ValueError("Unknown dataset: " + name)
    filename = extra.SPECS[name]["checkpoint_file"]
    digest = metadata.get("pretrained_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Missing native RanPAC checkpoint digest for " + name)
    return filename, digest


def source_metadata(output_root, name):
    """Read dataset/order provenance from the already verified RanPAC run."""
    support = output_root / "_ranpac_support/original"
    candidates = [
        support / ("ranpac_original_extra_v1-" + name) / "summary.json",
        support / ("preflight-extra-" + name) / "metadata.json",
    ]
    paths = [p for p in candidates if p.is_file()]
    if not paths:
        raise ValueError("Missing verified RanPAC metadata for " + name)
    raw = json.loads(paths[0].read_text())
    meta = raw.get("metadata", raw)
    spec = SPECS[name]
    order = meta.get("class_order")
    if (meta.get("dataset") != name or meta.get("revision") != common.REVISION
            or meta.get("torch_seed_from_original_trainer") != 1
            or not isinstance(order, list) or sorted(order) != list(range(spec["classes"]))
            or len(meta.get("counts", {}).get("train", [])) != spec["classes"]
            or len(meta.get("counts", {}).get("test", [])) != spec["classes"]):
        raise ValueError("Unexpected/incomplete RanPAC metadata: " + str(paths[0]))
    return meta, paths[0]


def data_record(data_root, name):
    import numpy as np
    from torchvision.datasets import CIFAR100, ImageFolder
    target = extra.resolve_data(data_root, name)
    digest = hashlib.sha256()
    counts = {}
    if name == "cifar100":
        parent = target.parent
        for split, train in (("train", True), ("test", False)):
            ds = CIFAR100(str(parent), train=train, download=False)
            labels = np.asarray(ds.targets, dtype=np.int64)
            counts[split] = np.bincount(labels, minlength=100).tolist()
            digest.update(split.encode())
            digest.update(ds.data.tobytes())
            digest.update(labels.tobytes())
    else:
        parent = target.parent
        for split in ("train", "test"):
            ds = ImageFolder(str(target / split))
            count = [0] * SPECS[name]["classes"]
            for path, label in ds.samples:
                path = Path(path)
                count[label] += 1
                digest.update((split + "/" + path.relative_to(target).as_posix()
                               + ":" + common.sha(path)).encode())
            counts[split] = count
    return parent.resolve(), counts, digest.hexdigest()


def cli_arguments(name, phase, folder, data_parent):
    if name not in SPECS or phase not in PHASES:
        raise ValueError("Unknown dataset/phase")
    spec = SPECS[name]
    tii = phase == "tii"
    args = [spec["tii"] if tii else spec["lora"],
            "--model", MODEL, "--original_model", MODEL,
            "--data-path", str(data_parent), "--dataset", spec["dataset"],
            "--seed", "1", "--num_tasks", str(spec["tasks"]),
            "--world_size", "1", "--num_workers", "4",
            "--output_dir", str(folder / ("tii" if tii else "lora")),
            "--batch-size", spec["tii_batch"] if tii else spec["lora_batch"],
            "--epochs", "20" if tii else "50", "--ca_lr", "0.005",
            "--crct_epochs", "30"]
    if tii:
        args += ["--lr", "0.0005", "--ca_storage_efficient_method", "covariance",
                 "--train_inference_task_only"]
    else:
        args += ["--trained_original_model", str(folder / "tii"),
                 "--lr", "0.03", "--con", "0.2", "--lora_rank", "8",
                 "--En", "gen", "--tau", "-10", "--K", "5", "--sched", "cosine",
                 "--lora_momentum", "0.4", "--lora_type", "hide",
                 "--strict_exemplar_free"]
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


def ordered_split(module, original, order, classes):
    def split(train, test, args):
        old_random, old_shuffle = module.random, args.shuffle
        def exact_shuffle(labels):
            if labels != list(range(classes)):
                raise ValueError("Unexpected labels for fixed RanPAC order")
            labels[:] = order
        try:
            module.random = SimpleNamespace(shuffle=exact_shuffle)
            args.shuffle = True
            result = original(train, test, args)
        finally:
            module.random, args.shuffle = old_random, old_shuffle
        width = math.ceil(classes / args.num_tasks)
        if result[1] != [order[i:i + width] for i in range(0, classes, width)]:
            raise ValueError("Class masks differ from RanPAC order")
        print("RANPAC_CLASS_MASKS=PASS", flush=True)
        return result
    return split


def checkpoint_audit(folder, name, role, meta, torch):
    records = {}
    spec = SPECS[name]
    for stage in range(1, spec["tasks"] + 1):
        path = folder / role / "checkpoint" / ("task%d_checkpoint.pth" % stage)
        state = torch.load(path, map_location="cpu", weights_only=False)
        saved = state["args"] if isinstance(state["args"], dict) else vars(state["args"])
        if (saved.get("experiment_pretrained_sha256") != meta["pretrained_sha256"]
                or saved.get("experiment_class_order") != meta["class_order"]
                or saved.get("seed") != 1 or saved.get("num_tasks") != spec["tasks"]
                or saved.get("dataset") != spec["dataset"] or saved.get("model") != MODEL):
            raise ValueError("Wrong checkpoint provenance: " + str(path))
        if imr_trial.backbone_digest(state["model"]) != meta["backbone_sha256"]:
            raise ValueError("Frozen backbone changed: " + str(path))
        records[str(path.relative_to(folder))] = common.sha(path)
        del state
    return records


def worker(args):
    import numpy as np
    import timm
    import torch
    import torchvision
    import datasets
    from vits import hide_prompt_vision_transformer as tii_vit
    from vits import hrm_lora_vision_transformer as lora_vit

    name = args.datasets[0]
    spec = SPECS[name]
    folder = args.output_root / args.tag / name
    meta_source, source_path = source_metadata(args.output_root, name)
    data_parent, counts, dataset_sha = data_record(args.data_root, name)
    if counts != meta_source["counts"] or dataset_sha != meta_source["dataset_sha256"]:
        raise ValueError("Live dataset differs from verified RanPAC metadata")
    npz_file, npz_sha = native_backbone(name, meta_source)
    npz = args.output_root / "_ranpac_support/original/torch-cache/hub/checkpoints" / npz_file
    if not npz.is_file() or common.sha(npz) != npz_sha:
        raise ValueError("Pinned AugReg-21K->1K checkpoint missing/changed: " + str(npz))
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)

    if args.worker == "preflight":
        hashes = []
        for module, extra_args in ((tii_vit, {"mlp_structure": [2]}),
                                   (lora_vit, {"lora": True, "lora_type": "hide",
                                               "rank": 8, "lora_pool_size": 10})):
            model = module.vit_base_patch16_224(pretrained=False,
                                                num_classes=spec["classes"], **extra_args)
            model.load_pretrained(str(npz))
            hashes.append(imr_trial.backbone_digest(model.state_dict()))
            del model
        if hashes[0] != hashes[1]:
            raise ValueError("TII and LoRA loaders produced different backbone tensors")
        meta = {"dataset": name, "dataset_name": spec["dataset"], "classes": spec["classes"],
                "tasks": spec["tasks"], "training_seed": 1, "class_order_seed": 1993,
                "class_order": meta_source["class_order"], "counts": counts,
                "dataset_sha256": dataset_sha, "data_parent": str(data_parent),
                "pretrained_sha256": npz_sha, "pretrained_path": str(npz),
                "backbone_sha256": hashes[0], "source_metadata": str(source_path),
                "source_metadata_sha256": common.sha(source_path),
                "source_sha256": verify.source_digest(), "driver_sha256": common.sha(__file__),
                "runtime": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                            "numpy": np.__version__, "timm": timm.__version__},
                "commands": {p: cli_arguments(name, p, folder, data_parent) for p in PHASES},
                "limitations": ["Single exploratory order/run; no multi-seed claim",
                                "Same backbone/data/order as verified RanPAC metadata; method budgets differ"]}
        path = folder / "preflight.json"
        if path.exists() and json.loads(path.read_text()) != meta:
            raise ValueError("Inputs/source changed; existing run preserved")
        if not path.exists():
            imr_trial.save_json_exclusive(path, meta)
        print("MATCHED_PREFLIGHT_PASS=" + name, flush=True)
        return

    meta = json.loads((folder / "preflight.json").read_text())
    if verify.source_digest() != meta["source_sha256"] or common.sha(__file__) != meta["driver_sha256"]:
        raise ValueError("Source changed after preflight")
    phase = args.worker
    if phase != "tii":
        checkpoint_audit(folder, name, "tii", meta, torch)
    if phase in ("baseline", "full"):
        checkpoint_audit(folder, name, "lora", meta, torch)
    datasets.split_single_dataset = ordered_split(
        datasets, datasets.split_single_dataset, meta["class_order"], spec["classes"])
    import main as entry
    sys.argv = ["main.py"] + cli_arguments(name, phase, folder, data_parent)
    training = entry.get_args()
    training.shuffle = True
    training.experiment_class_order = meta["class_order"]
    training.experiment_pretrained_sha256 = meta["pretrained_sha256"]
    training.experiment_tag = args.tag + "/" + name
    module = __import__("trainers.tii_trainer" if phase == "tii" else "trainers.lora_trainer",
                        fromlist=["train"])
    def create(model_name, **kwargs):
        if model_name != MODEL:
            raise ValueError("Unexpected model")
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        kwargs["pretrained"] = False
        constructor = tii_vit if phase == "tii" else lora_vit
        model = constructor.vit_base_patch16_224(**kwargs)
        model.load_pretrained(str(npz))
        if imr_trial.backbone_digest(model.state_dict()) != meta["backbone_sha256"]:
            raise ValueError("Factory did not load pinned backbone")
        print("EXACT_RANPAC_PRETRAINED_TENSORS=PASS", flush=True)
        return model
    module.create_model = create
    import utils
    utils.save_on_master = imr_trial.atomic_checkpoint(utils.save_on_master, folder)
    try:
        entry.main(training)
    finally:
        utils.cleanup_distributed()
    records = {}
    for role in (("tii",) if phase == "tii" else ("tii", "lora")):
        records.update(checkpoint_audit(folder, name, role, meta, torch))
    imr_trial.save_json_exclusive(folder / (phase + "_audit.json"), records)
    print("MATCHED_PHASE_COMPLETE=%s:%s" % (name, phase), flush=True)


def run_streaming(cmd, env, lock, log, minutes):
    """Write a durable phase log and mirror it to the nohup launcher log."""
    process = None
    with log.open("x", encoding="utf-8") as stream:
        try:
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, errors="replace",
                                       bufsize=1, start_new_session=True, pass_fds=(lock.fileno(),))
            def pump():
                for line in process.stdout:
                    stream.write(line)
                    stream.flush()
                    print(line, end="", flush=True)
            thread = threading.Thread(target=pump, daemon=True)
            thread.start()
            try:
                code = process.wait(timeout=minutes * 60)
            except subprocess.TimeoutExpired:
                code = 124
            finally:
                native.stop_owned_process(process)
                thread.join(timeout=30)
            stream.write("\nMATCHED_RUN_EXIT_CODE=%d\n" % code)
            stream.flush()
            if code:
                raise RuntimeError("Phase failed/timeout; inspect " + str(log))
        finally:
            native.stop_owned_process(process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=tuple(SPECS),
                        default=["cifar100", "ima"])
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-minutes-per-dataset", type=float, default=1440)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker", choices=("preflight",) + PHASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("Duplicate datasets")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,80}", args.tag):
        parser.error("Unsafe tag")
    if (not 1 <= args.cpu_threads <= 16 or not math.isfinite(args.max_minutes_per_dataset)
            or not 1 <= args.max_minutes_per_dataset <= 2880):
        parser.error("Invalid resource limits")
    args.output_root, args.data_root = args.output_root.resolve(), args.data_root.resolve()
    if args.worker:
        if len(args.datasets) != 1:
            raise ValueError("Worker requires one dataset")
        worker(args)
        return
    if sys.platform != "linux":
        raise RuntimeError("Run on the Linux 4090 host")
    if not args.output_root.is_dir() or not args.data_root.is_dir():
        raise ValueError("Existing output/data roots required")
    root = args.output_root / args.tag
    root.mkdir(exist_ok=True)
    incomplete = sum(not (root / d / "summary.json").is_file() for d in args.datasets)
    if args.run and shutil.disk_usage(args.output_root).free < incomplete * 12 * 1024**3:
        raise RuntimeError("Need at least 12 GiB free per incomplete dataset")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               PYTHONPYCACHEPREFIX=str(root / "unused-bytecode"),
               OMP_NUM_THREADS=str(args.cpu_threads), MKL_NUM_THREADS=str(args.cpu_threads),
               OPENBLAS_NUM_THREADS=str(args.cpu_threads))
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    for name in args.datasets:
        folder = root / name
        folder.mkdir(exist_ok=True)
        base = [str(Path(__file__).resolve()), "--output-root", str(args.output_root),
                "--data-root", str(args.data_root), "--datasets", name, "--tag", args.tag,
                "--cpu-threads", str(args.cpu_threads),
                "--max-minutes-per-dataset", str(args.max_minutes_per_dataset)]
        subprocess.run([sys.executable, "-B", "-u"] + base + ["--worker", "preflight"],
                       cwd=ROOT, env=env, check=True, timeout=1800)
        print("PLAN_%s=TII20x10 + LoRA50x10 + baseline/full eval" % name.upper(), flush=True)
        if not args.run:
            continue
        start = time.monotonic()
        for phase in PHASES:
            if imr_trial.validate_done(folder, phase):
                print("REUSE_COMPLETED_PHASE=%s:%s" % (name, phase), flush=True)
                continue
            log = folder / (phase + ".log")
            if log.exists() or (phase in ("tii", "lora") and (folder / phase).exists()):
                raise RuntimeError("Incomplete phase preserved; use a new tag: " + str(log))
            remaining = args.max_minutes_per_dataset - (time.monotonic() - start) / 60
            if remaining <= 0:
                raise RuntimeError("Dataset time budget exhausted; completed phases reusable")
            common.idle_gpu_preflight()
            with common.acquire_shared_lock(args.output_root / ".paper_backbone_verifier.lock") as lock:
                common.idle_gpu_preflight()
                cmd = [sys.executable, "-B", "-m", "torch.distributed.run", "--standalone",
                       "--nproc_per_node=1"] + base + ["--worker", phase]
                print("RUN_PHASE=%s:%s LOG=%s" % (name, phase, log), flush=True)
                run_streaming(cmd, env, lock, log, remaining)
            marker = "MATCHED_PHASE_COMPLETE=%s:%s" % (name, phase)
            if marker not in log.read_text(errors="replace"):
                raise RuntimeError("Completion marker missing: " + marker)
            audit = folder / (phase + "_audit.json")
            files = json.loads(audit.read_text())
            files[log.name], files[audit.name] = common.sha(log), common.sha(audit)
            imr_trial.save_json_exclusive(folder / (phase + "_done.json"), {"files": files})
        baseline = verify.read_stages(folder / "baseline.log", SPECS[name]["tasks"])
        full = verify.read_stages(folder / "full.log", SPECS[name]["tasks"])
        result = {"metadata": json.loads((folder / "preflight.json").read_text()),
                  "baseline": baseline, "full": full,
                  "final": {"HRM-PET": baseline[SPECS[name]["tasks"]],
                            "Full": full[SPECS[name]["tasks"]]},
                  "full_minus_baseline": {k: full[10][k] - baseline[10][k]
                                           for k in verify.CORE + verify.RETENTION}}
        summary = folder / "summary.json"
        if not summary.exists():
            imr_trial.save_json_exclusive(summary, result)
        elif json.loads(summary.read_text()) != json.loads(json.dumps(result)):
            raise ValueError("Existing summary differs; preserved")
        common.emit("MATCHED_EXTRA_FINAL", {"dataset": name,
                    "HRM-PET": result["final"]["HRM-PET"]["Acc@1"],
                    "Full": result["final"]["Full"]["Acc@1"]})
        print("MATCHED_DATASET_COMPLETE=" + name, flush=True)
    if args.run and all((root / d / "summary.json").is_file() for d in args.datasets):
        print("MATCHED_EXTRA_ALL_COMPLETE=%d/%d" % (len(args.datasets), len(args.datasets)), flush=True)
        print("RESULT_ROOT=" + str(root), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("MATCHED_EXTRA_STOP: " + str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1)
