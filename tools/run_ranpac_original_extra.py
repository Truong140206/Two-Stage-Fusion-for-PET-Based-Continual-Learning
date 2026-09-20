#!/usr/bin/env python3
"""Run official RanPAC ID7 on CIFAR-100 and ImageNet-A, without 5-Datasets.

The pinned upstream repository and existing private Python 3.9 environment are
reused.  Each dataset keeps its own official ID7 backbone/PETL configuration.
No paper file or earlier ImageNet-R run is modified.  ``--run`` is required to
start GPU work; otherwise the command performs data/model provenance checks.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import runpy
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import run_ranpac_baseline as common
from tools import run_ranpac_original as native


def _row(dataset, model, convnet, classes, batch):
    return {"ID": "7", "dataset": dataset, "shuffle": "True",
            "init_cls": str(classes), "increment": str(classes),
            "model_name": model, "convnet_type": convnet, "device": "0",
            "seed": "1993", "batch_size": str(batch), "tuned_epoch": "20",
            "body_lr": "0.01", "head_lr": "0.01", "weight_decay": "0.0005",
            "min_lr": "0.0", "use_RP": "True", "M": "10000",
            "use_input_norm": "False"}


SPECS = {
    "cifar100": {
        "official": "cifar224", "csv": "cifar224_publish.csv",
        "classes": 100, "increment": 10, "tasks": 10, "kind": "cifar",
        "checkpoint_file":
            "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz",
        "expected": _row("cifar224", "adapter",
                         "pretrained_vit_b16_224_in21k_adapter", 10, 48),
    },
    "ima": {
        "official": "imageneta", "csv": "imageneta_publish.csv",
        "classes": 200, "increment": 20, "tasks": 10, "kind": "imagefolder",
        "checkpoint_file":
            "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--"
            "imagenet2012-steps_20k-lr_0.01-res_224.npz",
        "expected": _row("imageneta", "ssf",
                         "pretrained_vit_b16_224_ssf", 20, 48),
    },
}
EASYDICT_VERSION = "1.13"


def ensure_extra_environment(support, env, python, install):
    """Add the Adapter-only dependency without rebuilding the proven env."""
    marker = support / "environment-extra-ready.json"
    expected = {"easydict": EASYDICT_VERSION}
    if marker.exists() and json.loads(marker.read_text()) != expected:
        raise ValueError("Existing extra environment marker differs; preserved")
    probe = [str(python), "-I", "-c",
             "import importlib.metadata as m; from easydict import EasyDict; "
             "assert m.version('easydict') == '" + EASYDICT_VERSION + "'; "
             "assert EasyDict(value=1).value == 1"]
    checked = subprocess.run(probe, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
    if checked.returncode:
        if not install:
            raise ValueError("Run --prepare to install easydict==" + EASYDICT_VERSION)
        pip = [str(python), "-I", "-m", "pip", "--isolated", "install",
               "--no-cache-dir", "--index-url", "https://pypi.org/simple"]
        subprocess.run(pip + ["easydict==" + EASYDICT_VERSION], env=env,
                       check=True, timeout=600)
        subprocess.run(probe, env=env, check=True, timeout=120)
        subprocess.run([str(python), "-I", "-m", "pip", "check"],
                       env=env, check=True, timeout=120)
    if not marker.exists():
        with marker.open("x", encoding="utf-8") as stream:
            json.dump(expected, stream, indent=2)


def config_row(upstream, name):
    spec = SPECS[name]
    with (upstream / "args" / spec["csv"]).open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["ID"] == "7"]
    if rows != [spec["expected"]]:
        raise ValueError("Pinned official %s ID7 configuration differs" % name)
    return rows[0]


def resolve_data(root, name):
    root = root.resolve()
    if name == "cifar100":
        candidates = [root / "cifar-100-python",
                      root / "cifar100" / "cifar-100-python",
                      root / "CIFAR100" / "cifar-100-python"]
        valid = [p.resolve() for p in candidates
                 if all((p / f).is_file() for f in ("train", "test", "meta"))]
    else:
        candidates = [root / "imagenet-a", root,
                      root.parent / "imagenet-a" if root.name == "imagenet-a" else root]
        valid = []
        for p in candidates:
            base = p if p.name == "imagenet-a" else p / "imagenet-a"
            labels = []
            for split in ("train", "test"):
                folder = base / split
                dirs = sorted(x for x in folder.iterdir() if x.is_dir()) if folder.is_dir() else []
                if len(dirs) != 200 or not all(any(y.is_file() for y in d.rglob("*")) for d in dirs):
                    break
                labels.append([d.name for d in dirs])
            if len(labels) == 2 and labels[0] == labels[1]:
                valid.append(base.resolve())
    valid = sorted(set(valid))
    if len(valid) != 1:
        raise ValueError("Need exactly one existing %s dataset; found %s" % (name, valid))
    return valid[0]


def layout(folder, upstream, target, name):
    folder.mkdir(parents=True, exist_ok=True)
    data = folder / "data"
    data.mkdir(exist_ok=True)
    native.link_existing(folder / "args", upstream / "args")
    link = data / ("cifar-100-python" if name == "cifar100" else "imagenet-a")
    native.link_existing(link, target)


def dataset_record(name):
    import numpy as np
    from torchvision.datasets import CIFAR100, ImageFolder
    spec = SPECS[name]
    digest = hashlib.sha256()
    counts = {}
    if spec["kind"] == "cifar":
        for split, train in (("train", True), ("test", False)):
            ds = CIFAR100("./data", train=train, download=False)
            targets = np.asarray(ds.targets, dtype=np.int64)
            count = np.bincount(targets, minlength=100).tolist()
            digest.update(split.encode())
            digest.update(ds.data.tobytes())
            digest.update(targets.tobytes())
            counts[split] = count
    else:
        for split in ("train", "test"):
            ds = ImageFolder("./data/imagenet-a/" + split)
            count = [0] * spec["classes"]
            for path, target in ds.samples:
                path = Path(path)
                if not path.stat().st_size:
                    raise ValueError("Empty image: " + str(path))
                count[target] += 1
                digest.update((split + "/" + path.relative_to("data/imagenet-a").as_posix()
                               + ":" + common.sha(path)).encode())
            counts[split] = count
    if len(counts["train"]) != spec["classes"] or min(counts["train"] + counts["test"]) < 1:
        raise ValueError("Missing class data for " + name)
    return counts, digest.hexdigest()


def pretrained_source(name, model, timm_module):
    """Recover the actual source used by each unmodified upstream PETL path."""
    if name == "cifar100":
        cfg = timm_module.models.vision_transformer.default_cfgs.get(
            "vit_base_patch16_224_in21k")
    else:
        cfg = getattr(model, "pretrained_cfg", None) or getattr(model, "default_cfg", None)
    if not isinstance(cfg, dict):
        raise ValueError("Original model exposes no pretrained configuration")
    url = cfg.get("url", "")
    expected = SPECS[name]["checkpoint_file"]
    if (not url.startswith("https://storage.googleapis.com/vit_models/augreg/")
            or not url.endswith(expected)):
        raise ValueError("Unexpected original pretrained source: " + str(cfg))
    return url


def child(args):
    name = args.datasets[0]
    spec = SPECS[name]
    support = args.output_root / "_ranpac_support/original"
    upstream = args.output_root / "_ranpac_support/upstream"
    common.check_upstream(upstream)
    sys.path[:] = [str(upstream)] + [p for p in sys.path if p and
                    Path(p).resolve() not in (common.ROOT, common.ROOT / "tools")]
    if args.child == "native":
        sys.argv = [str(upstream / "main.py"), "-i", "7", "-d", spec["official"]]
        runpy.run_path(str(upstream / "main.py"), run_name="__main__")
        return
    import importlib.metadata as package_metadata
    import numpy as np
    import pandas
    import timm
    import torch
    import torchvision
    import tqdm
    import inc_net
    versions = {"python": platform.python_version(), "torch": torch.__version__,
                "torchvision": torchvision.__version__, "timm": timm.__version__,
                "pandas": pandas.__version__, "numpy": np.__version__,
                "tqdm": tqdm.__version__, "easydict": package_metadata.version("easydict"),
                "cuda": torch.version.cuda}
    if (sys.version_info[:2] != (3, 9) or torch.__version__ != "1.13.1+cu117"
            or torchvision.__version__ != "0.14.1+cu117" or timm.__version__ != "0.6.12"
            or pandas.__version__ != "1.5.2" or np.__version__ != "1.24.4"
            or tqdm.__version__ != "4.65.0"
            or versions["easydict"] != EASYDICT_VERSION or torch.version.cuda != "11.7"):
        raise ValueError("Private environment version mismatch: " + str(versions))
    counts, dataset_sha = dataset_record(name)
    print("ORIGINAL_PRETRAINED_DOWNLOAD_OR_CACHE_CHECK", flush=True)
    config = config_row(upstream, name)
    model = inc_net.get_convnet(dict(config))
    url = pretrained_source(name, model, timm)
    cached = support / "torch-cache/hub/checkpoints" / SPECS[name]["checkpoint_file"]
    if not cached.is_file():
        raise ValueError("Original pretrained file not found in isolated cache")
    order = np.random.RandomState(1993).permutation(spec["classes"]).tolist()
    record = {"dataset": name, "runtime": versions, "config": config,
              "class_order": order, "torch_seed_from_original_trainer": 1,
              "dataset_sha256": dataset_sha, "counts": counts,
              "pretrained_url": url, "pretrained_sha256": common.sha(cached),
              "revision": common.REVISION, "runner_sha256": common.sha(__file__),
              "limitations": ["Official dataset-specific ID7 configuration",
                              "Local dataset contents hashed but not authenticated against author archive",
                              "pip CUDA wheels and CPU thread cap differ from README conda setup"]}
    path = support / ("preflight-extra-" + name) / "metadata.json"
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError("Existing preflight metadata differs; preserved")
    if not path.exists():
        with path.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
    common.emit("RANPAC_EXTRA_PREFLIGHT", {"dataset": name,
                "train": sum(counts["train"]), "test": sum(counts["test"]),
                "pretrained_sha256": record["pretrained_sha256"]})


def summarize(folder, name, metadata):
    spec = SPECS[name]
    stem = spec["official"] + "_publish_7.csv"
    path = folder / "results/class_preds" / (spec["official"] + "_class_preds_publish_7.csv")
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"%s_task_%d" % (kind, i) for i in range(spec["tasks"])
                    for kind in ("pred", "true")}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Missing stage columns; run incomplete")
        records = list(reader)
    total = sum(metadata["counts"]["test"])
    if len(records) != total:
        raise ValueError("Expected %d prediction rows" % total)
    rows = []
    for i in range(spec["tasks"]):
        pairs = [(int(r["pred_task_%d" % i]), int(r["true_task_%d" % i])) for r in records]
        if any((p == -1) != (t == -1) for p, t in pairs):
            raise ValueError("Inconsistent prediction padding")
        pairs = [(p, t) for p, t in pairs if t != -1]
        row = exact_metrics([p for p, _ in pairs], [t for _, t in pairs],
                            i + 1, spec["increment"])
        row["stage"] = i + 1
        rows.append(row)
    with (folder / "results" / stem).open() as stream:
        official = list(csv.DictReader(stream))
    if len(official) != spec["tasks"]:
        raise ValueError("Original accuracy curve is incomplete")
    for row, saved in zip(rows, official):
        if (abs(row["Acc@1"] - float(saved["ave_acc"])) > .011 or
                abs(row["pooled_acc"] - float(saved["top1_total"])) > .011):
            raise ValueError("Exact predictions disagree with original CSV")
    expected_counts = [sum(metadata["counts"]["test"][c] for c in
                           metadata["class_order"][i:i + spec["increment"]])
                       for i in range(0, spec["classes"], spec["increment"])]
    if rows[-1]["task_counts"] != expected_counts:
        raise ValueError("Predictions disagree with dataset/order metadata")
    return {"stages": rows, "final": dict(rows[-1], **common.retention(rows)),
            "official_curve": official, "metadata": metadata}


def exact_metrics(predicted, targets, tasks, increment):
    """Recompute RanPAC's task-mean and pooled Top-1 for any class increment."""
    if len(predicted) != len(targets) or not targets:
        raise ValueError("Empty or mismatched predictions")
    groups = [[] for _ in range(tasks)]
    upper = tasks * increment
    for pred, target in zip(predicted, targets):
        pred, target = int(pred), int(target)
        if not 0 <= target < upper or not 0 <= pred < upper:
            raise ValueError("Prediction/target outside seen classes")
        groups[target // increment].append(pred == target)
    if any(not group for group in groups):
        raise ValueError("Missing task in evaluation")
    per_task = [100 * sum(group) / len(group) for group in groups]
    return {"Acc@1": statistics.mean(per_task),
            "pooled_acc": 100 * sum(map(sum, groups)) / len(targets),
            "per_task": per_task, "task_counts": list(map(len, groups))}


def completed(folder, name):
    path = folder / "summary.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if (result.get("metadata", {}).get("dataset") != name or
            result.get("metadata", {}).get("runner_sha256") != common.sha(__file__) or
            result.get("final", {}).get("stage") != 10):
        raise ValueError("Existing completed summary does not match this runner")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=tuple(SPECS),
                        default=["cifar100", "ima"])
    parser.add_argument("--tag", default="ranpac_original_extra_v1")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-minutes", type=float, default=90,
                        help="Per-dataset GPU timeout")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--child", choices=("inspect", "native"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("Duplicate datasets")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,80}", args.tag):
        parser.error("Unsafe tag")
    if not 1 <= args.cpu_threads <= 16 or not math.isfinite(args.max_minutes) or not 1 <= args.max_minutes <= 1440:
        parser.error("Invalid resource limits")
    args.output_root, args.data_root = args.output_root.resolve(), args.data_root.resolve()
    if args.child:
        if len(args.datasets) != 1:
            raise ValueError("Child requires exactly one dataset")
        child(args)
        return
    if sys.platform != "linux" or platform.machine() not in ("x86_64", "amd64"):
        raise RuntimeError("Run on the Linux x86_64 4090 host")
    if not args.output_root.is_dir():
        raise ValueError("Existing output-root required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "0"):
        raise ValueError("CUDA_VISIBLE_DEVICES must be unset or 0")
    upstream = args.output_root / "_ranpac_support/upstream"
    support = args.output_root / "_ranpac_support/original"
    support.mkdir(parents=True, exist_ok=True)
    env = native.isolated_env(support, args.cpu_threads)
    summaries = {}
    with common.acquire_shared_lock(support / ".prepare.lock"):
        if args.prepare and not upstream.exists():
            subprocess.run(["git", "clone", "--no-checkout", common.UPSTREAM, str(upstream)], check=True)
            subprocess.run(["git", "-C", str(upstream), "checkout", "--detach", common.REVISION], check=True)
        common.check_upstream(upstream)
        python = native.prepare(support, env) if args.prepare else support / "env-py39/bin/python"
        if not python.is_file():
            raise ValueError("Run --prepare first")
        ensure_extra_environment(support, env, python, args.prepare)
        for name in args.datasets:
            config_row(upstream, name)
            target = resolve_data(args.data_root, name)
            preflight = support / ("preflight-extra-" + name)
            layout(preflight, upstream, target, name)
            base = [str(python), "-B", "-u", str(Path(__file__).resolve()),
                    "--output-root", str(args.output_root), "--data-root", str(args.data_root),
                    "--datasets", name, "--tag", args.tag,
                    "--cpu-threads", str(args.cpu_threads), "--max-minutes", str(args.max_minutes)]
            subprocess.run(base + ["--child", "inspect"], cwd=preflight,
                           env=env, check=True, timeout=1800)
            folder = support / (args.tag + "-" + name)
            old = completed(folder, name) if folder.exists() else None
            if old:
                summaries[name] = old
                print("REUSE_COMPLETED_RANPAC=" + name, flush=True)
                continue
            if not args.run:
                print("PLAN_RANPAC_EXTRA=" + name, flush=True)
                continue
            common.idle_gpu_preflight()
            with common.acquire_shared_lock(args.output_root / ".paper_backbone_verifier.lock") as lock:
                common.idle_gpu_preflight()
                if folder.exists():
                    raise RuntimeError("Incomplete run folder preserved: " + str(folder))
                layout(folder, upstream, target, name)
                metadata = json.loads((preflight / "metadata.json").read_text())
                with (folder / "metadata.json").open("x") as stream:
                    json.dump(metadata, stream, indent=2)
                log = folder / "console.log"
                print("RANPAC_EXTRA_RUN_DIR=" + str(folder), flush=True)
                start = time.monotonic()
                native.run_bounded(base + ["--child", "native"], folder, env, lock,
                                   log, args.max_minutes)
                common.check_upstream(upstream)
                result = summarize(folder, name, metadata)
                result["elapsed_seconds"] = time.monotonic() - start
                with (folder / "summary.json").open("x") as stream:
                    json.dump(result, stream, indent=2, allow_nan=False)
                summaries[name] = result
                common.emit("RANPAC_EXTRA_FINAL", {"dataset": name,
                            "Acc@1": result["final"]["Acc@1"],
                            "Forgetting": result["final"]["Forgetting"],
                            "Backward": result["final"]["Backward"],
                            "elapsed_seconds": result["elapsed_seconds"]})
                print("RANPAC_EXTRA_COMPLETE=%s:10/10" % name, flush=True)
    if args.run and set(summaries) == set(args.datasets):
        out = support / args.tag
        out.mkdir(exist_ok=True)
        summary = {name: {"final": result["final"],
                          "elapsed_seconds": result.get("elapsed_seconds"),
                          "pretrained_sha256": result["metadata"]["pretrained_sha256"],
                          "config": result["metadata"]["config"]}
                   for name, result in summaries.items()}
        path = out / "summary.json"
        if path.exists() and json.loads(path.read_text()) != summary:
            raise ValueError("Existing aggregate summary differs; preserved")
        if not path.exists():
            with path.open("x") as stream:
                json.dump(summary, stream, indent=2, allow_nan=False)
        print("RANPAC_EXTRA_ALL_COMPLETE=%d/%d" % (len(summaries), len(args.datasets)), flush=True)
        print("SUMMARY=" + str(path), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("RANPAC_EXTRA_STOP: " + str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1)
