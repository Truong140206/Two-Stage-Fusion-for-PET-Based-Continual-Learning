#!/usr/bin/env python3
"""Bounded, isolated official RanPAC/SSF baseline on the paper's ImageNet-R.

No paper/checkpoint edits. --prepare installs only a private timm overlay.
--run is explicit. Defaults to CPU-only preflight. Completed logs are exclusive.
This is protocol-matched RanPAC, NOT a reproduction of the published table.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "https://github.com/McDonnell-Research-Lab/RanPAC.git"
REVISION = "cf4b301d18b0c27db030f4371b72b768005ae58a"
TIMM = "0.6.13"  # 0.6.12 API, with upstream's Python 3.11 dataclass fix.
EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def emit(name, value):
    print(name + "=" + json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def class_order(seed, shuffle, count=200):
    if not isinstance(shuffle, bool):
        raise ValueError("Checkpoint shuffle must be a boolean; do not guess class order")
    order = list(range(count))
    if shuffle:
        random.Random(seed).shuffle(order)
    return order


def exact_metrics(predicted, targets, tasks):
    if len(predicted) != len(targets) or not targets:
        raise ValueError("Empty or mismatched predictions")
    groups = [[] for _ in range(tasks)]
    for pred, target in zip(predicted, targets):
        if not 0 <= int(target) < tasks * 20 or not 0 <= int(pred) < tasks * 20:
            raise ValueError("Prediction/target outside seen classes")
        groups[int(target) // 20].append(int(pred) == int(target))
    if any(not group for group in groups):
        raise ValueError("Missing task in evaluation")
    per_task = [100 * sum(group) / len(group) for group in groups]
    return {"Acc@1": statistics.mean(per_task),
            "pooled_acc": 100 * sum(map(sum, groups)) / len(targets),
            "per_task": per_task, "task_counts": list(map(len, groups))}


def retention(rows):
    tasks = len(rows)
    if any(len(row["per_task"]) != i + 1 for i, row in enumerate(rows)):
        raise ValueError("Incomplete accuracy triangle")
    if tasks < 2:
        return {}
    final = rows[-1]["per_task"]
    return {
        "Forgetting": statistics.mean(
            max(rows[t]["per_task"][i] for t in range(i, tasks)) - final[i]
            for i in range(tasks - 1)),
        "Backward": statistics.mean(final[i] - rows[i]["per_task"][i]
                                    for i in range(tasks - 1))}


def resolve_data(root, explicit=None):
    candidates = [Path(explicit)] if explicit else [
        root / "imagenet-r", root / "imagenet-r" / "imagenet-r", root]
    valid = {}
    for candidate in candidates:
        candidate = candidate.resolve()
        if all((candidate / split).is_dir() for split in ("train", "test")):
            valid[str(candidate)] = candidate
    if len(valid) != 1:
        raise ValueError("Select the existing split with --data-path (folder containing train/test)")
    result = next(iter(valid.values()))
    names = []
    for split in ("train", "test"):
        classes = sorted(p.name for p in (result / split).iterdir() if p.is_dir())
        if len(classes) != 200:
            raise ValueError("Expected exactly 200 classes: " + str(result / split))
        names.append(classes)
    if names[0] != names[1]:
        raise ValueError("Train/test class names differ")
    return result


def check_upstream(path):
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                     text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"],
        text=True).strip()
    if actual != REVISION or dirty:
        raise ValueError("RanPAC source must be clean at pinned revision " + REVISION)


def acquire_shared_lock(path, locker=None):
    """Use the persistent flock inode shared by existing paper drivers.

    File existence is NOT ownership. Never unlink/replace the lock: that could
    allow another process to acquire a different inode while the first is held.
    """
    if locker is None:
        import fcntl
        locker = fcntl
    stream = path.open("a")
    try:
        locker.flock(stream, locker.LOCK_EX | locker.LOCK_NB)
    except BlockingIOError as exc:
        stream.close()
        raise RuntimeError("Another paper verifier holds the shared lock; "
                           "no run log created. Retry when it finishes; do not delete the lock") from exc
    except BaseException:
        stream.close()
        raise
    return stream


def prepare(support):
    support.mkdir(parents=True, exist_ok=True)
    upstream, runtime = support / "upstream", support / "runtime"
    if not upstream.exists():
        subprocess.run(["git", "clone", "--no-checkout", UPSTREAM, str(upstream)], check=True)
        subprocess.run(["git", "-C", str(upstream), "checkout", "--detach", REVISION], check=True)
    check_upstream(upstream)
    if not runtime.exists():
        # Never pip-install into the user's existing environment.
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                        "--no-deps", "--target", str(runtime), "timm==" + TIMM], check=True)
    return upstream, runtime


def gpu_guard(torch):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no training started")
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True)
    others = [p.strip() for p in processes.splitlines()
              if p.strip().isdigit() and int(p.strip()) != os.getpid()]
    if others:
        raise RuntimeError("GPU busy (other compute processes); retry later, do not stop them")
    if torch.cuda.mem_get_info(0)[0] < 16 * 1024**3:
        raise RuntimeError("Less than 16 GiB GPU memory free")


def worker(args):
    # No project modules imported: official utils/petl must win name resolution.
    sys.dont_write_bytecode = True
    support = args.output_root / "_ranpac_support"
    upstream, runtime = support / "upstream", support / "runtime"
    check_upstream(upstream)
    sys.path[:] = [str(runtime), str(upstream)] + [
        p for p in sys.path if p and Path(p).resolve() not in (ROOT, ROOT / "tools")]
    import numpy as np
    import torch
    import torchvision
    import timm
    from torchvision import datasets
    from utils import data as official_data
    from utils import data_manager as official_dm
    import inc_net
    import RanPAC
    from petl.vision_transformer_ssf import VisionTransformer

    if timm.__version__ != TIMM:
        raise ValueError("Private timm overlay missing/wrong: " + timm.__version__)
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    emit("RUNTIME", {"python": sys.version, "torch": torch.__version__,
                    "torchvision": torchvision.__version__, "timm": timm.__version__})

    # Load only the user's trusted baseline checkpoints, never download weights.
    run = args.output_root / ("imr_tii_original_10tasks_seed%d" % args.seed) / "checkpoint"
    weights = []
    records = []
    saved_first = None
    for stage in (1, 10):
        path = run / ("task%d_checkpoint.pth" % stage)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        saved = checkpoint.get("args")
        saved = saved if isinstance(saved, dict) else vars(saved)
        for key, expected in (("dataset", "Split-Imagenet-R"), ("seed", args.seed),
                              ("model", "vit_base_patch16_224"), ("num_tasks", 10)):
            if saved.get(key) != expected:
                raise ValueError("Checkpoint mismatch %s: %s" % (path, key))
        if saved.get("input_size", 224) != 224:
            raise ValueError("Unexpected input size")
        if saved_first is None:
            saved_first = saved
        elif saved.get("shuffle") != saved_first.get("shuffle"):
            raise ValueError("Class-order setting differs between checkpoints")
        base = {k: v for k, v in checkpoint["model"].items()
                if k in ("cls_token", "pos_embed")
                or k.startswith(("patch_embed.", "blocks.", "norm."))}
        if not base or any("lora" in k or "ssf" in k for k in base):
            raise ValueError("Unexpected backbone state")
        weights.append(base)
        records.append({"path": str(path), "sha256": sha(path)})
    if weights[0].keys() != weights[1].keys() or any(
            not torch.equal(v, weights[1][k]) for k, v in weights[0].items()):
        raise ValueError("Backbone is not fixed between task 1 and 10")
    base = weights[0]
    del weights
    order = class_order(args.seed, saved_first.get("shuffle", False))
    data_path = resolve_data(args.data_root, args.data_path)
    emit("BACKBONE_CHECK", {"status": "PASS", "checkpoint_stages": [1, 10],
                            "shuffle": saved_first.get("shuffle", False)})
    emit("DATA_HASH_START", str(data_path))

    # Only adapt data paths, paper class order and seed. Official transforms remain.
    image_sets = [datasets.ImageFolder(str(data_path / split)) for split in ("train", "test")]
    if image_sets[0].class_to_idx != image_sets[1].class_to_idx:
        raise ValueError("ImageFolder mappings differ")
    dataset_digest = hashlib.sha256()
    dataset_counts = {}
    for split, ds in zip(("train", "test"), image_sets):
        counts = [0] * 200
        for path, label in ds.samples:
            counts[label] += 1
            file = Path(path)
            if file.stat().st_size == 0:
                raise ValueError("Empty image: " + path)
            dataset_digest.update((file.relative_to(data_path).as_posix() + ":" + sha(file)).encode())
        if min(counts) == 0:
            raise ValueError("An image class is empty")
        dataset_counts[split] = counts
        emit("DATA_HASH_SPLIT", {"split": split, "images": sum(counts)})

    def load_existing(idata):
        idata.train_data = np.array([p for p, _ in image_sets[0].samples])
        idata.train_targets = np.array(image_sets[0].targets)
        idata.test_data = np.array([p for p, _ in image_sets[1].samples])
        idata.test_targets = np.array(image_sets[1].targets)
    official_data.iImageNetR.download_data = load_existing
    official_data.iImageNetR.class_order = order
    manager = official_dm.DataManager("imagenetr", False, args.seed, 20, 20, False)
    assert manager._class_order == order and manager.nb_tasks == 10

    def fixed_backbone(config, pretrained=False):
        if config["model_name"] != "ssf":
            raise ValueError("Only official ImageNet-R SSF baseline is supported")
        net = VisionTransformer(num_classes=0)
        expected = {k for k in net.state_dict() if "ssf_scale" not in k and "ssf_shift" not in k}
        if expected != set(base):
            raise ValueError("Backbone keys differ: " + str(sorted(expected.symmetric_difference(base))))
        missing, unexpected = net.load_state_dict(base, strict=False)
        if unexpected or any("ssf_scale" not in k and "ssf_shift" not in k for k in missing):
            raise ValueError("Unexpected missing backbone weights")
        if any(not torch.equal(net.state_dict()[k], v) for k, v in base.items()):
            raise ValueError("Backbone tensor equality failed")
        net.out_dim = 768
        return net.eval()
    inc_net.get_convnet = fixed_backbone
    # Validate constructor/weight mapping on CPU even in preflight.
    probe = fixed_backbone({"model_name": "ssf"})
    del probe
    with (upstream / "args" / "imagenetr_publish.csv").open() as handle:
        official = next(row for row in csv.DictReader(handle) if row["ID"] == "7")
    expected_official = {"model_name": "ssf", "batch_size": "48", "tuned_epoch": "20",
                         "M": "10000", "body_lr": "0.01", "use_RP": "True",
                         "init_cls": "20", "increment": "20", "use_input_norm": "False"}
    if any(official[k] != v for k, v in expected_official.items()):
        raise ValueError("Pinned official configuration differs from expected")
    metadata = {"upstream_revision": REVISION, "runner_sha256": sha(__file__),
                "seed": args.seed, "class_order": order, "checkpoints": records,
                "dataset_sha256": dataset_digest.hexdigest(), "counts": dataset_counts,
                "official_row": official, "cpu_threads": args.cpu_threads,
                "deviations": ["paper backbone tensors, not upstream weight download",
                               "paper class order and seed for all RNGs",
                               "timm 0.6.13 compatibility, not upstream 0.6.12",
                               "4 loader workers; exact task-balanced metrics",
                               "upstream training/PETL/17-value lambda search unchanged"]}
    emit("RANPAC_PREFLIGHT_PASS", metadata)
    if not args.run:
        return
    gpu_guard(torch)
    RanPAC.num_workers = 4
    torch.cuda.manual_seed_all(args.seed)
    # Reset RNG after CPU validation so the check does not change initialization.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    config = dict(model_name="ssf", convnet_type="pretrained_vit_b16_224_ssf",
                  dataset="imagenetr", device=[torch.device("cuda:0")],
                  batch_size=48, tuned_epoch=20, body_lr=0.01, head_lr=0.01,
                  weight_decay=0.0005, min_lr=0.0, use_RP=True, M=10000,
                  use_input_norm=False, seed=args.seed, do_not_save=True)
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        stream=sys.stdout)
    model = RanPAC.Learner(config)
    model.dil_init = False
    model.is_dil = False
    original_train = model._init_train
    def timed_train(*pos, **kw):
        start = time.monotonic()
        emit("PHASE1_START", {"epochs": 20, "batch_size": 48})
        result = original_train(*pos, **kw)
        torch.cuda.synchronize()
        emit("PHASE1_SECONDS", time.monotonic() - start)
        return result
    model._init_train = timed_train
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for stage in range(10):
        stage_start = time.monotonic()
        emit("RANPAC_TASK_START", stage + 1)
        model.incremental_train(manager)
        _, _, pred, true = model.eval_task()
        row = exact_metrics(pred.tolist(), true.tolist(), stage + 1)
        row.update(stage=stage + 1, seconds=time.monotonic() - stage_start)
        rows.append(row)
        emit("RANPAC_STAGE", row)
        model.after_task()
    final = dict(rows[-1], **retention(rows))
    result = {"metadata": metadata, "stages": rows, "final": final,
              "elapsed_seconds": time.monotonic() - start,
              "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
              "note": "single seed; not published-table reproduction; no measured baseline overhead"}
    # Check inputs/source were not mutated while running.
    check_upstream(upstream)
    if any(sha(record["path"]) != record["sha256"] for record in records):
        raise RuntimeError("Checkpoint changed during run")
    summary = args.output_root / (args.tag + "__summary.json")
    with summary.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    emit("RANPAC_FINAL", final)
    print("RANPAC_COMPLETE=imr:10/10", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, help="Exact existing folder containing train/test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="ranpac_imr42_v1")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-minutes", type=float, default=60)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,80}", args.tag):
        parser.error("Unsafe tag")
    if not math.isfinite(args.max_minutes) or not 1 <= args.max_minutes <= 1440:
        parser.error("--max-minutes must be 1..1440")
    if not 1 <= args.cpu_threads <= 16 or not 0 <= args.seed < 2**32:
        parser.error("Invalid threads or seed")
    args.output_root = args.output_root.resolve()
    args.data_root = args.data_root.resolve()
    if args.worker:
        worker(args)
        return
    if os.name != "posix":
        raise RuntimeError("Launcher is for the Linux 4090 host; local unit tests are CPU-only")
    if not args.output_root.is_dir():
        raise ValueError("Existing output-root required")
    support = args.output_root / "_ranpac_support"
    if args.prepare:
        prepare(support)
    else:
        check_upstream(support / "upstream")
    resolve_data(args.data_root, args.data_path)
    log = args.output_root / (args.tag + ".log")
    summary = args.output_root / (args.tag + "__summary.json")
    if args.run and (log.exists() or summary.exists()):
        raise FileExistsError("Existing run preserved; inspect it or use a new --tag")
    if args.run:
        if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "0"):
            raise ValueError("This single-GPU runner requires CUDA_VISIBLE_DEVICES=0 or unset")
        idle_gpu_preflight()
    cmd = [sys.executable, "-B", "-u", str(Path(__file__).resolve()),
           *[x for x in sys.argv[1:] if x != "--prepare"], "--worker"]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=str(args.cpu_threads),
               MKL_NUM_THREADS=str(args.cpu_threads), OPENBLAS_NUM_THREADS=str(args.cpu_threads))
    # Reuse the existing paper-evaluation lock; never stop someone else's job.
    lock = acquire_shared_lock(args.output_root / ".paper_backbone_verifier.lock")
    process = None
    handle = None
    try:
        if args.run:
            handle = log.open("x", encoding="utf-8")
            emit("LOG", str(log))
        print("LIMIT_MINUTES=" + str(args.max_minutes), flush=True)
        process = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True,
                                   pass_fds=(lock.fileno(),))
        try:
            code = process.wait(timeout=args.max_minutes * 60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            print("RANPAC_TIMEOUT: only this run stopped; partial results are not final evidence")
            code = 124
        if handle:
            handle.write("\nRANPAC_EXIT_CODE=%d\n" % code)
            handle.flush()
        if code:
            raise RuntimeError("RanPAC did not complete, exit=%d; see %s" % (code, log))
        if args.run:
            print("SUMMARY=" + str(summary), flush=True)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if handle:
            handle.close()
        lock.close()


def idle_gpu_preflight():
    """Read-only check before opening a log; no parent CUDA context."""
    pids = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True)
    if any(line.strip().isdigit() for line in pids.splitlines()):
        raise RuntimeError("GPU busy; no run log created. Retry the same command when free")
    free = subprocess.check_output(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.free",
         "--format=csv,noheader,nounits"], text=True).strip()
    if int(free) < 16 * 1024:
        raise RuntimeError("GPU 0 has less than 16 GiB free; no run started")
    available = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    if available < 12 * 1024**2:
        raise RuntimeError("Less than 12 GiB system RAM available; retry later")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("RANPAC_STOP: " + str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1)
