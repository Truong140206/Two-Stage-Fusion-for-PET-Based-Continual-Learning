#!/usr/bin/env python3
"""Run unmodified RanPAC main.py, ImageNet-R ID7, in a private Python 3.9 env.

The old matched-backbone experiment and paper are never modified.
--prepare installs/downloads only; --run explicitly starts the GPU job.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import re
import runpy
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.request

try:
    from tools import run_ranpac_baseline as common
except ModuleNotFoundError:
    import run_ranpac_baseline as common

MAMBA_URL = ("https://api.anaconda.org/download/conda-forge/micromamba/2.3.2/"
             "linux-64/micromamba-2.3.2-0.tar.bz2")
MAMBA_SHA = "5512233cdd8564a671626081026dc861537a963baa06706baab08fac6f3bb9d2"
PACKAGES = ["numpy==1.24.4", "pandas==1.5.2", "tqdm==4.65.0", "timm==0.6.12",
            "Pillow==9.5.0", "PyYAML==6.0.1", "huggingface-hub==0.16.4"]
EXPECTED = {"ID": "7", "dataset": "imagenetr", "shuffle": "True", "init_cls": "20",
            "increment": "20", "model_name": "ssf",
            "convnet_type": "pretrained_vit_b16_224_ssf", "device": "0", "seed": "1993",
            "batch_size": "48", "tuned_epoch": "20", "body_lr": "0.01",
            "head_lr": "0.01", "weight_decay": "0.0005", "min_lr": "0.0",
            "use_RP": "True", "M": "10000", "use_input_norm": "False"}


def config_row(upstream):
    with (upstream / "args/imagenetr_publish.csv").open() as stream:
        rows = [r for r in csv.DictReader(stream) if r["ID"] == "7"]
    if rows != [EXPECTED]:
        raise ValueError("Pinned official ImageNet-R ID7 configuration differs")
    return rows[0]


def link_existing(link, target):
    """Never replace an existing file/directory/link."""
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise ValueError("Existing link points elsewhere: " + str(link))
    elif link.exists():
        raise ValueError("Existing non-link preserved: " + str(link))
    else:
        link.symlink_to(target.resolve(), target_is_directory=True)


def layout(folder, upstream, data):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "data").mkdir(exist_ok=True)
    link_existing(folder / "args", upstream / "args")
    link_existing(folder / "data/imagenet-r", data)


def isolated_env(support, threads):
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV",
                "CONDA_PREFIX", "LD_LIBRARY_PATH", "LD_PRELOAD"):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads),
               TORCH_HOME=str(support / "torch-cache"),
               HF_HOME=str(support / "hf-cache"),
               MAMBA_ROOT_PREFIX=str(support / "mamba-root"),
               PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_CONFIG_FILE=os.devnull)
    return env


def download_mamba(support):
    archive = support / "micromamba-2.3.2-0.tar.bz2"
    if not archive.exists():
        partial = archive.with_suffix(".partial")
        with urllib.request.urlopen(MAMBA_URL, timeout=120) as src, partial.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        if common.sha(partial) != MAMBA_SHA:
            raise ValueError("Micromamba download checksum mismatch; not executed")
        partial.replace(archive)
    if common.sha(archive) != MAMBA_SHA:
        raise ValueError("Micromamba archive checksum mismatch; not executed")
    executable = support / "micromamba"
    with tarfile.open(archive, "r:bz2") as tar:
        member = tar.getmember("bin/micromamba")
        if not member.isfile():
            raise ValueError("Unexpected micromamba archive")
        payload = tar.extractfile(member).read()
    if executable.exists():
        if executable.read_bytes() != payload:
            raise ValueError("Existing micromamba differs; preserved, not executed")
    else:
        with executable.open("xb") as stream:
            stream.write(payload)
        executable.chmod(0o755)
    return executable


def prepare(support, env):
    """Private prefix; no activation, shell init, or current-venv package changes."""
    support.mkdir(parents=True, exist_ok=True)
    python = support / "env-py39/bin/python"
    ready = support / "environment-ready.json"
    if ready.exists():
        if not python.is_file():
            raise ValueError("Environment marker exists but Python is missing")
        return python
    if shutil.disk_usage(support).free < 8 * 1024**3:
        raise RuntimeError("Preparation needs at least 8 GiB free disk")
    mamba = download_mamba(support)
    if not python.exists():
        subprocess.run([str(mamba), "--no-rc", "create", "-y", "-p",
                        str(support / "env-py39"), "--override-channels",
                        "-c", "conda-forge", "python=3.9", "pip=24.3.1"],
                       env=env, check=True, timeout=1200)
    pip = [str(python), "-I", "-m", "pip", "--isolated", "install", "--no-cache-dir"]
    subprocess.run(pip + ["--index-url", "https://download.pytorch.org/whl/cu117",
                         "torch==1.13.1+cu117", "torchvision==0.14.1+cu117",
                         "torchaudio==0.13.1+cu117"],
                   env=env, check=True, timeout=2400)
    subprocess.run(pip + ["--index-url", "https://pypi.org/simple"] + PACKAGES,
                   env=env, check=True, timeout=1200)
    subprocess.run([str(python), "-I", "-m", "pip", "check"],
                   env=env, check=True, timeout=120)
    freeze = subprocess.check_output([str(python), "-I", "-m", "pip", "freeze"],
                                     env=env, text=True)
    with ready.open("x", encoding="utf-8") as stream:
        json.dump({"packages": freeze.splitlines(), "micromamba_sha256": MAMBA_SHA,
                   "packaging": "pip CUDA 11.7 wheels; README uses conda packages"},
                  stream, indent=2)
    return python


def child(args):
    """Only path setup; execute original main/trainer without algorithm patches."""
    support = args.output_root / "_ranpac_support/original"
    upstream = args.output_root / "_ranpac_support/upstream"
    common.check_upstream(upstream)
    sys.path[:] = [str(upstream)] + [p for p in sys.path if p and
                    Path(p).resolve() not in (common.ROOT, common.ROOT / "tools")]
    if args.child == "native":
        sys.argv = [str(upstream / "main.py"), "-i", "7", "-d", "imagenetr"]
        runpy.run_path(str(upstream / "main.py"), run_name="__main__")
        return
    import numpy as np
    import torch
    import torchvision
    import timm
    import pandas
    import tqdm
    from torchvision.datasets import ImageFolder
    import inc_net
    import trainer
    versions = {"python": platform.python_version(), "torch": torch.__version__,
                "torchvision": torchvision.__version__, "timm": timm.__version__,
                "pandas": pandas.__version__, "numpy": np.__version__, "tqdm": tqdm.__version__,
                "cuda": torch.version.cuda}
    if (sys.version_info[:2] != (3, 9) or torch.__version__ != "1.13.1+cu117"
            or torchvision.__version__ != "0.14.1+cu117" or timm.__version__ != "0.6.12"
            or pandas.__version__ != "1.5.2" or np.__version__ != "1.24.4"
            or tqdm.__version__ != "4.65.0" or torch.version.cuda != "11.7"):
        raise ValueError("Private environment version mismatch: " + str(versions))
    common.emit("ORIGINAL_RUNTIME", versions)
    data = common.resolve_data(args.data_root)
    digest = __import__("hashlib").sha256()
    counts = {}
    sets = [ImageFolder(str(data / split)) for split in ("train", "test")]
    if sets[0].class_to_idx != sets[1].class_to_idx:
        raise ValueError("Train/test class mapping differs")
    for split, ds in zip(("train", "test"), sets):
        count = [0] * 200
        for path, target in ds.samples:
            path = Path(path)
            if not path.stat().st_size:
                raise ValueError("Empty image: " + str(path))
            count[target] += 1
            digest.update((path.relative_to(data).as_posix() + ":" + common.sha(path)).encode())
        if min(count) < 1:
            raise ValueError("Empty class")
        counts[split] = count
    if [sum(counts[s]) for s in ("train", "test")] != [24000, 6000]:
        raise ValueError("Expected 24000/6000 ImageNet-R split")
    # A separate CPU process downloads the original pretrained model. Its RNG
    # state cannot affect the subsequent main.py process.
    print("ORIGINAL_PRETRAINED_DOWNLOAD_OR_CACHE_CHECK", flush=True)
    model = inc_net.get_convnet(dict(EXPECTED))
    cfg = getattr(model, "pretrained_cfg", model.default_cfg)
    if "imagenet2012" not in cfg.get("url", "") or not cfg["url"].endswith(".npz"):
        raise ValueError("Unexpected original pretrained source: " + str(cfg))
    cached = support / "torch-cache/hub/checkpoints" / cfg["url"].split("/")[-1]
    if not cached.is_file():
        raise ValueError("Original pretrained file not found in isolated cache")
    # Record original DataManager permutation without changing the actual run.
    order = np.random.RandomState(1993).permutation(200).tolist()
    record = {"runtime": versions, "config": config_row(upstream),
              "class_order": order, "torch_seed_from_original_trainer": 1,
              "dataset_sha256": digest.hexdigest(), "counts": counts,
              "data_path": str(data), "pretrained_url": cfg["url"],
              "pretrained_sha256": common.sha(cached), "revision": common.REVISION,
              "runner_sha256": common.sha(__file__),
              "limitations": ["Existing split is hashed, not authenticated against author archive",
                              "pip CUDA wheels instead of README conda packaging",
                              "CPU thread cap; otherwise original main/trainer/config unchanged"]}
    with (support / "preflight/metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    common.check_upstream(upstream)
    print("ORIGINAL_PREFLIGHT=PASS", flush=True)


def summarize(folder):
    """Read official saved predictions; never alter trainer or metric output."""
    path = folder / "results/class_preds/imagenetr_class_preds_publish_7.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {f"{kind}_task_{i}" for i in range(10) for kind in ("pred", "true")}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Missing stage columns; run incomplete")
        records = list(reader)
    if len(records) != 6000:
        raise ValueError("Expected 6000 prediction rows")
    rows = []
    for i in range(10):
        pairs = [(int(r[f"pred_task_{i}"]), int(r[f"true_task_{i}"])) for r in records]
        if any((p == -1) != (t == -1) for p, t in pairs):
            raise ValueError("Inconsistent prediction padding")
        pairs = [(p, t) for p, t in pairs if t != -1]
        if i == 9 and len(pairs) != 6000:
            raise ValueError("Final stage must contain all 6000 images")
        row = common.exact_metrics([p for p, _ in pairs], [t for _, t in pairs], i + 1)
        row["stage"] = i + 1
        rows.append(row)
    if any(row["task_counts"] != rows[-1]["task_counts"][:row["stage"]] for row in rows):
        raise ValueError("Per-task sample counts changed between stages")
    with (folder / "results/imagenetr_publish_7.csv").open() as stream:
        official = list(csv.DictReader(stream))
    if len(official) != 10:
        raise ValueError("Original accuracy curve is incomplete")
    for row, saved in zip(rows, official):
        if (abs(row["Acc@1"] - float(saved["ave_acc"])) > .011 or
                abs(row["pooled_acc"] - float(saved["top1_total"])) > .011):
            raise ValueError("Exact predictions disagree with original accuracy CSV")
    return {"stages": rows, "final": dict(rows[-1], **common.retention(rows)),
            "official_curve": official}


def stop_owned_process(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def run_bounded(cmd, cwd, env, lock, log, minutes):
    process = None
    with log.open("x", encoding="utf-8") as stream:
        try:
            process = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True,
                                       pass_fds=(lock.fileno(),))
            try:
                code = process.wait(timeout=minutes * 60)
            except subprocess.TimeoutExpired:
                code = 124
            finally:
                stop_owned_process(process)
            stream.write("\nRANPAC_ORIGINAL_EXIT_CODE=%d\n" % code)
            if code:
                raise RuntimeError("Original run failed/timeout; inspect " + str(log))
        finally:
            stop_owned_process(process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tag", default="ranpac_original_imr_id7_v1")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-minutes", type=float, default=60)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--child", choices=("inspect", "native"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,80}", args.tag):
        parser.error("Unsafe tag")
    if not 1 <= args.cpu_threads <= 16 or not math.isfinite(args.max_minutes) or not 1 <= args.max_minutes <= 1440:
        parser.error("Invalid threads/time limit")
    args.output_root = args.output_root.resolve()
    args.data_root = args.data_root.resolve()
    if args.child:
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
    env = isolated_env(support, args.cpu_threads)
    # Separate preparation lock, persistent flock, also protects downloads.
    with common.acquire_shared_lock(support / ".prepare.lock"):
        if args.prepare and not upstream.exists():
            subprocess.run(["git", "clone", "--no-checkout", common.UPSTREAM, str(upstream)], check=True)
            subprocess.run(["git", "-C", str(upstream), "checkout", "--detach", common.REVISION], check=True)
        common.check_upstream(upstream)
        config_row(upstream)
        data = common.resolve_data(args.data_root)
        folder = support / args.tag
        if folder.exists():
            raise FileExistsError("Existing run folder preserved: " + str(folder))
        python = prepare(support, env) if args.prepare else support / "env-py39/bin/python"
        if not python.is_file():
            raise ValueError("Run --prepare first")
        layout(support / "preflight", upstream, data)
        base = [str(python), "-B", "-u", str(Path(__file__).resolve()),
                "--output-root", str(args.output_root), "--data-root", str(args.data_root)]
        subprocess.run(base + ["--child", "inspect"], cwd=support / "preflight",
                       env=env, check=True, timeout=1800)
        if not args.run:
            return
        common.idle_gpu_preflight()
        with common.acquire_shared_lock(args.output_root / ".paper_backbone_verifier.lock") as lock:
            # No CUDA context in parent; check again after acquiring shared lock.
            common.idle_gpu_preflight()
            layout(folder, upstream, data)
            metadata = json.loads((support / "preflight/metadata.json").read_text())
            with (folder / "metadata.json").open("x") as stream:
                json.dump(metadata, stream, indent=2)
            log = folder / "console.log"
            print("ORIGINAL_RUN_DIR=" + str(folder), flush=True)
            print("ORIGINAL_CONSOLE=" + str(log), flush=True)
            start = time.monotonic()
            run_bounded(base + ["--child", "native"], folder, env, lock, log, args.max_minutes)
            common.check_upstream(upstream)
            result = summarize(folder)
            expected_counts = [sum(metadata["counts"]["test"][c] for c in
                                   metadata["class_order"][i:i + 20]) for i in range(0, 200, 20)]
            if result["final"]["task_counts"] != expected_counts:
                raise ValueError("Predictions disagree with preflight dataset/class order")
            result.update(metadata=metadata, elapsed_seconds=time.monotonic() - start)
            with (folder / "summary.json").open("x") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
            common.emit("RANPAC_ORIGINAL_FINAL", result["final"])
            print("RANPAC_ORIGINAL_COMPLETE=imr:10/10", flush=True)
            print("SUMMARY=" + str(folder / "summary.json"), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("RANPAC_ORIGINAL_STOP: " + str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1)
