#!/usr/bin/env python3
"""Check checkpoint provenance and re-evaluate without replacing old logs/weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from tools.audit_weight_evidence import FINAL, NUM, paired
from protocols import exemplar_free_violations

DATASETS = {
    'imr': ('Split-Imagenet-R', 'imr_lora'),
    'cifar100': ('Split-CIFAR100', 'cifar100_lora'),
    'cub200': ('Split-CUB200', 'imr_lora'),
}
CORE = ('Acc@task', 'Acc@1', 'Acc@5', 'Loss')
RETENTION = ('Forgetting', 'Backward')
RP_PREFIX = '_eval_rp_lora_d10000_relu_l10000_nnone_t0_b0p0_p1_inone_c0_ra0ls0_f1d1'
RP_SUFFIX = 'lsw0p0c0ca0cw{beta}sh1p0m1gmargin'
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm',
                    '.tif', '.tiff', '.webp'}


def _imr_split_problem(data_path, expected_classes):
    """Read-only layout check; never extract, move images, or create a split."""
    split_root = data_path / 'imagenet-r'
    labels = []
    for split in ('train', 'test'):
        folder = split_root / split
        if not folder.is_dir():
            return 'missing ' + str(folder)
        classes = sorted(p for p in folder.iterdir() if p.is_dir())
        if len(classes) != expected_classes:
            return '%s: expected %d class folders, found %d' % (
                folder, expected_classes, len(classes))
        for class_dir in classes:
            if not any(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                       for p in class_dir.rglob('*')):
                return 'no image files in ' + str(class_dir)
        labels.append([p.name for p in classes])
    if labels[0] != labels[1]:
        return 'train/test class names differ under ' + str(split_root)
    return None


def resolve_imr_data_path(data_root, exact_path=None, expected_classes=200):
    """Return the parent that Imagenet_R will append /imagenet-r to.

    An archive or an empty nested directory is not evidence of a usable split.
    Multiple prepared splits are ambiguous: require an explicit --data-path.
    """
    root = Path(data_root).expanduser().resolve()
    if exact_path is not None:
        candidates = [Path(exact_path).expanduser().resolve()]
    else:
        candidates = [root, root / 'imagenet-r']
        if root.name == 'imagenet-r':
            candidates.append(root.parent)
    valid, failures, checked = [], [], set()
    for candidate in candidates:
        # Deduplicate the actual split root, including symlink aliases.
        split_root = (candidate / 'imagenet-r').resolve()
        if split_root in checked:
            continue
        checked.add(split_root)
        problem = _imr_split_problem(candidate, expected_classes)
        if problem is None:
            valid.append(candidate)
        else:
            failures.append(problem)
    if len(valid) == 1:
        return valid[0]
    if len(valid) > 1:
        raise ValueError('Multiple prepared ImageNet-R splits found. Select the '
                         'historical one with --data-path: ' + ', '.join(map(str, valid)))
    raise ValueError('No prepared ImageNet-R train/test split found. '
                     'Nothing was downloaded, extracted or re-split. '
                     'Set --data-path to the parent of the existing imagenet-r folder.\n  '
                     + '\n  '.join(failures))


def log_name(run, arm, tag=None):
    if arm == 'baseline':
        name = run + '_eval_conventional'
    else:
        w, beta = ('1p0', '0p0') if arm == 'identity' else ('0p7', '0p5')
        name = run + RP_PREFIX + 'w' + w + RP_SUFFIX.format(beta=beta)
    return name + ('__' + tag if tag else '') + '.log'


def read_stages(path, tasks=10):
    rows = {}
    for line in Path(path).read_text(encoding='utf-8', errors='replace').splitlines():
        match = FINAL.search(line)
        if not match:
            continue
        stage = int(match.group(1))
        values = {k: float(v) for k, v in NUM.findall(match.group(2))}
        required = CORE + (RETENTION if stage > 1 else ())
        if not all(k in values for k in required):
            raise ValueError('Missing metrics: %s task %s' % (path, stage))
        import math
        if not all(math.isfinite(values[k]) for k in required):
            raise ValueError('Non-finite metrics: ' + str(path))
        if stage in rows and any(rows[stage][k] != values[k] for k in required):
            raise ValueError('Conflicting duplicate task row: ' + str(path))
        rows[stage] = values
    if set(rows) != set(range(1, tasks + 1)):
        raise ValueError('Incomplete stage sequence: %s has %s' % (path, sorted(rows)))
    return rows


def compare(left, right, stages, metrics, tolerance=1e-4, label='comparison'):
    passed = True
    for stage in stages:
        for metric in metrics:
            if stage == 1 and metric in RETENTION:
                continue
            delta = right[stage][metric] - left[stage][metric]
            ok = abs(delta) <= tolerance + 1e-10
            if not ok:
                print('%s task%d %s delta=%+.6f FAIL' % (label, stage, metric, delta))
            passed = passed and ok
    print(label + '=' + ('PASS' if passed else 'FAIL'))
    return passed


def check_saved_args(saved, dataset, seed, label):
    if saved is None:
        raise ValueError(label + ': checkpoint has no saved args')
    from types import SimpleNamespace
    if isinstance(saved, dict):
        saved = SimpleNamespace(**saved)
    expected = dict(dataset=dataset, seed=seed, num_tasks=10,
                    model='vit_base_patch16_224')
    for key, value in expected.items():
        actual = getattr(saved, key, None)
        if actual != value:
            raise ValueError('%s: %s=%r; expected %r' % (label, key, actual, value))
    violations = exemplar_free_violations(saved)
    if violations:
        raise ValueError(label + ': ' + '; '.join(violations))


def checkpoint_manifest(tii, lora, dataset, seed):
    # Only load your own trusted research checkpoints: weights_only=False
    # is needed because these contain an argparse Namespace.
    import torch
    records = []
    for role, root in [('tii', tii), ('lora', lora)]:
        for stage in range(1, 11):
            path = root / 'checkpoint' / ('task%d_checkpoint.pth' % stage)
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            check_saved_args(checkpoint.get('args'), dataset, seed, str(path))
            if checkpoint.get('real_feature_memory'):
                raise ValueError(str(path) + ': per-example real features present')
            if role == 'lora':
                ranks = [v.shape[-1] for k, v in checkpoint['model'].items()
                         if k.endswith('lora_layer.k_lora_A')]
                if ranks != [8]:
                    raise ValueError(str(path) + ': expected LoRA rank 8')
            del checkpoint
            with path.open('rb') as handle:
                digest = hashlib.file_digest(handle, 'sha256').hexdigest()
            records.append({'path': str(path.resolve()), 'sha256': digest})
    print('CHECKPOINT_DATASET_SEED_MODEL_PROTOCOL=PASS', dataset, seed)
    return records


def source_digest():
    paths = list(REPO.glob('*.py'))
    for folder in ('configs', 'engines', 'trainers', 'vits', 'peft', 'continual_datasets'):
        paths.extend((REPO / folder).rglob('*.py'))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(REPO).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def command(args, seed, arm, tii, lora):
    dataset, config = DATASETS[args.dataset]
    data_path = getattr(args, 'resolved_data_path', None)
    if data_path is None:
        data_path = (resolve_imr_data_path(args.data_root, getattr(args, 'data_path', None))
                     if args.dataset == 'imr'
                     else (getattr(args, 'data_path', None) or args.data_root))
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node=1',
           '--master_port=' + str(args.port), 'main.py', config,
           '--model', 'vit_base_patch16_224', '--original_model', 'vit_base_patch16_224',
           '--batch-size', str(args.batch_size), '--epochs', '1',
           '--data-path', str(data_path), '--seed', str(seed), '--lr', '0.03',
           '--con', '0.2', '--lora_rank', '8', '--En', 'gen', '--tau', '-10',
           '--K', '5', '--sched', 'cosine', '--dataset', dataset,
           '--lora_momentum', '0.4', '--lora_type', 'hide',
           '--trained_original_model', str(tii), '--num_tasks', '10',
           '--strict_exemplar_free', '--eval', '--output_dir', str(lora)]
    if arm != 'baseline':
        w, beta = ('1.0', '0.0') if arm == 'identity' else ('0.7', '0.5')
        cmd += ['--rp_head', '--rp_dim', '10000', '--rp_activation', 'relu',
                '--rp_lambda', '10000', '--rp_feature_source', 'lora',
                '--rp_normalize', 'none', '--rp_lora_task', '0',
                '--rp_logit_blend', '0.0', '--rp_input_norm', 'none',
                '--rp_pin_extractor', '--rp_route_fusion', '--rp_route_fusion_drm',
                '--rp_route_fusion_weight', w, '--rp_route_fusion_ls_weight', '0.0',
                '--rp_class_fusion_weight', beta, '--rp_class_fusion_sharpen', '1.0',
                '--rp_class_fusion_min_tasks', '1', '--rp_class_fusion_gate', 'margin',
                '--rp_fusion_ramp', '0.0', '--rp_fusion_ramp_scope', 'both']
    return cmd


def check_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no evaluation was started.')
    device = int(os.environ.get('LOCAL_RANK', '0'))
    free, _ = torch.cuda.mem_get_info(device)
    if free < 5 * 1024 ** 3:
        raise RuntimeError('GPU has less than 5 GiB free. Retry later; do not stop other users jobs.')


def run_or_read(path, cmd, metadata, execute):
    header = 'VERIFICATION_META=' + json.dumps(metadata, sort_keys=True)
    if path.exists():
        text = path.read_text(encoding='utf-8', errors='replace')
        if text.splitlines()[:1] != [header] or 'VERIFICATION_EXIT_CODE=0' not in text.splitlines():
            raise ValueError('Existing log has different provenance or is incomplete: %s. '
                             'Keep it; use a new --tag.' % path)
        return read_stages(path)
    if not execute:
        raise FileNotFoundError('Missing corrected log: %s. Use --run to evaluate.' % path)
    check_gpu()
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    print('RUN', path.name, flush=True)
    # Exclusive creation: never overwrite a log, even an incomplete one.
    with path.open('x', encoding='utf-8') as handle:
        handle.write(header + '\n')
        handle.write('COMMAND=' + json.dumps(cmd) + '\n')
        handle.flush()
        process = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   encoding='utf-8', errors='replace')
        show_traceback = False
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            show_traceback = show_traceback or 'Traceback' in line
            if FINAL.search(line) or show_traceback:
                print(line.rstrip(), flush=True)
        code = process.wait()
        handle.write('VERIFICATION_EXIT_CODE=%d\n' % code)
    if code:
        raise RuntimeError('Evaluation failed (%d): %s' % (code, path))
    return read_stages(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--data-path', type=Path,
                        help='Exact loader root; ImageNet-R appends /imagenet-r to this path')
    parser.add_argument('--tag', default='maskfix_verify_v1')
    parser.add_argument('--port', type=int, default=29558)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--run', action='store_true', help='Run missing evaluations; never train')
    parser.add_argument('--checkpoints-only', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.tag):
        parser.error('tag must contain only letters, digits, underscores or hyphens')
    if len(args.seeds) != len(set(args.seeds)):
        parser.error('seeds must be unique')
    args.output_root = args.output_root.resolve()
    args.data_root = args.data_root.resolve()
    if not args.output_root.is_dir() or not args.data_root.is_dir():
        parser.error('data-root and output-root must already exist')
    if not args.checkpoints_only:
        args.resolved_data_path = (
            resolve_imr_data_path(args.data_root, args.data_path)
            if args.dataset == 'imr' else (args.data_path or args.data_root).resolve())
        print('EVALUATION_DATA_PATH=' + str(args.resolved_data_path), flush=True)
    source = source_digest()
    passed = True
    historical_complete = True
    gains = {metric: [] for metric in CORE + RETENTION}
    for seed in args.seeds:
        run = '%s_lora_rank8_baseline_10tasks_seed%d' % (args.dataset, seed)
        lora = args.output_root / run
        tii = args.output_root / ('%s_tii_original_10tasks_seed%d' % (args.dataset, seed))
        manifest = checkpoint_manifest(tii, lora, DATASETS[args.dataset][0], seed)
        if args.checkpoints_only:
            continue
        rows = {}
        for arm in ('baseline', 'identity', 'full'):
            cmd = command(args, seed, arm, tii, lora)
            metadata = dict(source_sha256=source, checkpoints=manifest, command=cmd)
            rows[arm] = run_or_read(args.output_root / log_name(run, arm, args.tag),
                                    cmd, metadata, args.run)
        passed = compare(rows['baseline'], rows['identity'], range(1, 11),
                         CORE + RETENTION, label='IDENTITY_ALL_STAGES') and passed
        for arm in ('baseline', 'full'):
            old = args.output_root / log_name(run, arm)
            if old.exists():
                historical = read_stages(old)
                passed = compare(historical, rows[arm], [10], CORE,
                                 label='HISTORICAL_FINAL_' + arm) and passed
                print('Intermediate deltas (corrected - historical):', arm)
                for stage in range(1, 11):
                    print(' task%d Acc@1=%+.4f Loss=%+.4f' %
                          (stage, rows[arm][stage]['Acc@1'] - historical[stage]['Acc@1'],
                           rows[arm][stage]['Loss'] - historical[stage]['Loss']))
                for metric in RETENTION:
                    print(' task10 %s delta=%+.4f' %
                          (metric, rows[arm][10][metric] - historical[10][metric]))
            else:
                historical_complete = False
                print('HISTORICAL_NOT_VERIFIED missing:', old)
        print('Corrected full minus baseline; seed', seed)
        for metric in gains:
            delta = rows['full'][10][metric] - rows['baseline'][10][metric]
            gains[metric].append(delta)
            print(' %s %+.4f' % (metric, delta))
    if not args.checkpoints_only:
        print('PAIRED_CORRECTED_SUMMARY n=%d (not a claim of improvement)' % len(args.seeds))
        for metric, values in gains.items():
            mean, sd, lo, hi = paired(values)
            print(metric, 'mean=%+.6f SD=%.6f CI95=%s' % (mean, sd, (lo, hi)))
        print('CONSISTENCY_CHECKS=' + ('PASS' if passed else 'FAIL'))
        print('HISTORICAL_COVERAGE=' + ('COMPLETE' if historical_complete else 'INCOMPLETE'))
    return 0 if passed else 2


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        raise SystemExit('VERIFICATION_STOP: ' + str(error))
