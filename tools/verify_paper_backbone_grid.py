#!/usr/bin/env python3
"""Finish the paper's fixed 4-dataset x 5-backbone seed42 grid, evaluation only.

Historical baseline means identity (w=1,beta=0), NOT gated-class-only.
Reuse eight known verified logs when exact provenance matches; evaluate only
missing identity/full arms with --run. No training, weight search or overwrites.
Only load the authors' trusted research checkpoints.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import collect_results as collect
from tools import finish_soict_checks as finish
from tools import verify_paper_results as verify
from tools.verify_paper_weight_grid import PAPER_SOURCE, LEGACY_DRIVER_COMMIT
from protocols import exemplar_free_violations

DATASETS = {
    'imr': ('Split-Imagenet-R', 'imr_lora', 10),
    'cifar100': ('Split-CIFAR100', 'cifar100_lora', 10),
    'ima': ('Split-Imagenet-A', 'ima_lora', 10),
    'fivedatasets': ('5-datasets', 'five_datasets_lora', 5),
}
BACKBONES = {
    'sup': ('vit_base_patch16_224', None, collect.BACKBONES[0]),
    'mocov3': ('vit_base_patch16_224_mocov3', 'mocov3-vit-base-300ep.pth', collect.BACKBONES[1]),
    'ibot1k': ('vit_base_patch16_224_ibot', 'checkpoint_teacher.pth', collect.BACKBONES[2]),
    'ibot21k': ('vit_base_patch16_224_21k_ibot', 'checkpoint.pth', collect.BACKBONES[3]),
    'dino': ('vit_base_patch16_224_dino', 'dino_vitbase16_pretrain.pth', collect.BACKBONES[4]),
}
DRIVER_FILES = ('verify_paper_backbone_grid.py', 'verify_paper_results.py',
                'finish_soict_checks.py', 'collect_results.py',
                'verify_paper_weight_grid.py', 'audit_weight_evidence.py')


def historical_pair(root, dataset, backbone):
    tags, log_tag, _ = BACKBONES[backbone][2]
    tasks = DATASETS[dataset][2]
    paths = {}
    for arm, old_arm in (('identity', 'baseline'), ('full', 'proposed')):
        matches = {Path(p).resolve() for tag in tags
                   if (p := collect.find_log(str(root), dataset, (tag,), log_tag, tasks, old_arm))}
        if len(matches) != 1:
            raise ValueError('Expected one historical log for %s/%s/%s; found %s' %
                             (dataset, backbone, arm, sorted(map(str, matches))))
        paths[arm] = matches.pop()
    runs = {p.name.split(verify.RP_PREFIX)[0] for p in paths.values()}
    if len(runs) != 1:
        raise ValueError('Baseline/full use different checkpoint directory names')
    run = runs.pop()
    return paths, root / run, root / run.replace('_lora_rank8_baseline_', '_tii_original_')


def command(args, dataset, backbone, arm, tii, lora, data_path):
    if arm not in ('identity', 'full'):
        raise ValueError('Grid arm must be identity or full')
    ds, config, tasks = DATASETS[dataset]
    model = BACKBONES[backbone][0]
    base = SimpleNamespace(dataset='imr', batch_size=24, port=29558,
                           resolved_data_path=data_path)
    cmd = verify.command(base, 42, arm, tii, lora)
    cmd[cmd.index('main.py') + 1] = config
    for key, value in (('--dataset', ds), ('--model', model),
                       ('--original_model', model), ('--num_tasks', tasks)):
        finish.set_flag(cmd, key, value)
    return cmd


def parsed_config(cmd):
    config = cmd[cmd.index('main.py') + 1]
    parser = argparse.ArgumentParser()
    importlib.import_module('configs.' + config).get_args_parser(parser)
    return parser.parse_args(cmd[cmd.index('main.py') + 2:])


def feature_tensors(state, model, tasks):
    # The old audit is correct for 10 adapters; 5-Datasets has only 5.
    selected = {k: v for k, v in state.items()
                if k in ('cls_token', 'pos_embed')
                or k.startswith(('patch_embed.', 'blocks.', 'norm.'))}
    for prefix in ('patch_embed.', 'blocks.0.', 'blocks.11.'):
        if not any(k.startswith(prefix) for k in selected):
            raise ValueError('Missing feature tensors: ' + prefix)
    if not {'cls_token', 'pos_embed'} <= selected.keys():
        raise ValueError('Missing embedding tensors')
    has_norm = any(k.startswith('norm.') for k in selected)
    if model == 'vit_base_patch16_224_mocov3':
        if has_norm or not {'fc_norm.weight', 'fc_norm.bias'} <= state.keys():
            raise ValueError('Unexpected MoCo norm structure')
    elif not has_norm:
        raise ValueError('Missing pre_logits norm')
    for key in finish.LORA_KEYS:
        tensor = state[key]
        axis = -1 if key.endswith('_A') else -2
        if (tensor.ndim != 4 or tensor.shape[:2] != (tasks, 5)
                or tensor.shape[axis] != 8):
            raise ValueError('Unexpected adapter pool/rank: ' + key)
        selected[key] = tensor[0]
    return selected


def feature_hash(state, model, tasks):
    import torch
    digest = hashlib.sha256()
    for key, tensor in sorted(feature_tensors(state, model, tasks).items()):
        tensor = tensor.detach().cpu().contiguous()
        if not torch.isfinite(tensor).all():
            raise ValueError('Nonfinite tensor: ' + key)
        digest.update(key.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def checkpoint_args(saved, expected, role):
    if saved is None:
        raise ValueError('Missing checkpoint args')
    if isinstance(saved, dict):
        saved = SimpleNamespace(**saved)
    for key in ('dataset', 'model', 'seed', 'num_tasks'):
        if getattr(saved, key, None) != getattr(expected, key):
            raise ValueError('Checkpoint mismatch: ' + key)
    if exemplar_free_violations(saved):
        raise ValueError('Checkpoint is not exemplar-free')
    # Check explicitly saved settings; old files may predate newer CLI fields.
    keys = ('shuffle', 'train_mask', 'task_inc', 'input_size')
    if role == 'lora':
        keys += ('lora_type', 'lora_rank', 'size')
    for key in keys:
        if hasattr(saved, key) and getattr(saved, key) != getattr(expected, key):
            raise ValueError('Checkpoint/evaluation configuration mismatch: ' + key)


def stamp(path):
    s = path.stat()
    return (s.st_size, s.st_mtime_ns)


def manifest(tii, lora, expected, pretrained):
    import torch
    torch.set_num_threads(2)
    records, stamps, fixed = [], {}, None
    for role, folder in (('tii', tii), ('lora', lora)):
        for task in range(1, expected.num_tasks + 1):
            path = folder / 'checkpoint' / ('task%d_checkpoint.pth' % task)
            before = stamp(path)
            saved = torch.load(path, map_location='cpu', weights_only=False)
            checkpoint_args(saved.get('args'), expected, role)
            if saved.get('real_feature_memory'):
                raise ValueError('Per-example real features in checkpoint: ' + str(path))
            if role == 'lora':
                current = feature_hash(saved['model'], expected.model, expected.num_tasks)
                if fixed is not None and current != fixed:
                    raise ValueError('Fixed first-adapter feature map changed: ' + str(path))
                fixed = current
            del saved
            records.append(dict(path=str(path.resolve()), sha256=finish.file_hash(path)))
            if stamp(path) != before:
                raise ValueError('Checkpoint changed while reading: ' + str(path))
            stamps[str(path.resolve())] = before
    if pretrained:
        path = ROOT / 'checkpoints' / pretrained
        before = stamp(path)
        records.append(dict(path=str(path.resolve()), sha256=finish.file_hash(path)))
        if stamp(path) != before:
            raise ValueError('Pretrained file changed while reading')
        stamps[str(path.resolve())] = before
    print('CHECKPOINT_PROTOCOL_AND_FIXED_FEATURES=PASS',
          expected.dataset, expected.model, expected.num_tasks, fixed, flush=True)
    return records, stamps


def resolve_data(root, dataset):
    if dataset == 'imr':
        return verify.resolve_imr_data_path(root)
    if dataset == 'ima':
        valid = []
        for parent in dict.fromkeys((root, root / 'imagenet-a',
                                     root.parent if root.name == 'imagenet-a' else root)):
            labels = []
            for split in ('train', 'test'):
                folder = parent / 'imagenet-a' / split
                dirs = sorted(p for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []
                if len(dirs) != 200 or not all(
                        any(f.suffix.lower() in verify.IMAGE_EXTENSIONS
                            for f in d.rglob('*') if f.is_file()) for d in dirs):
                    break
                labels.append([d.name for d in dirs])
            if len(labels) == 2 and labels[0] == labels[1]:
                valid.append(parent.resolve())
        valid = sorted(set(valid))
        if len(valid) != 1:
            raise ValueError('Need one existing 200-class ImageNet-A train/test split; '
                             'will NOT move images or create a split: ' + str(valid))
        return valid[0]
    # CPU-only constructors with download=False prevent the evaluation loader
    # from silently downloading missing data. Never call Imagenet_A here: its
    # constructor can move images even when download=False.
    from torchvision import datasets as tv
    if dataset == 'cifar100':
        for train in (True, False):
            tv.CIFAR100(str(root), train=train, download=False)
    else:
        from continual_datasets.continual_datasets import MNIST_RGB, FashionMNIST, NotMNIST, SVHN
        for train in (True, False):
            for cls in (tv.CIFAR10, MNIST_RGB, FashionMNIST, NotMNIST):
                obj = cls(str(root), train=train, download=False)
                if len(obj) == 0:
                    raise ValueError('Empty dataset: ' + cls.__name__)
                del obj
            SVHN(str(root), split='train' if train else 'test', download=False)
    return root


def known_log(args, dataset, backbone, arm, lora, cmd, records):
    meta = dict(source_sha256=PAPER_SOURCE, checkpoints=records, command=cmd)
    if backbone == 'sup' and dataset in ('imr', 'cifar100'):
        path = args.output_root / verify.log_name(lora.name, arm, 'maskfix_verify_v2')
    elif dataset == 'imr' and backbone != 'sup' and arm == 'full':
        commit = ('4989ab866b83bafe04ed7aa2d23ec4f7235e22c4'
                  if backbone == 'mocov3' else LEGACY_DRIVER_COMMIT)
        blob = subprocess.check_output(['git', 'show', commit + ':tools/finish_soict_checks.py'], cwd=ROOT)
        meta['driver_sha256'] = hashlib.sha256(blob).hexdigest()
        path = args.output_root / (lora.name + '__soict_final_v1__full.log')
    else:
        return None
    return (path, meta) if path.exists() else None


def read_verified(path, metadata, tasks):
    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    header = 'VERIFICATION_META=' + json.dumps(metadata, sort_keys=True)
    if lines[:1] != [header] or lines[-1:] != ['VERIFICATION_EXIT_CODE=0']:
        raise ValueError('Log provenance mismatch or unfinished: ' + str(path) +
                         '. Preserve it; do not edit metadata.')
    return verify.read_stages(path, tasks)


def retention_decomposition(path, tasks, rows):
    """Recover old-task/learning-time means from rounded per-task summaries.

    Never infer these from overall stage averages. Missing per-task lines are
    reported as unavailable; they do not invalidate otherwise verified metrics.
    """
    stages, pending = {}, []
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        item = re.match(r'^\s*\* Acc@task\s+[-.\d]+\s+Acc@1\s+([-.\d]+)', line)
        if item:
            pending.append(float(item.group(1)))
        final = verify.FINAL.search(line)
        if final:
            stage = int(final.group(1))
            if len(pending) != stage or stage in stages:
                return dict(status='UNAVAILABLE', reason='Ambiguous/missing per-task summaries')
            stages[stage], pending = pending, []
    if set(stages) != set(range(1, tasks + 1)) or pending:
        return dict(status='UNAVAILABLE', reason='Incomplete per-task sequence')
    for stage, values in stages.items():
        if abs(statistics.mean(values) - rows[stage]['Acc@1']) > .0006:
            return dict(status='UNAVAILABLE', reason='Per-task mean disagrees with stage Acc@1')
    old_final = statistics.mean(stages[tasks][:-1])
    learning = statistics.mean(stages[i][i - 1] for i in range(1, tasks))
    forgetting = statistics.mean(max(stages[t][i - 1] for t in range(i, tasks + 1))
                                - stages[tasks][i - 1] for i in range(1, tasks))
    if (abs(old_final - learning - rows[tasks]['Backward']) > .0011
            or abs(forgetting - rows[tasks]['Forgetting']) > .0011):
        return dict(status='UNAVAILABLE', reason='Retention reconstruction mismatch')
    return dict(status='PASS_ROUNDED_3DP', old_task_final_mean=old_final,
                learning_time_old_task_mean=learning, reconstructed_forgetting=forgetting,
                reconstructed_backward=old_final-learning)


def gpu_ready():
    import torch
    if not torch.cuda.is_available() or torch.cuda.mem_get_info(0)[0] < 16 * 1024**3:
        raise RuntimeError('GPU busy/unavailable; run the same command when free. '
                           'No log created; do not stop another job.')


def run_or_read(path, cmd, meta, tasks, execute):
    if path.exists():
        return read_verified(path, meta, tasks)
    if not execute:
        raise FileNotFoundError('Missing verified log (add --run): ' + str(path))
    gpu_ready()
    print('RUN', path.name, flush=True)
    with path.open('x', encoding='utf-8') as handle:
        handle.write('VERIFICATION_META=' + json.dumps(meta, sort_keys=True) + '\n')
        handle.write('COMMAND=' + json.dumps(cmd) + '\n')
        handle.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, env=dict(os.environ, PYTHONUNBUFFERED='1'),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding='utf-8', errors='replace')
        traceback = False
        for line in proc.stdout:
            handle.write(line)
            handle.flush()
            traceback = traceback or 'Traceback' in line
            if verify.FINAL.search(line) or traceback:
                print(line.rstrip(), flush=True)
        code = proc.wait()
        handle.write('VERIFICATION_EXIT_CODE=%d\n' % code)
    if code:
        raise RuntimeError('Evaluation failed; preserve log: ' + str(path))
    return read_verified(path, meta, tasks)


def assert_unchanged(source, drivers, stamps):
    if verify.source_digest() != source:
        raise ValueError('Evaluator source changed during verification')
    if any(finish.file_hash(ROOT / 'tools' / name) != digest for name, digest in drivers.items()):
        raise ValueError('Verification driver changed during batch')
    if any(stamp(Path(path)) != before for path, before in stamps.items()):
        raise ValueError('Checkpoint/pretrained file changed during batch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--tag', default='paper_backbone_verify_v1')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.tag) or len(set(args.datasets)) != len(args.datasets):
        parser.error('Invalid tag or duplicate dataset')
    args.output_root, args.data_root = args.output_root.resolve(), args.data_root.resolve()
    if not args.output_root.is_dir() or not args.data_root.is_dir():
        parser.error('Existing output/data directories required')
    # Single owner of this runner's logs; lock is released automatically on exit.
    import fcntl
    with (args.output_root / '.paper_backbone_verifier.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute_grid(args)


def execute_grid(args):
    source = verify.source_digest()
    print('EVALUATOR_SOURCE_SHA256=' + source, flush=True)
    if source != PAPER_SOURCE:
        raise ValueError('Evaluator is not the restored paper version')
    drivers = {name: finish.file_hash(ROOT / 'tools' / name) for name in DRIVER_FILES}
    jobs, all_stamps = [], {}
    # Preflight all selected datasets and checkpoints before starting any GPU job.
    for dataset in args.datasets:
        print('PREFLIGHT_DATASET', dataset, flush=True)
        data_path = resolve_data(args.data_root, dataset)
        for backbone in BACKBONES:
            print('PREFLIGHT_CELL', dataset, backbone, flush=True)
            old_paths, lora, tii = historical_pair(args.output_root, dataset, backbone)
            tasks = DATASETS[dataset][2]
            cmds = {arm: command(args, dataset, backbone, arm, tii, lora, data_path)
                    for arm in ('identity', 'full')}
            config = parsed_config(cmds['full'])
            records, stamps = manifest(tii, lora, config, BACKBONES[backbone][1])
            all_stamps.update(stamps)
            for arm in ('identity', 'full'):
                old = verify.read_stages(old_paths[arm], tasks)
                known = known_log(args, dataset, backbone, arm, lora, cmds[arm], records)
                if known:
                    path, meta = known
                else:
                    path = args.output_root / (lora.name + '__' + args.tag + '__' + arm + '.log')
                    meta = dict(source_sha256=source, checkpoints=records, command=cmds[arm],
                                driver_sha256=drivers, purpose='verify fixed paper 20-cell grid')
                rows = read_verified(path, meta, tasks) if path.exists() else None
                jobs.append(dict(dataset=dataset, backbone=backbone, arm=arm, tasks=tasks,
                                 path=path, meta=meta, cmd=cmds[arm], rows=rows, old=old))
    print('PREFLIGHT_COMPLETE arms=%d reused=%d missing=%d' %
          (len(jobs), sum(j['rows'] is not None for j in jobs),
           sum(j['rows'] is None for j in jobs)), flush=True)
    results = []
    for job in jobs:
        assert_unchanged(source, drivers, all_stamps)
        rows = job['rows']
        if rows is None:
            rows = run_or_read(job['path'], job['cmd'], job['meta'], job['tasks'], args.run)
        row = dict(dataset=job['dataset'], backbone=job['backbone'], arm=job['arm'],
                   log=str(job['path']), final=rows[job['tasks']],
                   historical_final=job['old'][job['tasks']])
        row['final_delta'] = {m: row['final'][m] - row['historical_final'][m] for m in finish.METRICS}
        row['stages'] = rows
        row['retention_decomposition'] = retention_decomposition(job['path'], job['tasks'], rows)
        results.append(row)
        print('BACKBONE_GRID_FINAL ' + json.dumps(row, sort_keys=True), flush=True)
    assert_unchanged(source, drivers, all_stamps)
    core_match = all(abs(r['final_delta'][m]) <= 1.000001e-4 for r in results for m in verify.CORE)
    retention_match = all(abs(r['final_delta'][m]) <= 1.000001e-4 for r in results for m in verify.RETENTION)
    for dataset in args.datasets:
        for backbone in BACKBONES:
            pair = {r['arm']: r['final'] for r in results
                    if r['dataset'] == dataset and r['backbone'] == backbone}
            delta = {m: pair['full'][m] - pair['identity'][m] for m in finish.METRICS}
            print('GRID_FULL_MINUS_IDENTITY', dataset, backbone, json.dumps(delta, sort_keys=True), flush=True)
    decompositions = []
    for dataset in args.datasets:
        for backbone in BACKBONES:
            pair = {r['arm']: r['retention_decomposition'] for r in results
                    if r['dataset'] == dataset and r['backbone'] == backbone}
            if all(v['status'] == 'PASS_ROUNDED_3DP' for v in pair.values()):
                decompositions.append({key: pair['full'][key] - pair['identity'][key]
                                       for key in ('old_task_final_mean', 'learning_time_old_task_mean')})
    decomposition_summary = dict(complete_cells=len(decompositions), expected_cells=len(results)//2)
    if len(decompositions) == len(results)//2:
        decomposition_summary.update({
            key: statistics.mean(d[key] for d in decompositions)
            for key in ('old_task_final_mean', 'learning_time_old_task_mean')})
        decomposition_summary['old_task_final_improved_cells'] = sum(
            d['old_task_final_mean'] > 0 for d in decompositions)
    print('RETENTION_DECOMPOSITION', json.dumps(decomposition_summary, sort_keys=True), flush=True)
    report = dict(source_sha256=source, results=results,
                  retention_decomposition=decomposition_summary,
                  historical_core_match=core_match, historical_retention_match=retention_match)
    report_path = args.output_root / (args.tag + '__' + '-'.join(args.datasets) + '__summary.json')
    if report_path.exists():
        if json.loads(report_path.read_text(encoding='utf-8')) != json.loads(json.dumps(report)):
            raise ValueError('Existing summary differs; preserve it: ' + str(report_path))
    else:
        with report_path.open('x', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write('\n')
    print('SUMMARY=' + str(report_path), flush=True)
    print('PAPER_BACKBONE_GRID_COMPLETE=%d/%d' % (len(results)//2, len(args.datasets)*5), flush=True)
    print('HISTORICAL_FINAL_CORE=' + ('PASS' if core_match else 'CHANGED'), flush=True)
    print('HISTORICAL_RETENTION=' + ('PASS' if retention_match else 'CHANGED'), flush=True)
    print('NOTE: completion is not equality; use verified metrics, review CHANGED values.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.CalledProcessError) as error:
        raise SystemExit('BACKBONE_GRID_STOP: ' + str(error))
