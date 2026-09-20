#!/usr/bin/env python3
"""Fixed full w=.6 versus .7, three datasets/four seeds; evaluation only.

Reuse four verified IMR w=.6 runs and all twelve baseline/w=.7 pairs.
Only eight CIFAR/CUB w=.6 evaluations can be new. Never promote a winner,
change the paper, train, overwrite logs, or modify existing evidence drivers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import verify_paper_backbone_grid as grid

verify, finish = grid.verify, grid.finish
SEEDS = (42, 43, 44, 45)
DRIVERS = grid.DRIVER_FILES + ('check_full_w06.py',)


def command(base, seed, tii, lora):
    cmd = verify.command(base, seed, 'full', tii, lora)
    finish.set_flag(cmd, '--rp_route_fusion_weight', '0.6')
    return cmd


def reference_tag(dataset, seed):
    return ('paperfix_verify_v3' if dataset == 'cub200' and seed != 42
            else 'maskfix_verify_v2')


def old_w06_hash():
    blob = subprocess.check_output(
        ['git', 'show', grid.LEGACY_DRIVER_COMMIT + ':tools/finish_soict_checks.py'],
        cwd=ROOT)
    return hashlib.sha256(blob).hexdigest()


def summarise(results):
    report = {}
    for dataset in verify.DATASETS:
        rows = results[dataset]
        if set(rows) != set(SEEDS):
            raise ValueError('Incomplete paired seeds: ' + dataset)
        arms = {}
        for arm in ('baseline', 'w07', 'w06'):
            arms[arm] = {
                m: dict(mean=statistics.mean(rows[s][arm][m] for s in SEEDS),
                        sd=statistics.stdev(rows[s][arm][m] for s in SEEDS))
                for m in finish.METRICS}
        contrasts = {}
        for right in ('w07', 'baseline'):
            contrasts['w06_minus_' + right] = {
                m: dict(zip(('mean', 'sd', 'ci95_low', 'ci95_high'),
                            finish.paired([rows[s]['w06'][m] - rows[s][right][m]
                                           for s in SEEDS])))
                for m in finish.METRICS}
        report[dataset] = dict(n=4, arms=arms, contrasts=contrasts, per_seed=rows)
    return report


def save_summary(path, report):
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != json.loads(json.dumps(report)):
            raise ValueError('Existing summary differs; preserve it: ' + str(path))
    else:
        with path.open('x', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write('\n')


def execute(args):
    source = verify.source_digest()
    if source != grid.PAPER_SOURCE:
        raise ValueError('Evaluator is not the restored paper version')
    print('EVALUATOR_SOURCE_SHA256=' + source, flush=True)
    drivers = {n: finish.file_hash(ROOT / 'tools' / n) for n in DRIVERS}
    legacy = old_w06_hash()
    jobs, stamps, results = [], {}, {d: {} for d in verify.DATASETS}
    for dataset in verify.DATASETS:
        data = (verify.resolve_imr_data_path(args.data_root) if dataset == 'imr'
                else args.data_root)
        for seed in SEEDS:
            print('PREFLIGHT', dataset, seed, flush=True)
            base = SimpleNamespace(dataset=dataset, resolved_data_path=data,
                                   batch_size=24, port=29558)
            run = '%s_lora_rank8_baseline_10tasks_seed%d' % (dataset, seed)
            lora = args.output_root / run
            tii = args.output_root / ('%s_tii_original_10tasks_seed%d' % (dataset, seed))
            cmd = command(base, seed, tii, lora)
            records, current = grid.manifest(tii, lora, grid.parsed_config(cmd), None)
            stamps.update(current)
            final = {}
            for arm, label in (('baseline', 'baseline'), ('full', 'w07')):
                ref_cmd = verify.command(base, seed, arm, tii, lora)
                path = args.output_root / verify.log_name(
                    run, arm, reference_tag(dataset, seed))
                meta = dict(source_sha256=source, checkpoints=records, command=ref_cmd)
                final[label] = grid.read_verified(path, meta, 10)[10]
            meta = dict(source_sha256=source, checkpoints=records, command=cmd)
            if dataset == 'imr':
                path = args.output_root / (run + '__soict_final_v1__w06.log')
                meta['driver_sha256'] = legacy
                # Missing or mismatched original evidence must stop, not rerun silently.
                rows = grid.read_verified(path, meta, 10)
            else:
                path = args.output_root / (run + '__' + args.tag + '__w06.log')
                meta.update(driver_sha256=drivers,
                            purpose='fixed w06 comparison; exploratory, no automatic selection')
                rows = grid.read_verified(path, meta, 10) if path.exists() else None
            results[dataset][seed] = final
            jobs.append(dict(dataset=dataset, seed=seed, cmd=cmd, path=path,
                             meta=meta, rows=rows))
    missing = sum(j['rows'] is None for j in jobs)
    print('W06_PREFLIGHT total=12 reused=%d missing=%d' % (12-missing, missing), flush=True)
    grid.assert_unchanged(source, drivers, stamps)
    if missing and not args.run:
        print('NOT_RUN: add --run to evaluate missing CIFAR/CUB runs only.', flush=True)
        return
    for job in jobs:
        grid.assert_unchanged(source, drivers, stamps)
        rows = job['rows']
        if rows is None:
            # Only small logs are created, no checkpoints or feature caches.
            if shutil.disk_usage(args.output_root).free < 1024**3:
                raise RuntimeError('Disk free <1 GiB; free space before retrying. Nothing deleted.')
            start = time.monotonic()
            rows = grid.run_or_read(job['path'], job['cmd'], job['meta'], 10, args.run)
            print('NEW_EVAL_SECONDS %.3f %s seed%d' %
                  (time.monotonic()-start, job['dataset'], job['seed']), flush=True)
        results[job['dataset']][job['seed']]['w06'] = rows[10]
        print('W06_FINAL', job['dataset'], job['seed'],
              json.dumps(rows[10], sort_keys=True), flush=True)
    grid.assert_unchanged(source, drivers, stamps)
    report = dict(source_sha256=source, datasets=summarise(results),
                  scope='Sup-21K three datasets, four seeds; not full manuscript migration',
                  selection='Exploratory test-informed comparison. No automatic paper change.',
                  remaining='If changing default: grid, routing-dependent ablations/diagnostics, '
                            'AugReg comparison, all derived text/charts must be reconciled.')
    path = args.output_root / (args.tag + '__summary.json')
    save_summary(path, report)
    for dataset, item in report['datasets'].items():
        print('W06_COMPARISON', dataset, json.dumps(item, sort_keys=True), flush=True)
    print('SUMMARY=' + str(path), flush=True)
    print('FULL_W06_MAIN_COMPLETE=12/12; PAPER_UNCHANGED', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--tag', default='full_w06_main_v1')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.tag):
        parser.error('Invalid tag')
    args.output_root, args.data_root = args.output_root.resolve(), args.data_root.resolve()
    if not args.output_root.is_dir() or not args.data_root.is_dir():
        parser.error('Existing output/data directories required')
    import fcntl
    with (args.output_root / '.paper_backbone_verifier.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.CalledProcessError) as error:
        raise SystemExit('FULL_W06_STOP: ' + str(error))
