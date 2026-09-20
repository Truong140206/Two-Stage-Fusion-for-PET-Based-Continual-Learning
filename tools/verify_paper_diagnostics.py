#!/usr/bin/env python3
"""Close the paper's seed-42 oracle diagnostics, not a parameter search.

Two instrumented evaluations per dataset: RP-only for bare-TII/RP routing
overlap, routing-only for classifier overlap. The counters exist already;
no evaluator, checkpoint, dataset, or completed verification log is modified.
Without --run this is a preflight/reuse check. Only trusted local checkpoints.
"""
import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import verify_paper_ablation as ab

grid, verify, finish = ab.grid, ab.verify, ab.finish
DRIVERS = ab.DRIVERS + ('verify_paper_diagnostics.py',)
FIELDS = {
    'rp_only': ('RouteTII', 'RouteRP', 'RouteUnion', 'RouteBoth',
                'RouteAgree', 'RouteRPOnly'),
    'route_only': ('ClsRouted', 'ClsRP', 'ClsUnion', 'ClsRPOnly'),
}
PAPER_CLAIMS = {
    'imr': {'RouteTII': (63.80, .0051), 'RouteRP': (77.12, .0051),
            'RouteUnion': (82.99, .0051), 'DRM': (77.79, .0051),
            'task_balanced_error_recovery_pct': (53.0, .051),
            'ClsRouted': (74.31, .0051), 'ClsRP': (70.08, .0051),
            'ClsUnion': (78.89, .0051), 'ClsRPOnly': (4.58, .0051)},
    'cifar100': {'task_balanced_error_recovery_pct': (62.0, .051),
                 'ClsRPOnly': (2.43, .0051)},
    'cub200': {'task_balanced_error_recovery_pct': (46.9, .051),
               'ClsRPOnly': (3.87, .0051)},
}


def compare_claims(dataset, pair, baseline):
    values = dict(pair['rp_only']['diagnostics'], **pair['route_only']['diagnostics'],
                  DRM=baseline['Acc@task'])
    return {key: dict(paper=expected, measured=values[key],
                     status='MATCH_ROUNDING' if values[key] is not None and
                     abs(values[key]-expected) <= tolerance else 'UPDATE_PAPER')
            for key, (expected, tolerance) in PAPER_CLAIMS[dataset].items()}


def command(args, arm, tii, lora, diagnostic=True):
    if arm == 'rp_only':
        cmd = finish.variant_command(args, 42, tii, lora,
                                     'vit_base_patch16_224', 'rp_only')
        flag = '--rp_route_audit'
    elif arm == 'route_only':
        cmd = verify.command(args, 42, 'full', tii, lora)
        finish.set_flag(cmd, '--rp_class_fusion_weight', '0.0')
        finish.set_flag(cmd, '--rp_class_fusion_gate', 'none')
        flag = '--classifier_union_audit'
    else:
        raise ValueError('Unknown diagnostic arm: ' + arm)
    if diagnostic:
        cmd.append(flag)
    return cmd


def diagnostics(rows, arm):
    """Validate all stages and overlap identities; never infer missing counts."""
    fields = FIELDS[arm]
    for stage, row in rows.items():
        for key in fields:
            value = row.get(key)
            if value is None or not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError('Missing/invalid diagnostic: task%d %s' % (stage, key))
        if arm == 'rp_only':
            a, b, union, both = (row[k] for k in fields[:4])
            only = row['RouteRPOnly']
            identities = (union-a-only, a+b-both-union, b-both-only,
                          b-row['Acc@task'])
        else:
            a, b, union, only = (row[k] for k in fields)
            identities = (union-a-only, a-row['Acc@1'])
        if (union < max(a, b)-.0003 or union > min(100, a+b)+.0003
                or any(abs(x) > .00031 for x in identities)):
            raise ValueError('Inconsistent overlap at task%d' % stage)
    result = {key: rows[10][key] for key in fields}
    if arm == 'rp_only':
        error = 100 - result['RouteTII']
        result['task_balanced_error_recovery_pct'] = (
            100 * result['RouteRPOnly'] / error if error > 0 else None)
        result['task_balanced_tii_only_pp'] = (
            result['RouteUnion'] - result['RouteRP'])
    else:
        result['task_balanced_routed_only_pp'] = (
            result['ClsUnion'] - result['ClsRP'])
    return result


def execute(args):
    source = verify.source_digest()
    print('EVALUATOR_SOURCE_SHA256=' + source, flush=True)
    if source != grid.PAPER_SOURCE:
        raise ValueError('Evaluator is not the restored paper version')
    drivers = {n: finish.file_hash(ROOT / 'tools' / n) for n in DRIVERS}
    jobs, baseline, stamps = [], {}, {}
    for dataset in verify.DATASETS:
        data = (verify.resolve_imr_data_path(args.data_root) if dataset == 'imr'
                else args.data_root)
        # Dataset constructors below are deliberately not called here: the
        # already verified main logs pin the selected dataset/checkpoints.
        base = SimpleNamespace(dataset=dataset, batch_size=24, port=29558,
                               resolved_data_path=data)
        run = dataset + '_lora_rank8_baseline_10tasks_seed42'
        lora = args.output_root / run
        tii = args.output_root / (dataset + '_tii_original_10tasks_seed42')
        cmds = {arm: command(base, arm, tii, lora) for arm in FIELDS}
        for cmd in cmds.values():
            grid.parsed_config(cmd)
        config = grid.parsed_config(cmds['route_only'])
        records, current = grid.manifest(tii, lora, config, None)
        stamps.update(current)
        base_cmd = verify.command(base, 42, 'baseline', tii, lora)
        base_path = args.output_root / verify.log_name(run, 'baseline', 'maskfix_verify_v2')
        base_meta = dict(source_sha256=source, checkpoints=records, command=base_cmd)
        baseline[dataset] = grid.read_verified(base_path, base_meta, 10)[10]
        for arm in FIELDS:
            # RP-only has an existing, fully verified, non-instrumented control.
            reference = None
            if arm == 'rp_only':
                path = args.output_root / (run + '__soict_final_v1__rp_only.log')
                meta = dict(source_sha256=source, checkpoints=records,
                            command=command(base, arm, tii, lora, False),
                            driver_sha256=ab.original_driver_hash())
                reference = grid.read_verified(path, meta, 10)
            elif dataset == 'imr':
                path = args.output_root / (run + '__paper_ablation_verify_v1__route_only.log')
                meta = dict(source_sha256=source, checkpoints=records,
                            command=command(base, arm, tii, lora, False),
                            driver_sha256={n: drivers[n] for n in ab.DRIVERS},
                            purpose='verify fixed ungated paper ablation')
                reference = grid.read_verified(path, meta, 10)
            path = args.output_root / (run + '__' + args.tag + '__' + arm + '.log')
            meta = dict(source_sha256=source, checkpoints=records, command=cmds[arm],
                        driver_sha256=drivers, purpose='verify fixed paper oracle diagnostics')
            rows = grid.read_verified(path, meta, 10) if path.exists() else None
            if rows is not None:
                diagnostics(rows, arm)
            jobs.append(dict(dataset=dataset, arm=arm, path=path, meta=meta,
                             cmd=cmds[arm], rows=rows, reference=reference))
    missing = sum(j['rows'] is None for j in jobs)
    print('DIAGNOSTIC_PREFLIGHT total=6 reused=%d missing=%d' % (6-missing, missing), flush=True)
    if missing and not args.run:
        print('NOT_RUN: add --run to evaluate only missing diagnostic logs.', flush=True)
        return
    results = {}
    for job in jobs:
        grid.assert_unchanged(source, drivers, stamps)
        rows = job['rows'] if job['rows'] is not None else grid.run_or_read(
            job['path'], job['cmd'], job['meta'], 10, args.run)
        diag = diagnostics(rows, job['arm'])
        if job['reference'] is not None:
            if not verify.compare(job['reference'], rows, range(1, 11), finish.METRICS,
                                  label='COUNTERS_DO_NOT_CHANGE_' + job['dataset'] + '_' + job['arm']):
                raise ValueError('Instrumented result differs from verified control')
        results.setdefault(job['dataset'], {})[job['arm']] = dict(
            log=str(job['path']), diagnostics=diag, final=rows[10])
        print('DIAGNOSTIC_FINAL ' + json.dumps(dict(dataset=job['dataset'],
              arm=job['arm'], diagnostics=diag, final=rows[10]), sort_keys=True), flush=True)
    comparisons = {}
    for dataset, pair in results.items():
        if abs(pair['rp_only']['final']['Acc@1'] -
               pair['route_only']['diagnostics']['ClsRP']) > .00031:
            raise ValueError('RP predictions disagree across branches: ' + dataset)
        print('BASELINE_DRM_PROPOSAL', dataset, baseline[dataset]['Acc@task'], flush=True)
        comparisons[dataset] = compare_claims(dataset, pair, baseline[dataset])
        print('PAPER_DIAGNOSTIC_CLAIMS ' + json.dumps(
            dict(dataset=dataset, claims=comparisons[dataset]), sort_keys=True), flush=True)
    grid.assert_unchanged(source, drivers, stamps)
    report = dict(source_sha256=source, results=results, baseline=baseline,
                  paper_claims=comparisons,
                  weighting='Equal weight per task; NOT pooled image counts.',
                  scope='Seed42 oracle overlap; NOT actual full-fusion repair/harm or timing.')
    path = args.output_root / (args.tag + '__summary.json')
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != report:
            raise ValueError('Existing summary differs; preserve it: ' + str(path))
    else:
        with path.open('x', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write('\n')
    print('SUMMARY=' + str(path), flush=True)
    print('PAPER_DIAGNOSTICS_COMPLETE=6/6', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--tag', default='paper_diagnostics_verify_v1')
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
        raise SystemExit('PAPER_DIAGNOSTICS_STOP: ' + str(error))
