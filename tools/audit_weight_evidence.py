#!/usr/bin/env python3
"""Audit the fusion-weight evidence in the paper against the evaluation logs.

The paper's Sect. 4.6 makes four claims that can only be checked against the
runs themselves: the nine cells of the w-by-beta grid, the four-seed comparison
between w = 0.6 and w = 0.7, the provenance of the two self-run rows of the
placement table, and whether a class-plus-gate control without routing was ever
run. This reads every log under the output root, reconstructs each run's
configuration from its filename, and reports what exists.

The filename is the record: eval_rp_head_any_4090.sh encodes source, dimension,
lambda, the fusion flags, w, beta, the gate and the ramp into the log path, with
dots written as 'p'. Nothing here re-runs anything or writes to the run
directories.

Usage:  python3 audit_weight_evidence.py [OUTPUT_ROOT]
        default OUTPUT_ROOT is ~/hrm-pet-output
"""
import glob
import math
import os
import re
import sys
from collections import defaultdict

FINAL = re.compile(r'\[Average accuracy till task(\d+)\]\s*(.*)')
NUM = re.compile(r'([A-Za-z@0-9/]+):\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')
# Exact small-sample quantiles; larger studies use scipy (a repo dependency).
T975 = {1: 12.706204736432095, 2: 4.302652729696142,
        3: 3.182446305284263, 4: 2.7764451051977987}


def undot(s):
    return s.replace('p', '.')


def read_final(path, expected_tasks=None):
    """Accept only a completed final stage, never a partial run's last row."""
    if expected_tasks is None:
        match = re.search(r'_(\d+)tasks_seed\d+', os.path.basename(path))
        if not match:
            return None, {}
        expected_tasks = int(match.group(1))
    last = None
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as handle:
            for line in handle:
                found = FINAL.search(line)
                if found:
                    last = found
    except OSError:
        return None, {}
    if last is None or int(last.group(1)) != expected_tasks:
        return None, {}
    metrics = {k: float(v) for k, v in NUM.findall(last.group(2))}
    if not all(k in metrics and math.isfinite(metrics[k])
               for k in ('Acc@1', 'Acc@task', 'Acc@5', 'Loss')):
        return None, {}
    if expected_tasks > 1 and not all(k in metrics and math.isfinite(metrics[k])
                                      for k in ('Forgetting', 'Backward')):
        return None, {}
    return expected_tasks, metrics


def configuration_key(cfg, varying=('w',), across_seeds=True):
    """Keep EVERY filename field except the explicitly varied coordinates.

    This preserves dimension, calibration, pinning, backbone, ramps, and the
    evaluator-version suffix, including fields this reader does not interpret.
    """
    key = cfg['file']
    if across_seeds:
        key = re.sub(r'_seed\d+(?=_eval_)', '_seed*', key)
    if 'w' in varying:
        key = re.sub(r'(?<=d[01])w[0-9p]+(?=lsw)', 'w*', key)
    if 'beta' in varying:
        key = re.sub(r'cw[0-9p]+(?=sh)', 'cw*', key)
    return key


def parse_name(name):
    """Reconstruct the run configuration from the log filename."""
    cfg = {'file': name, 'run': name.split('_eval_')[0]}
    seed = re.search(r'seed(\d+)', name)
    cfg['seed'] = seed.group(1) if seed else '?'
    if '_eval_conventional' in name:
        cfg['kind'] = 'baseline'
        return cfg
    if '_eval_rp_' not in name:
        cfg['kind'] = 'other'
        return cfg
    cfg['kind'] = 'rp'
    fuse = re.search(r'_f(\d)d(\d)w([0-9p]+)lsw', name)
    if fuse:
        cfg['route_fusion'] = fuse.group(1) == '1'
        cfg['route_drm'] = fuse.group(2) == '1'
        cfg['w'] = undot(fuse.group(3))
    beta = re.search(r'cw([0-9p]+)sh', name)
    cfg['beta'] = undot(beta.group(1)) if beta else '?'
    cfg['gate'] = 'margin_both' if 'gmargin_both' in name else (
        'margin' if 'gmargin' in name else (
            'entropy' if 'gentropy' in name else 'none'))
    cfg['ramp'] = 'yes' if re.search(r'm\d+(?:g[a-z_]+)?r\d+p\d+', name) else 'no'
    src = re.search(r'_eval_rp_([a-z]+)_d(\d+)_([a-z]+)_l([0-9p]+)', name)
    if src:
        cfg['source'], cfg['dim'] = src.group(1), src.group(2)
        cfg['act'], cfg['lambda'] = src.group(3), undot(src.group(4))
    if not fuse or not beta or not src:
        cfg['kind'] = 'other'
    else:
        try:
            float(cfg['w'])
            float(cfg['beta'])
        except ValueError:
            cfg['kind'] = 'other'
    return cfg


def stage(cfg):
    """Name the experimental configuration in the paper's own terms."""
    if cfg['kind'] != 'rp':
        return cfg['kind']
    w = cfg.get('w', '?')
    beta = cfg.get('beta', '?')
    if not cfg.get('route_drm'):
        return 'RP-route-fusion' if cfg.get('route_fusion') else 'RP-only'
    if float(beta) == 0.0 and float(w) == 1.0:
        return 'baseline-identity'
    if float(beta) == 0.0:
        return 'routing-only'
    if float(w) == 1.0:
        return 'class-only' + ('+gate' if cfg['gate'] != 'none' else '')
    return 'full' + ('+gate' if cfg['gate'] != 'none' else '')


def paired(pairs):
    """Mean, sample SD and 95% t interval of a list of paired differences."""
    n = len(pairs)
    if not n or not all(math.isfinite(d) for d in pairs):
        raise ValueError('paired differences must be nonempty and finite')
    mean = sum(pairs) / n
    if n < 2:
        return mean, 0.0, None, None
    var = sum((d - mean) ** 2 for d in pairs) / (n - 1)
    sd = var ** 0.5
    critical = T975.get(n - 1)
    if critical is None:
        from scipy.stats import t
        critical = float(t.ppf(0.975, df=n - 1))
    half = critical * sd / (n ** 0.5)
    return mean, sd, mean - half, mean + half


def main(root):
    logs = sorted(glob.glob(os.path.join(root, '*.log')))
    print('thu muc :', root)
    print('so log  :', len(logs))
    if not logs:
        print('KHONG THAY LOG. Truyen duong dan that lam tham so.')
        return 1

    rows = []
    for path in logs:
        cfg = parse_name(os.path.basename(path))
        if cfg['kind'] == 'other':
            continue
        tasks, met = read_final(path)
        if not met:
            print('SKIP incomplete/invalid:', os.path.basename(path))
            continue
        cfg['tasks'] = tasks
        cfg['met'] = met
        cfg['stage'] = stage(cfg)
        rows.append(cfg)
    print('log co dong ket qua cuoi:', len(rows))
    print()

    # ---- 1. inventory ---------------------------------------------------
    print('=' * 78)
    print('1. KIEM KE: moi cau hinh da chay')
    print('=' * 78)
    print('%-26s %-5s %-16s %-5s %-5s %-7s %8s %8s'
          % ('run', 'seed', 'stage', 'w', 'beta', 'gate', 'Acc@task', 'Acc@1'))
    for r in sorted(rows, key=lambda r: (r['run'], r['stage'],
                                         r.get('w', ''), r.get('beta', ''))):
        print('%-26s %-5s %-16s %-5s %-5s %-7s %8.4f %8.4f'
              % (r['run'][:26], r['seed'], r['stage'], r.get('w', '-'),
                 r.get('beta', '-'), r.get('gate', '-'),
                 r['met'].get('Acc@task', float('nan')),
                 r['met'].get('Acc@1', float('nan'))))
    print()

    # ---- 2. the w-by-beta grid -----------------------------------------
    print('=' * 78)
    print('2. LUOI w x beta (bang 3 trong bai): can du 9 o, mot seed, gate margin')
    print('=' * 78)
    grid = defaultdict(dict)
    for r in rows:
        if r['stage'].startswith('full') and r.get('gate') == 'margin':
            key = configuration_key(r, varying=('w', 'beta'), across_seeds=False)
            cell = (r.get('w'), r.get('beta'))
            if cell in grid[(key, r['seed'])]:
                raise ValueError('Ambiguous duplicate grid cell: ' + r['file'])
            grid[(key, r['seed'])][cell] = r['met'].get('Acc@1')
    if not grid:
        print('KHONG co o nao khop (full + gate margin).')
    for (run, seed), cells in sorted(grid.items()):
        print('%s seed %s: %d o' % (run, seed, len(cells)))
        ws = sorted({w for w, _ in cells})
        bs = sorted({b for _, b in cells})
        print('       ' + ''.join('%10s' % ('beta=' + b) for b in bs))
        for w in ws:
            line = '  w=%-4s' % w
            for b in bs:
                v = cells.get((w, b))
                line += '%10s' % ('-' if v is None else '%.2f' % v)
            print(line)
        expected = {(w, b) for w in ('0.6', '0.7', '0.8')
                    for b in ('0.3', '0.5', '0.7')}
        missing = expected - cells.keys()
        if missing:
            print('  THIEU dung cac o trong bai:', sorted(missing))
    print()

    # ---- 3. w = 0.6 against w = 0.7, paired over seeds -------------------
    print('=' * 78)
    print('3. w=0.6 SO w=0.7 GHEP CAP THEO SEED (cung beta, cung gate, cung mode)')
    print('=' * 78)
    buckets = defaultdict(dict)
    for r in rows:
        if r['kind'] != 'rp' or r.get('w') not in ('0.6', '0.7'):
            continue
        dataset = configuration_key(r, varying=('w',))
        key = (dataset, r['stage'], r.get('beta'), r.get('gate'))
        if r['w'] in buckets[key].get(r['seed'], {}):
            raise ValueError('Ambiguous duplicate paired run: ' + r['file'])
        buckets[key].setdefault(r['seed'], {})[r['w']] = r['met']
    if not buckets:
        print('KHONG co cap w nao de so.')
    for key, by_seed in sorted(buckets.items()):
        dataset, st, beta, gate = key
        seeds = sorted(s for s, d in by_seed.items() if '0.6' in d and '0.7' in d)
        print('%s | %s | beta=%s gate=%s : %d seed ghep duoc %s'
              % (dataset, st, beta, gate, len(seeds), seeds))
        if len(seeds) < 2:
            continue
        for name in ('Acc@1', 'Acc@task', 'Forgetting', 'Backward'):
            diffs = []
            for s in seeds:
                a = by_seed[s]['0.6'].get(name)
                b = by_seed[s]['0.7'].get(name)
                if a is not None and b is not None:
                    diffs.append(a - b)
            if len(diffs) != len(seeds):
                continue
            mean, sd, lo, hi = paired(diffs)
            span = '' if lo is None else '  CI95 [%+.3f, %+.3f]' % (lo, hi)
            print('   %-11s 0.6 - 0.7 = %+.4f +- %.4f%s' % (name, mean, sd, span))
    print()

    # ---- 4. controls the paper says it does not have --------------------
    print('=' * 78)
    print('4. KIEM KE DOI CHUNG (KHONG TU SUY RA CUNG PROTOCOL)')
    print('=' * 78)
    for want, label in (('class-only+gate', 'class+gate, khong routing'),
                        ('RP-only', 'RP-only (khong dong nghia RanPAC)')):
        got = defaultdict(set)
        for r in rows:
            if r['stage'] == want:
                got[configuration_key(r, varying=())].add(r['seed'])
        if not got:
            print('%-34s: KHONG CO log nao' % label)
        for dataset, seeds in sorted(got.items()):
            print('%-34s: %s -> seed %s'
                  % (label, dataset, sorted(seeds)))
    print()

    # ---- 5. what the placement table quotes -----------------------------
    print('=' * 78)
    print('5. HAI HANG TU CHAY CUA BANG 2 (ImageNet-A va 5-Datasets)')
    print('=' * 78)
    for r in sorted(rows, key=lambda r: r['run']):
        low = r['run'].lower()
        if low.startswith('ima_') or 'imagenet-a' in low or 'imr_a' in low or '5-dataset' in low \
                or 'five' in low or 'inat' in low:
            print('%-30s seed %-4s %-16s tasks=%s Acc@1=%.4f'
                  % (r['run'][:30], r['seed'], r['stage'], r['tasks'],
                     r['met'].get('Acc@1', float('nan'))))
            print('     %s' % r['file'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1
                          else os.path.expanduser('~/hrm-pet-output')))
