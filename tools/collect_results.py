#!/usr/bin/env python3
"""Collect the baseline-vs-proposed grid straight out of the evaluation logs.

Twenty-one cells were measured across four benchmarks and six backbones, over
several days and several restarts, with some logs written by runs that later
crashed. Transcribing those numbers by hand into a report is exactly the sort
of step that quietly introduces an error nobody can trace afterwards, so the
table is built from the files instead.

Logs are addressed by their EXACT name, rebuilt from the template
eval_rp_head_any_4090.sh uses. An earlier version globbed on the fusion weights
alone and took whichever file matched first, which is enough to pick up a stray
log from an old sweep that shares those two weights but differs elsewhere:
tools/ablation_table.py did exactly that and reported 75.08 where the measured
value is 75.51. Every field of the name is pinned here, and a cell whose file is
absent is reported missing rather than filled from a near neighbour.

  baseline   w = 1.0, beta = 0     routing goes to TII alone
  proposed   w = 0.7, beta = 0.5   the full method, margin gate on

Usage:
    python tools/collect_results.py
    python tools/collect_results.py --csv results.csv --worse
"""
import argparse
import csv
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS = [('imr', 'ImageNet-R', 10), ('cifar100', 'CIFAR-100', 10),
            ('ima', 'ImageNet-A', 10), ('fivedatasets', '5-Datasets', 5)]

# (directory tags to try, tag inside the log name, label). Two tags are needed
# because different things produce them: the directory tag is ours, the log tag
# is derived by the eval script from the timm model name. ImageNet-R was trained
# before the directory convention settled, so MoCo v3 sits under `mocov3` there
# and `moco1k` on the other three benchmarks.
BACKBONES = [
    (('',), '', 'Sup-21K'),
    (('moco1k', 'mocov3'), 'bmocov3', 'MoCo v3-1K'),
    (('ibot1k',), 'bibot', 'iBOT-1K'),
    (('ibot21k',), 'b21kibot', 'iBOT-21K'),
    (('dino',), 'bdino', 'DINO-1K'),
    (('mae',), 'bmae', 'MAE-1K'),
]

METRICS = ['Acc@task', 'Acc@1', 'Acc@5', 'Loss', 'Forgetting', 'Backward']
# +1 where a larger number is better, -1 where a smaller one is. Backward is
# normally negative and closer to zero is better, so it counts as larger.
DIRECTION = {'Acc@task': 1, 'Acc@1': 1, 'Acc@5': 1,
             'Loss': -1, 'Forgetting': -1, 'Backward': 1}

# (fusion weight tag, class weight tag, gate tag)
ARMS = {'baseline': ('w1p0', 'cw0p0', 'gmargin'),
        'proposed': ('w0p7', 'cw0p5', 'gmargin')}

# Everything else in the log name, fixed across this grid.
FIXED = ('_eval_rp_lora_d10000_relu_l10000_nnone_t0_b0p0_p1_inone_c0_ra0ls0'
         '_f1d1')


def output_root():
    return os.path.join(os.path.dirname(REPO_ROOT), 'hrm-pet-output')


def find_log(root, dataset, dir_tags, log_tag, num_tasks, arm):
    weight, class_weight, gate = ARMS[arm]
    for dir_tag in dir_tags:
        suffix = '_%s' % dir_tag if dir_tag else ''
        base = '%s%s_lora_rank8_baseline_%dtasks_seed42' % (
            dataset, suffix, num_tasks)
        name = '%s%s%slsw0p0c0ca0%ssh1p0m1%s%s.log' % (
            base, FIXED, weight, class_weight, gate, log_tag)
        path = os.path.join(root, name)
        if os.path.isfile(path):
            return path
    return None


def final_row(path, num_tasks):
    marker = 'Average accuracy till task%d]' % num_tasks
    row = None
    with open(path, encoding='utf-8', errors='replace') as handle:
        for line in handle:
            if marker in line:
                row = line
    if row is None:
        return None
    out = {}
    for name in METRICS:
        match = re.search(r'%s:\s*(-?[0-9]+(?:\.[0-9]+)?)' % re.escape(name),
                          row)
        if match:
            out[name] = float(match.group(1))
    return out or None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=None)
    parser.add_argument('--csv', default=None)
    parser.add_argument('--worse', action='store_true',
                        help='liet ke moi o va chi so ma de xuat kem hon moc')
    args = parser.parse_args()
    root = args.root or output_root()

    rows = []
    print('%-12s %-11s %9s %9s %8s   %9s %9s %8s   %7s %7s'
          % ('bo du lieu', 'backbone', 'moc@task', 'dx@task', 'delta',
             'moc@1', 'dx@1', 'delta', 'quen-m', 'quen-d'))
    print('-' * 104)
    for dataset, dataset_label, num_tasks in DATASETS:
        for dir_tags, log_tag, backbone_label in BACKBONES:
            found = {}
            for arm in ARMS:
                path = find_log(root, dataset, dir_tags, log_tag, num_tasks,
                                arm)
                found[arm] = final_row(path, num_tasks) if path else None
            if not any(found.values()):
                continue
            base, prop = found['baseline'], found['proposed']
            if base is None or prop is None:
                which = 'moc' if base is None else 'de xuat'
                print('%-12s %-11s  THIEU %s'
                      % (dataset_label, backbone_label, which))
                continue
            print('%-12s %-11s %9.2f %9.2f %+8.2f   %9.2f %9.2f %+8.2f   '
                  '%7.2f %7.2f'
                  % (dataset_label, backbone_label,
                     base['Acc@task'], prop['Acc@task'],
                     prop['Acc@task'] - base['Acc@task'],
                     base['Acc@1'], prop['Acc@1'],
                     prop['Acc@1'] - base['Acc@1'],
                     base['Forgetting'], prop['Forgetting']))
            row = {'dataset': dataset_label, 'backbone': backbone_label}
            for name in METRICS:
                row['base_' + name] = base.get(name)
                row['prop_' + name] = prop.get(name)
            rows.append(row)
        print()

    print('tong: %d o co du ca hai cau hinh' % len(rows))

    if args.worse:
        # Sweeping only the metrics the table prints would answer the question
        # from partial data, so this covers all six.
        print()
        print('Moi cho de xuat KEM hon moc, tren ca sau chi so:')
        found_worse = 0
        for row in rows:
            for name in METRICS:
                base, prop = row['base_' + name], row['prop_' + name]
                if base is None or prop is None:
                    continue
                delta = (prop - base) * DIRECTION[name]
                if delta < 0:
                    found_worse += 1
                    print('  %-12s %-11s %-11s %9.4f -> %9.4f  (kem %.4f)'
                          % (row['dataset'], row['backbone'], name,
                             base, prop, -delta))
        if not found_worse:
            print('  khong co cho nao')
        print('  -> %d cho kem hon tren %d o x %d chi so = %d phep so sanh'
              % (found_worse, len(rows), len(METRICS),
                 len(rows) * len(METRICS)))

    if args.csv and rows:
        with open(args.csv, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print('da ghi %s' % args.csv)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
