#!/usr/bin/env python3
"""Verify the existing paper's 3x3 ImageNet-R weight table; not a new sweep.

Reuse the two verified seed42 cells, run only seven missing post-fix cells with
--run. Never change the method, tune weights, train adapters, or overwrite logs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import finish_soict_checks as finish
from tools import verify_paper_results as verify
from tools.audit_weight_evidence import read_final

PAPER_SOURCE = '6747fed54a6f9679632bfdbbed1d3ebf2832dc801cf950b6164460f5a8858da9'
LEGACY_DRIVER_COMMIT = '74ff5c9a689c1fca9ba45e652d5701a968257bed'
CELLS = tuple((w, b) for w in ('0.6', '0.7', '0.8') for b in ('0.3', '0.5', '0.7'))
REUSE = {('0.6', '0.5'), ('0.7', '0.5')}


def command(args, tii, lora, w, beta):
    if (w, beta) not in CELLS:
        raise ValueError('Not a cell in the paper table')
    cmd = verify.command(args, 42, 'full', tii, lora)
    finish.set_flag(cmd, '--rp_route_fusion_weight', w)
    finish.set_flag(cmd, '--rp_class_fusion_weight', beta)
    return cmd


def historical_name(run, w, beta):
    return (run + verify.RP_PREFIX + 'w' + w.replace('.', 'p')
            + verify.RP_SUFFIX.format(beta=beta.replace('.', 'p')) + '.log')


def reuse_cell(args, tii, lora, records, w, beta):
    cmd = command(args, tii, lora, w, beta)
    meta = dict(source_sha256=PAPER_SOURCE, checkpoints=records, command=cmd)
    if (w, beta) == ('0.7', '0.5'):
        path = args.output_root / verify.log_name(lora.name, 'full', 'maskfix_verify_v2')
    elif (w, beta) == ('0.6', '0.5'):
        # Pin the original driver version; never accept arbitrary log metadata.
        blob = subprocess.check_output(['git', 'show', LEGACY_DRIVER_COMMIT +
                                        ':tools/finish_soict_checks.py'], cwd=ROOT)
        meta['driver_sha256'] = hashlib.sha256(blob).hexdigest()
        path = args.output_root / (lora.name + '__soict_final_v1__w06.log')
    else:
        raise ValueError('Cell has no verified reusable source')
    return verify.run_or_read(path, cmd, meta, execute=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--tag', default='paper_grid_verify_v1')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.tag):
        parser.error('Invalid tag')
    args.output_root, args.data_root = args.output_root.resolve(), args.data_root.resolve()
    if not args.output_root.is_dir() or not args.data_root.is_dir():
        parser.error('Existing output and data directories required')
    args.dataset, args.batch_size, args.port, args.data_path = 'imr', 24, 29558, None
    source = verify.source_digest()
    print('EVALUATOR_SOURCE_SHA256=' + source, flush=True)
    if source != PAPER_SOURCE:
        raise ValueError('Evaluator is not the restored paper source. Do not alter metadata.')
    args.resolved_data_path = verify.resolve_imr_data_path(args.data_root)
    lora = args.output_root / 'imr_lora_rank8_baseline_10tasks_seed42'
    tii = args.output_root / 'imr_tii_original_10tasks_seed42'
    records = finish.manifest(tii, lora, 'Split-Imagenet-R', 'vit_base_patch16_224', 42)
    drivers = {name: finish.file_hash(ROOT / 'tools' / name) for name in
               ('verify_paper_weight_grid.py', 'verify_paper_results.py',
                'finish_soict_checks.py', 'audit_weight_evidence.py')}
    rows = {}
    for w, beta in sorted(REUSE):
        rows[w, beta] = reuse_cell(args, tii, lora, records, w, beta)
        print('REUSED_VERIFIED_CELL', w, beta, flush=True)
    historical = {}
    for w, beta in CELLS:
        path = args.output_root / historical_name(lora.name, w, beta)
        stage, old = read_final(str(path), expected_tasks=10)
        if stage != 10:
            raise ValueError('Missing complete historical cell: ' + str(path))
        historical[w, beta] = old
    for w, beta in CELLS:
        if (w, beta) in REUSE:
            continue
        if verify.source_digest() != source:
            raise ValueError('Evaluator changed during verification')
        cmd = command(args, tii, lora, w, beta)
        name = (lora.name + '__' + args.tag + '__w' + w.replace('.', 'p')
                + '_b' + beta.replace('.', 'p') + '.log')
        path = args.output_root / name
        meta = dict(source_sha256=source, checkpoints=records, command=cmd,
                    driver_sha256=drivers, purpose='verify existing paper table')
        if args.run and not path.exists():
            import torch
            if not torch.cuda.is_available() or torch.cuda.mem_get_info(0)[0] < 16 * 1024**3:
                raise RuntimeError('GPU busy or unavailable; retry later, do not stop another job')
        rows[w, beta] = verify.run_or_read(path, cmd, meta, execute=args.run)
        print('GRID_FINAL', w, beta, json.dumps(rows[w, beta][10], sort_keys=True), flush=True)
    if verify.source_digest() != source or any(
            finish.file_hash(ROOT / 'tools' / name) != digest for name, digest in drivers.items()):
        raise ValueError('Source/driver changed during verification; do not use this batch')
    match = True
    for w, beta in CELLS:
        match = verify.compare({10: historical[w, beta]}, rows[w, beta], [10],
                               finish.METRICS, label='HISTORICAL_w%s_b%s' % (w, beta)) and match
        print('PAPER_GRID_CELL', w, beta, 'Acc@1=%.4f' % rows[w, beta][10]['Acc@1'])
    print('PAPER_WEIGHT_GRID_COMPLETE=9/9', flush=True)
    print('HISTORICAL_GRID_MATCH=' + ('PASS' if match else 'FAIL'), flush=True)
    if not match:
        raise ValueError('Grid evaluated, but historical numbers differ; review before updating paper')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.CalledProcessError) as error:
        raise SystemExit('PAPER_GRID_STOP: ' + str(error))
