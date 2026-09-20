#!/usr/bin/env python3
"""Verify the remaining fixed five-backbone ImageNet-R ablation, seed42.

Three arms: route_only(w=.7,beta=0), class_plain(w=1,beta=.5),
full_plain(w=.7,beta=.5); gate=none throughout. Reuse Sup class_plain.
Read all five verified identity baselines, never rerun completed grid/controls.
No tuning, training, checkpoint writes, or log replacement.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import verify_paper_backbone_grid as grid

verify, finish = grid.verify, grid.finish
ARMS = ('route_only', 'class_plain', 'full_plain')
DRIVERS = grid.DRIVER_FILES + ('verify_paper_ablation.py',)


def command(backbone, arm, tii, lora, data_path):
    if arm not in ARMS:
        raise ValueError('Unknown ablation arm: ' + arm)
    cmd = grid.command(None, 'imr', backbone, 'full', tii, lora, data_path)
    finish.set_flag(cmd, '--rp_class_fusion_gate', 'none')
    if arm == 'route_only':
        finish.set_flag(cmd, '--rp_class_fusion_weight', '0.0')
    if arm == 'class_plain':
        finish.set_flag(cmd, '--rp_route_fusion_weight', '1.0')
    return cmd


def original_driver_hash():
    blob = subprocess.check_output(['git', 'show', grid.LEGACY_DRIVER_COMMIT +
                                    ':tools/finish_soict_checks.py'], cwd=ROOT)
    return hashlib.sha256(blob).hexdigest()


def baseline(args, backbone, lora, tii, data_path, records):
    cmd = grid.command(None, 'imr', backbone, 'identity', tii, lora, data_path)
    if backbone == 'sup':
        path = args.output_root / verify.log_name(lora.name, 'identity', 'maskfix_verify_v2')
        meta = dict(source_sha256=grid.PAPER_SOURCE, checkpoints=records, command=cmd)
    else:
        path = args.output_root / (lora.name + '__paper_backbone_verify_v1__identity.log')
        # These original grid driver files must stay unchanged. Do not extend
        # the old runner in place: that would invalidate its completed logs.
        hashes = {n: finish.file_hash(ROOT / 'tools' / n) for n in grid.DRIVER_FILES}
        meta = dict(source_sha256=grid.PAPER_SOURCE, checkpoints=records, command=cmd,
                    driver_sha256=hashes, purpose='verify fixed paper 20-cell grid')
    return path, grid.read_verified(path, meta, 10)


def historical(root, backbone, lora, arm, new):
    # Exact historical name, no fuzzy search and no claim of old provenance.
    weight = '1p0' if arm == 'class_plain' else '0p7'
    beta = '0p0' if arm == 'route_only' else '0p5'
    model_tag = grid.BACKBONES[backbone][2][1]
    name = (lora.name + verify.RP_PREFIX + 'w' + weight +
            'lsw0p0c0ca0cw' + beta + 'sh1p0m1' + model_tag + '.log')
    path = root / name
    if not path.is_file():
        return dict(status='NOT_FOUND', expected_log=name)
    try:
        old = verify.read_stages(path, 10)
    except ValueError as error:
        return dict(status='INCOMPLETE', log=name, reason=str(error))
    return dict(status='HISTORICAL_ONLY', log=name, final=old[10],
                final_delta={m: new[10][m] - old[10][m] for m in finish.METRICS})


def audit_notmnist(root):
    """Read only the two reported files and matching ZIP members; never extract."""
    from PIL import Image
    names = ('F/Q3Jvc3NvdmVyIEJvbGRPYmxpcXVlLnR0Zg==.png',
             'A/RGVtb2NyYXRpY2FCb2xkT2xkc3R5bGUgQm9sZC50dGY=.png')
    def inspect(raw):
        out = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        try:
            with Image.open(io.BytesIO(raw)) as im:
                im.convert('RGB').load()
            out['status'] = 'READABLE'
        except (OSError, ValueError) as error:
            out.update(status='UNREADABLE', reason=str(error))
        return out
    result = []
    archive = root / 'notMNIST.zip'
    for name in names:
        item = dict(file=name, disk=[], archive=[])
        for split in ('Train', 'Test'):
            path = root / 'notMNIST' / split / name
            if path.is_file():
                item['disk'].append(dict(split=split, **inspect(path.read_bytes())))
        if archive.is_file():
            try:
                with zipfile.ZipFile(archive) as z:
                    for member in z.infolist():
                        if member.filename.replace('\\', '/').endswith('/' + name):
                            if member.file_size > 10 * 1024**2:
                                item['archive'].append(dict(member=member.filename, status='OVERSIZE_NOT_READ'))
                            else:
                                item['archive'].append(dict(member=member.filename, **inspect(z.read(member))))
            except (OSError, ValueError, zipfile.BadZipFile) as error:
                item['archive_error'] = str(error)
        result.append(item)
    print('NOTMNIST_READ_ONLY_AUDIT ' + json.dumps(result, sort_keys=True), flush=True)
    return result


def execute(args):
    source = verify.source_digest()
    print('EVALUATOR_SOURCE_SHA256=' + source, flush=True)
    if source != grid.PAPER_SOURCE:
        raise ValueError('Evaluator is not the restored paper version')
    drivers = {n: finish.file_hash(ROOT / 'tools' / n) for n in DRIVERS}
    data_path = grid.resolve_data(args.data_root, 'imr')
    data_audit = audit_notmnist(args.data_root) if args.audit_notmnist else None
    jobs, baselines, stamps = [], {}, {}
    for backbone in grid.BACKBONES:
        print('PREFLIGHT_ABLATION', backbone, flush=True)
        _, lora, tii = grid.historical_pair(args.output_root, 'imr', backbone)
        cmds = {arm: command(backbone, arm, tii, lora, data_path) for arm in ARMS}
        for cmd in cmds.values():
            grid.parsed_config(cmd)
        config = grid.parsed_config(cmds['class_plain'])
        records, current = grid.manifest(tii, lora, config, grid.BACKBONES[backbone][1])
        stamps.update(current)
        path, rows = baseline(args, backbone, lora, tii, data_path, records)
        baselines[backbone] = dict(log=str(path), final=rows[10])
        for arm in ARMS:
            meta = dict(source_sha256=source, checkpoints=records, command=cmds[arm])
            if backbone == 'sup' and arm == 'class_plain':
                path = args.output_root / (lora.name + '__soict_final_v1__class_plain.log')
                meta['driver_sha256'] = original_driver_hash()
                # Explicitly read-only; this run was already completed.
                rows = grid.read_verified(path, meta, 10)
            else:
                path = args.output_root / (lora.name + '__' + args.tag + '__' + arm + '.log')
                meta.update(driver_sha256=drivers, purpose='verify fixed ungated paper ablation')
                rows = grid.read_verified(path, meta, 10) if path.exists() else None
            jobs.append(dict(backbone=backbone, arm=arm, lora=lora, path=path,
                             command=cmds[arm], meta=meta, rows=rows))
    print('ABLATION_PREFLIGHT arms=15 reused=%d missing=%d' %
          (sum(j['rows'] is not None for j in jobs), sum(j['rows'] is None for j in jobs)), flush=True)
    results = []
    for job in jobs:
        grid.assert_unchanged(source, drivers, stamps)
        rows = job['rows'] if job['rows'] is not None else grid.run_or_read(
            job['path'], job['command'], job['meta'], 10, args.run)
        result = dict(backbone=job['backbone'], arm=job['arm'], log=str(job['path']),
                      final=rows[10], stages=rows,
                      historical=historical(args.output_root, job['backbone'], job['lora'], job['arm'], rows))
        results.append(result)
        print('ABLATION_FINAL ' + json.dumps(result, sort_keys=True), flush=True)
    grid.assert_unchanged(source, drivers, stamps)
    contrasts = []
    for backbone in grid.BACKBONES:
        rows = {r['arm']: r['final'] for r in results if r['backbone'] == backbone}
        rows['identity'] = baselines[backbone]['final']
        for a, b in (('route_only', 'identity'), ('class_plain', 'identity'),
                     ('full_plain', 'class_plain')):
            delta = {m: rows[a][m]-rows[b][m] for m in finish.METRICS}
            contrast = dict(backbone=backbone, contrast=a+'_MINUS_'+b, delta=delta)
            contrasts.append(contrast)
            print('ABLATION_CONTRAST ' + json.dumps(contrast, sort_keys=True), flush=True)
    report = dict(source_sha256=source, baselines=baselines, results=results,
                  contrasts=contrasts, notmnist_audit=data_audit)
    path = args.output_root / (args.tag + '__summary.json')
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != json.loads(json.dumps(report)):
            raise ValueError('Existing summary differs; preserve it: ' + str(path))
    else:
        with path.open('x', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write('\n')
    print('SUMMARY=' + str(path), flush=True)
    print('PAPER_ABLATION_COMPLETE=15/15', flush=True)
    print('Descriptive seed42 contrasts only; no parameter selection or multi-seed claim.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--tag', default='paper_ablation_verify_v1')
    parser.add_argument('--audit-notmnist', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.tag):
        parser.error('Invalid tag')
    args.output_root, args.data_root = args.output_root.resolve(), args.data_root.resolve()
    if not args.output_root.is_dir() or not args.data_root.is_dir():
        parser.error('Existing output/data directories required')
    import fcntl
    # Share the previous grid lock: do not overlap the two runners.
    with (args.output_root / '.paper_backbone_verifier.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.CalledProcessError) as error:
        raise SystemExit('PAPER_ABLATION_STOP: ' + str(error))
