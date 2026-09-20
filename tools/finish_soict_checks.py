#!/usr/bin/env python3
"""Deadline evidence checks. No training, checkpoint writes, or log replacement.

audit: CPU-only checkpoint invariance and the existing 3-dataset main results.
core: RP-only and ungated/gated class-only, 3 datasets x 4 seeds = 36 runs.
weights: full w=.6 on ImageNet-R, 4 seeds; compare with verified w=.7.
ssl: gated class-only/full on four SSL backbones at seed 42 = 8 runs.
Evaluation requires --run. With no --run, missing logs are reported as errors.
Only load the authors' trusted checkpoints (torch.load weights_only=False).
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import verify_paper_results as verify
from tools.audit_weight_evidence import paired
from protocols import exemplar_free_violations

BACKBONES = (
    ('_dino', 'vit_base_patch16_224_dino', 'dino_vitbase16_pretrain.pth'),
    ('_ibot1k', 'vit_base_patch16_224_ibot', 'checkpoint_teacher.pth'),
    ('_ibot21k', 'vit_base_patch16_224_21k_ibot', 'checkpoint.pth'),
    ('_mocov3', 'vit_base_patch16_224_mocov3', 'mocov3-vit-base-300ep.pth'),
)
METRICS = verify.CORE + verify.RETENTION
LORA_KEYS = tuple('lora_layer.' + x for x in
                  ('k_lora_A', 'k_lora_B', 'v_lora_A', 'v_lora_B'))


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def feature_tensors(state, model='vit_base_patch16_224'):
    """Actual pre_logits path: embeddings, transformer, norm, adapter index 0.

    The classifier MLP/head/fc_norm run AFTER pre_logits and are excluded.
    Other task adapters are intentionally excluded. No numerical tolerance.
    """
    selected = {k: v for k, v in state.items()
                if k in ('cls_token', 'pos_embed')
                or k.startswith(('patch_embed.', 'blocks.', 'norm.'))}
    if not all(any(k.startswith(prefix) for k in selected)
               for prefix in ('patch_embed.', 'blocks.0.', 'blocks.11.')):
        raise ValueError('Unexpected ViT feature state; cannot certify invariance')
    has_norm = any(k.startswith('norm.') for k in selected)
    if model == 'vit_base_patch16_224_mocov3':
        # This constructor sets fc_norm=True: norm is Identity (no tensors).
        # fc_norm is applied AFTER pre_logits; validate its presence, but do
        # not include classifier-side parameters in the RP feature hash.
        if has_norm or not {'fc_norm.weight', 'fc_norm.bias'} <= state.keys():
            raise ValueError('Unexpected MoCo-v3 norm/fc_norm structure')
    elif not has_norm:
        raise ValueError('Missing pre_logits norm tensors: ' + model)
    if not all(k in selected for k in ('cls_token', 'pos_embed')):
        raise ValueError('Missing embedding tensors')
    for key in LORA_KEYS:
        tensor = state[key]
        if tensor.ndim != 4 or tensor.shape[0] != 10 or tensor.shape[1] != 5:
            raise ValueError('Unexpected LoRA pool shape: ' + key)
        rank_axis = -1 if key.endswith('_A') else -2
        if tensor.shape[rank_axis] != 8:
            raise ValueError('Expected LoRA rank 8: ' + key)
        selected[key] = tensor[0]
    return selected


def feature_hash(state, components=None, model='vit_base_patch16_224'):
    import torch
    digest = hashlib.sha256()
    for key, tensor in sorted(feature_tensors(state, model).items()):
        tensor = tensor.detach().cpu().contiguous()
        if not torch.isfinite(tensor).all():
            raise ValueError('Nonfinite feature tensor: ' + key)
        digest.update(key.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(raw)
        if components is not None:
            components[key] = hashlib.sha256(raw).hexdigest()
    return digest.hexdigest()


def manifest(tii, lora, dataset, model, seed, pretrained=None):
    import torch
    torch.set_num_threads(2)
    records, fixed = [], None
    for role, folder in (('tii', tii), ('lora', lora)):
        for task in range(1, 11):
            path = folder / 'checkpoint' / ('task%d_checkpoint.pth' % task)
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            saved = checkpoint.get('args')
            if saved is None:
                raise ValueError('Missing saved args: ' + str(path))
            if isinstance(saved, dict):
                saved = SimpleNamespace(**saved)
            for key, expected in dict(dataset=dataset, model=model, seed=seed,
                                      num_tasks=10).items():
                if getattr(saved, key, None) != expected:
                    raise ValueError('Checkpoint mismatch %s: %s' % (key, path))
            if exemplar_free_violations(saved) or checkpoint.get('real_feature_memory'):
                raise ValueError('Checkpoint protocol mismatch: ' + str(path))
            if role == 'lora':
                for key, expected in (('lora_type', 'hide'), ('lora_rank', 8)):
                    if getattr(saved, key, None) != expected:
                        raise ValueError('Unsupported LoRA configuration: ' + str(path))
                parts = {}
                current = feature_hash(checkpoint['model'], parts, model=model)
                if fixed is None:
                    fixed = current
                    fixed_parts = parts
                elif current != fixed:
                    changed = [key for key in set(parts) | set(fixed_parts)
                               if parts.get(key) != fixed_parts.get(key)]
                    raise ValueError('FIXED_FEATURE_TENSORS=FAIL %s; changed=%s' %
                                     (path, ', '.join(sorted(changed))))
            del checkpoint
            records.append(dict(path=str(path.resolve()), sha256=file_hash(path)))
    print('FIXED_FEATURE_TENSORS=PASS', dataset, model, seed, fixed, flush=True)
    print('CHECKPOINT_PROTOCOL=PASS', dataset, model, seed, flush=True)
    if pretrained:
        path = ROOT / 'checkpoints' / pretrained
        records.append(dict(path=str(path.resolve()), sha256=file_hash(path)))
    return records


def set_flag(cmd, key, value):
    cmd[cmd.index(key) + 1] = str(value)


def variant_command(args, seed, tii, lora, model, variant):
    cmd = verify.command(args, seed, 'full', tii, lora)
    set_flag(cmd, '--model', model)
    set_flag(cmd, '--original_model', model)
    if variant == 'rp_only':
        cmd.remove('--rp_route_fusion')
        cmd.remove('--rp_route_fusion_drm')
        set_flag(cmd, '--rp_route_fusion_weight', 1.0)
        set_flag(cmd, '--rp_class_fusion_weight', 0.0)
        set_flag(cmd, '--rp_class_fusion_gate', 'none')
    elif variant in ('class_plain', 'class_gate'):
        set_flag(cmd, '--rp_route_fusion_weight', 1.0)
        set_flag(cmd, '--rp_class_fusion_gate',
                 'none' if variant == 'class_plain' else 'margin')
    elif variant == 'w06':
        set_flag(cmd, '--rp_route_fusion_weight', 0.6)
    elif variant != 'full':
        raise ValueError('Unknown variant: ' + variant)
    return cmd


def summary(title, rows):
    print('\n' + title, 'n=' + str(len(rows)), flush=True)
    for metric in METRICS:
        values = [row[metric] for row in rows]
        print(metric, 'mean=%.6f SD=%.6f' %
              (statistics.mean(values), statistics.stdev(values)
               if len(values) > 1 else 0.0), flush=True)


def contrast(title, left, right):
    print('\n' + title, 'n=' + str(len(left)), flush=True)
    for metric in METRICS:
        values = [a[metric] - b[metric] for a, b in zip(left, right)]
        print(metric, 'mean,SD,CI95=', paired(values), flush=True)


def known_main(args, seed, tii, lora, records, source):
    tag = ('paperfix_verify_v3' if args.dataset == 'cub200' and seed != 42
           else 'maskfix_verify_v2')
    rows = {}
    for arm in ('baseline', 'identity', 'full'):
        cmd = verify.command(args, seed, arm, tii, lora)
        meta = dict(source_sha256=source, checkpoints=records, command=cmd)
        path = args.output_root / verify.log_name(lora.name, arm, tag)
        rows[arm] = verify.run_or_read(path, cmd, meta, execute=False)
    if not verify.compare(rows['baseline'], rows['identity'], range(1, 11),
                          METRICS, label='IDENTITY_ALL_STAGES'):
        raise ValueError('Identity comparison failed')
    for arm in ('baseline', 'full'):
        old = verify.read_stages(args.output_root / verify.log_name(lora.name, arm))
        if not verify.compare(old, rows[arm], [10], verify.CORE,
                              label='HISTORICAL_FINAL_' + arm):
            raise ValueError('Historical comparison failed')
        for metric in verify.RETENTION:
            print('HISTORICAL_RETENTION_DELTA', arm, metric,
                  rows[arm][10][metric] - old[10][metric], flush=True)
    return {arm: stages[10] for arm, stages in rows.items()}


def run_variant(args, seed, tii, lora, model, variant, records, source):
    if verify.source_digest() != source:
        raise ValueError('Evaluator source changed during this batch')
    cmd = variant_command(args, seed, tii, lora, model, variant)
    # Deliberately distinct from historical filename parsing. Metadata is truth.
    path = args.output_root / (lora.name + '__' + args.tag + '__' + variant + '.log')
    meta = dict(source_sha256=source, checkpoints=records, command=cmd,
                driver_sha256=file_hash(Path(__file__)))
    if args.run and not path.exists():
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        free, _ = torch.cuda.mem_get_info(0)
        if free < args.min_free_gib * 1024 ** 3:
            raise RuntimeError('GPU busy: less than %.1f GiB free; retry when idle. '
                               'Do not stop another job.' % args.min_free_gib)
    rows = verify.run_or_read(path, cmd, meta, execute=args.run)
    print('EVIDENCE_FINAL', path.name, json.dumps(rows[10], sort_keys=True), flush=True)
    return rows[10]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('audit', 'core', 'weights', 'ssl'), default='audit')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--tag', default='soict_final_v1')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--min-free-gib', type=float, default=16.0)
    parser.add_argument('--ssl-backbones', nargs='+',
                        choices=tuple(suffix.lstrip('_') for suffix, _, _ in BACKBONES),
                        help='Only these SSL backbones; use mocov3 to finish the two remaining runs')
    args = parser.parse_args()
    if args.ssl_backbones and args.mode != 'ssl':
        parser.error('--ssl-backbones requires --mode ssl')
    if args.ssl_backbones and len(set(args.ssl_backbones)) != len(args.ssl_backbones):
        parser.error('Duplicate SSL backbone')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.tag):
        parser.error('Invalid tag')
    if not 5 <= args.min_free_gib <= 24:
        parser.error('--min-free-gib must be between 5 and 24')
    args.output_root = args.output_root.resolve()
    args.data_root = args.data_root.resolve()
    if not args.output_root.is_dir() or not args.data_root.is_dir():
        parser.error('Existing output/data directories required')
    # Keep identical to verified main-run commands, including port/batch size.
    args.batch_size, args.port, args.data_path = 24, 29558, None
    source = verify.source_digest()
    print('EVALUATOR_SOURCE_SHA256=' + source, flush=True)
    print('MODE=' + args.mode, 'RUN=' + str(args.run), flush=True)
    datasets = ('imr',) if args.mode in ('weights', 'ssl') else tuple(verify.DATASETS)
    for dataset in datasets:
        args.dataset = dataset
        args.resolved_data_path = (verify.resolve_imr_data_path(args.data_root)
                                  if dataset == 'imr' else args.data_root)
        results = {}
        entries = ([(suffix, model, pre, 42) for suffix, model, pre in BACKBONES
                    if not args.ssl_backbones or suffix.lstrip('_') in args.ssl_backbones]
                   if args.mode == 'ssl' else
                   [('', 'vit_base_patch16_224', None, seed) for seed in (42, 43, 44, 45)])
        for suffix, model, pretrained, seed in entries:
            run = '%s%s_lora_rank8_baseline_10tasks_seed%d' % (dataset, suffix, seed)
            lora = args.output_root / run
            tii = args.output_root / ('%s%s_tii_original_10tasks_seed%d' % (dataset, suffix, seed))
            records = manifest(tii, lora, verify.DATASETS[dataset][0], model, seed, pretrained)
            rows = {} if args.mode == 'ssl' else known_main(args, seed, tii, lora, records, source)
            variants = dict(audit=(), core=('rp_only', 'class_plain', 'class_gate'),
                            weights=('w06',), ssl=('class_gate', 'full'))[args.mode]
            for variant in variants:
                rows[variant] = run_variant(args, seed, tii, lora, model, variant, records, source)
            for variant, row in rows.items():
                results.setdefault(variant, []).append(row)
            if args.mode == 'ssl':
                contrast(model + ' FULL_MINUS_GATED_CLASS_ONLY',
                         [rows['full']], [rows['class_gate']])
        if args.mode != 'ssl':
            for variant, values in results.items():
                summary(dataset + ' ' + variant, values)
            contrast(dataset + ' FULL_MINUS_BASELINE', results['full'], results['baseline'])
            if args.mode == 'core':
                for a, b in (('full', 'rp_only'), ('full', 'class_gate'),
                             ('class_gate', 'class_plain'), ('class_plain', 'baseline')):
                    contrast(dataset + ' ' + a + '_MINUS_' + b, results[a], results[b])
            if args.mode == 'weights':
                contrast(dataset + ' FULL_W06_MINUS_W07', results['w06'], results['full'])
    selection = (':' + ','.join(args.ssl_backbones) if args.ssl_backbones else '')
    print('SOICT_EVIDENCE_COMPLETE=' + args.mode + selection, flush=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        raise SystemExit('SOICT_EVIDENCE_STOP: ' + str(error))
