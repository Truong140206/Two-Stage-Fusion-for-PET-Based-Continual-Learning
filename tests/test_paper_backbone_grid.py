"""CPU-only regression tests for the deadline grid driver."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import verify_paper_backbone_grid as grid


def log_text(meta, tasks=5, code=0):
    lines = ['VERIFICATION_META=' + json.dumps(meta, sort_keys=True)]
    for task in range(1, tasks + 1):
        lines.append('[Average accuracy till task%d] Acc@task: 80 Acc@1: 75 '
                     'Acc@5: 90 Loss: 1 Forgetting: 3 Backward: -2' % task)
    lines.append('VERIFICATION_EXIT_CODE=%d' % code)
    return '\n'.join(lines) + '\n'


class FakeTensor:
    def __init__(self, shape):
        self.shape, self.ndim = shape, len(shape)
    def __getitem__(self, index):
        return FakeTensor(self.shape[1:])


class BackboneGridTests(unittest.TestCase):
    def cmd(self, dataset='imr', backbone='sup', arm='full'):
        return grid.command(None, dataset, backbone, arm, Path('/tii'), Path('/lora'), Path('/data'))

    def test_exact_twenty_cells_no_mae(self):
        self.assertEqual(len(grid.DATASETS) * len(grid.BACKBONES), 20)
        self.assertNotIn('mae', grid.BACKBONES)

    def test_all_forty_commands_parse_and_are_evaluation_only(self):
        for ds in grid.DATASETS:
            for bb in grid.BACKBONES:
                for arm in ('identity', 'full'):
                    cmd = self.cmd(ds, bb, arm)
                    cfg = grid.parsed_config(cmd)
                    self.assertTrue(cfg.eval)
                    self.assertTrue(cfg.strict_exemplar_free)
                    self.assertEqual(cfg.dataset, grid.DATASETS[ds][0])
                    self.assertEqual(cfg.num_tasks, grid.DATASETS[ds][2])
                    self.assertEqual(cfg.size, cfg.num_tasks)
                    self.assertEqual(cfg.model, grid.BACKBONES[bb][0])
                    self.assertEqual(cfg.rp_route_fusion_weight, 1.0 if arm == 'identity' else .7)
                    self.assertEqual(cfg.rp_class_fusion_weight, 0.0 if arm == 'identity' else .5)
                    self.assertEqual(cfg.rp_class_fusion_gate, 'margin')
                    self.assertFalse(cfg.shuffle)

    def test_reusable_main_command_byte_equal(self):
        for ds in ('imr', 'cifar100'):
            for arm in ('identity', 'full'):
                args = SimpleNamespace(dataset=ds, batch_size=24, port=29558,
                                       resolved_data_path=Path('/data'))
                self.assertEqual(self.cmd(ds, 'sup', arm),
                                 grid.verify.command(args, 42, arm, Path('/tii'), Path('/lora')))

    def test_reusable_ssl_command_byte_equal(self):
        for bb in ('mocov3', 'dino', 'ibot1k', 'ibot21k'):
            args = SimpleNamespace(dataset='imr', batch_size=24, port=29558,
                                   resolved_data_path=Path('/data'))
            self.assertEqual(self.cmd('imr', bb),
                             grid.finish.variant_command(args, 42, Path('/tii'), Path('/lora'),
                                                         grid.BACKBONES[bb][0], 'full'))

    def test_unknown_arm_is_not_silently_full(self):
        with self.assertRaises(ValueError):
            self.cmd(arm='class_gate')

    def state(self, tasks, moco=False):
        out = {k: FakeTensor((1,)) for k in
               ('cls_token', 'pos_embed', 'patch_embed.weight', 'blocks.0.weight', 'blocks.11.weight')}
        out.update({k: FakeTensor((1,)) for k in
                    (('fc_norm.weight', 'fc_norm.bias') if moco else ('norm.weight',))})
        for key in grid.finish.LORA_KEYS:
            out[key] = FakeTensor((tasks, 5, 768, 8) if key.endswith('_A') else (tasks, 5, 8, 768))
        return out

    def test_five_task_features_and_frozen_first_adapter(self):
        state = self.state(5)
        out = grid.feature_tensors(state, 'vit_base_patch16_224', 5)
        self.assertEqual(out['lora_layer.k_lora_A'].shape, (5, 768, 8))
        with self.assertRaises(ValueError):
            grid.feature_tensors(state, 'vit_base_patch16_224', 10)

    def test_ten_task_selection_matches_existing_audit(self):
        state = self.state(10)
        old = grid.finish.feature_tensors(state)
        new = grid.feature_tensors(state, 'vit_base_patch16_224', 10)
        self.assertEqual({k: t.shape for k, t in old.items()}, {k: t.shape for k, t in new.items()})

    def test_moco_fc_norm_not_part_of_pre_logits(self):
        out = grid.feature_tensors(self.state(5, True), 'vit_base_patch16_224_mocov3', 5)
        self.assertNotIn('fc_norm.weight', out)

    def test_checkpoint_configuration_mismatch_stops(self):
        expected = grid.parsed_config(self.cmd('fivedatasets'))
        saved = vars(expected).copy()
        grid.checkpoint_args(saved, expected, 'lora')
        for key, bad in (('num_tasks', 10), ('seed', 43), ('size', 10), ('shuffle', True)):
            with self.assertRaises(ValueError):
                grid.checkpoint_args(dict(saved, **{key: bad}), expected, 'lora')

    def test_five_task_complete_log_is_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'eval.log'
            path.write_text(log_text({'a': 1}), encoding='utf-8')
            self.assertEqual(len(grid.read_verified(path, {'a': 1}, 5)), 5)
            with self.assertRaises(ValueError):
                grid.read_verified(path, {'a': 1}, 10)

    def test_metadata_nonzero_and_incomplete_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'eval.log'
            for text in (log_text({'a': 2}), log_text({'a': 1}, code=1),
                         log_text({'a': 1}).rsplit('VERIFICATION_EXIT_CODE', 1)[0]):
                path.write_text(text, encoding='utf-8')
                with self.assertRaises(ValueError):
                    grid.read_verified(path, {'a': 1}, 5)

    def test_existing_log_never_launches_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'eval.log'
            text = log_text({})
            path.write_text(text, encoding='utf-8')
            with patch.object(grid, 'gpu_ready') as gpu:
                grid.run_or_read(path, [], {}, 5, True)
                gpu.assert_not_called()
            self.assertEqual(path.read_text(encoding='utf-8'), text)

    def test_busy_gpu_does_not_create_partial_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'eval.log'
            with patch.object(grid, 'gpu_ready', side_effect=RuntimeError('busy')):
                with self.assertRaises(RuntimeError):
                    grid.run_or_read(path, [], {}, 5, True)
            self.assertFalse(path.exists())

    def test_read_only_never_runs_missing_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(grid, 'gpu_ready') as gpu:
                with self.assertRaises(FileNotFoundError):
                    grid.run_or_read(Path(tmp) / 'eval.log', [], {}, 5, False)
                gpu.assert_not_called()

    def test_historical_moco_alias_ambiguity_rejected(self):
        with patch.object(grid.collect, 'find_log', side_effect=lambda root, ds, tags, *rest:
                          '/out/' + tags[0] + '.log'):
            with self.assertRaises(ValueError):
                grid.historical_pair(Path('/out'), 'imr', 'mocov3')

    def test_known_logs_are_exactly_eight_arms(self):
        found = []
        args = SimpleNamespace(output_root=Path('/out'))
        with patch.object(Path, 'exists', return_value=True), \
                patch.object(grid.subprocess, 'check_output', return_value=b'driver'):
            for ds in grid.DATASETS:
                for bb in grid.BACKBONES:
                    for arm in ('identity', 'full'):
                        if grid.known_log(args, ds, bb, arm, Path('/lora'), [], []):
                            found.append((ds, bb, arm))
        self.assertEqual(len(found), 8)
        self.assertNotIn(('imr', 'mocov3', 'identity'), found)

    def test_ssl_reuse_pins_original_driver_not_log_claim(self):
        args = SimpleNamespace(output_root=Path('/out'))
        with patch.object(Path, 'exists', return_value=True), \
                patch.object(grid.subprocess, 'check_output', return_value=b'driver') as git:
            _, meta = grid.known_log(args, 'imr', 'dino', 'full', Path('/lora'), [], [])
        self.assertIn(grid.LEGACY_DRIVER_COMMIT, git.call_args.args[0][-1])
        self.assertEqual(meta['driver_sha256'], grid.hashlib.sha256(b'driver').hexdigest())

    def test_imagenet_a_missing_split_does_not_create_anything(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                grid.resolve_data(root, 'ima')
            self.assertEqual(list(root.iterdir()), [])

    def test_checkpoint_mutation_stops_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checkpoint'
            path.write_text('test')
            with patch.object(grid.verify, 'source_digest', return_value='same'):
                with self.assertRaises(ValueError):
                    grid.assert_unchanged('same', {}, {str(path): (0, 0)})

    def test_retention_decomposition_excludes_last_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'log'
            path.write_text('* Acc@task 80 Acc@1 80.000 Acc@5 90 loss 1\n'
                            '[Average accuracy till task1] Acc@1: 80\n'
                            '* Acc@task 80 Acc@1 75.000 Acc@5 90 loss 1\n'
                            '* Acc@task 80 Acc@1 90.000 Acc@5 90 loss 1\n'
                            '[Average accuracy till task2] Acc@1: 82.5\n')
            rows = {1: {'Acc@1': 80}, 2: {'Acc@1': 82.5, 'Forgetting': 5, 'Backward': -5}}
            result = grid.retention_decomposition(path, 2, rows)
            self.assertEqual(result['status'], 'PASS_ROUNDED_3DP')
            self.assertEqual(result['old_task_final_mean'], 75)
            self.assertEqual(result['learning_time_old_task_mean'], 80)
            rows[2]['Backward'] = 0
            self.assertEqual(grid.retention_decomposition(path, 2, rows)['status'], 'UNAVAILABLE')

    def test_aggregate_only_log_cannot_support_decomposition(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'log'
            path.write_text(log_text({}))
            self.assertEqual(grid.retention_decomposition(path, 5, {})['status'], 'UNAVAILABLE')

    def test_end_to_end_reuse_exports_summary_without_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(output_root=root, data_root=root, datasets=['imr'],
                                   tag='test', run=False)
            rows = {i: dict(zip(grid.finish.METRICS, (80, 75, 90, 1, 3, -2)))
                    for i in range(1, 11)}
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(grid.verify, 'source_digest', return_value=grid.PAPER_SOURCE))
                stack.enter_context(patch.object(grid.finish, 'file_hash', return_value='hash'))
                stack.enter_context(patch.object(grid, 'resolve_data', return_value=root))
                stack.enter_context(patch.object(grid, 'historical_pair',
                                                return_value=({'identity': root, 'full': root}, root/'lora', root/'tii')))
                stack.enter_context(patch.object(grid, 'manifest', return_value=([], {})))
                stack.enter_context(patch.object(grid.verify, 'read_stages', return_value=rows))
                stack.enter_context(patch.object(grid, 'known_log', return_value=(root, {})))
                stack.enter_context(patch.object(grid, 'read_verified', return_value=rows))
                stack.enter_context(patch.object(grid, 'retention_decomposition',
                                                return_value={'status': 'UNAVAILABLE'}))
                gpu = stack.enter_context(patch.object(grid, 'gpu_ready'))
                with contextlib.redirect_stdout(io.StringIO()) as stdout:
                    grid.execute_grid(args)
                    grid.execute_grid(args)  # same summary is reusable, not overwritten
                gpu.assert_not_called()
            self.assertIn('PAPER_BACKBONE_GRID_COMPLETE=5/5', stdout.getvalue())
            report = json.loads((root/'test__imr__summary.json').read_text())
            self.assertEqual(len(report['results']), 10)
            self.assertTrue(report['historical_core_match'])


if __name__ == '__main__':
    unittest.main()
