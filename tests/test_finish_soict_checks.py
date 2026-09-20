"""CPU-only tests of evidence selection and commands; not model evaluation."""
import contextlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import finish_soict_checks as finish


class FakeTensor:
    ndim = 4

    def __init__(self, shape):
        self.shape = shape

    def __getitem__(self, index):
        return ('adapter', index, self.shape)


class EvidenceTests(unittest.TestCase):
    def args(self):
        return SimpleNamespace(dataset='cub200', data_root=Path('/data'),
                               resolved_data_path=Path('/data'), batch_size=24,
                               port=29558, output_root=Path('/out'))

    def cmd(self, variant):
        return finish.variant_command(self.args(), 43, Path('/tii'),
                                      Path('/lora'), 'vit_base_patch16_224', variant)

    def test_full_unchanged(self):
        self.assertEqual(self.cmd('full'), finish.verify.command(
            self.args(), 43, 'full', Path('/tii'), Path('/lora')))

    def test_rp_only_bypasses_both_fusion_routes(self):
        cmd = self.cmd('rp_only')
        self.assertNotIn('--rp_route_fusion', cmd)
        self.assertNotIn('--rp_route_fusion_drm', cmd)
        self.assertIn('--rp_head', cmd)
        self.assertEqual(cmd[cmd.index('--rp_class_fusion_weight') + 1], '0.0')
        self.assertIn('--eval', cmd)
        self.assertIn('--strict_exemplar_free', cmd)

    def test_class_controls_differ_only_in_gate(self):
        plain, gated = self.cmd('class_plain'), self.cmd('class_gate')
        self.assertEqual(plain[plain.index('--rp_route_fusion_weight') + 1], '1.0')
        self.assertIn('--rp_route_fusion_drm', plain)
        idx = plain.index('--rp_class_fusion_gate') + 1
        self.assertEqual(plain[idx], 'none')
        plain[idx] = 'margin'
        self.assertEqual(plain, gated)

    def test_weight_only_changes_w(self):
        cmd = self.cmd('w06')
        idx = cmd.index('--rp_route_fusion_weight') + 1
        self.assertEqual(cmd[idx], '0.6')
        cmd[idx] = '0.7'
        self.assertEqual(cmd, self.cmd('full'))

    def test_unknown_variant_rejected(self):
        with self.assertRaises(ValueError):
            self.cmd('typo')

    def state(self):
        state = dict.fromkeys(('cls_token', 'pos_embed', 'patch_embed.proj.weight',
                               'blocks.0.norm1.weight', 'blocks.11.norm1.weight',
                               'norm.weight', 'head.weight', 'mlp.0.weight',
                               'fc_norm.weight'), 'tensor')
        state.update({key: FakeTensor((10, 5, 768, 8) if key.endswith('_A')
                                      else (10, 5, 8, 768)) for key in finish.LORA_KEYS})
        return state

    def test_actual_feature_path_excludes_classifier(self):
        selected = finish.feature_tensors(self.state())
        for key in ('head.weight', 'mlp.0.weight', 'fc_norm.weight'):
            self.assertNotIn(key, selected)
        for key in finish.LORA_KEYS:
            self.assertEqual(selected[key][1], 0)
        self.assertIn('norm.weight', selected)

    def test_missing_backbone_rejected(self):
        state = self.state()
        del state['blocks.11.norm1.weight']
        with self.assertRaises(ValueError):
            finish.feature_tensors(state)

    def test_wrong_rank_rejected(self):
        state = self.state()
        state[finish.LORA_KEYS[0]] = FakeTensor((10, 5, 768, 4))
        with self.assertRaises(ValueError):
            finish.feature_tensors(state)

    def moco_state(self):
        state = self.state()
        del state['norm.weight']
        state['fc_norm.bias'] = 'tensor'
        return state

    def test_moco_identity_norm_is_valid(self):
        selected = finish.feature_tensors(self.moco_state(), 'vit_base_patch16_224_mocov3')
        self.assertNotIn('fc_norm.weight', selected)
        self.assertNotIn('fc_norm.bias', selected)
        self.assertIn('blocks.11.norm1.weight', selected)
        self.assertTrue(all(k in selected for k in finish.LORA_KEYS))

    def test_missing_norm_not_allowed_for_other_models(self):
        for model in ('vit_base_patch16_224', 'vit_base_patch16_224_dino'):
            with self.assertRaises(ValueError):
                finish.feature_tensors(self.moco_state(), model)

    def test_moco_rejects_missing_fc_norm_or_unexpected_norm(self):
        state = self.moco_state()
        del state['fc_norm.bias']
        with self.assertRaises(ValueError):
            finish.feature_tensors(state, 'vit_base_patch16_224_mocov3')
        state = self.moco_state()
        state['norm.weight'] = 'tensor'
        with self.assertRaises(ValueError):
            finish.feature_tensors(state, 'vit_base_patch16_224_mocov3')

    def test_moco_missing_transformer_still_rejected(self):
        state = self.moco_state()
        del state['blocks.11.norm1.weight']
        with self.assertRaises(ValueError):
            finish.feature_tensors(state, 'vit_base_patch16_224_mocov3')

    def test_main_tags_and_read_only(self):
        stages = {i: dict.fromkeys(finish.METRICS, 1.0) for i in range(1, 11)}
        for dataset, seed, expected in (('cub200', 42, 'maskfix_verify_v2'),
                                        ('cub200', 43, 'paperfix_verify_v3'),
                                        ('imr', 45, 'maskfix_verify_v2')):
            args = self.args()
            args.dataset = dataset
            with patch.object(finish.verify, 'run_or_read', return_value=stages) as run:
                with patch.object(finish.verify, 'read_stages', return_value=stages):
                    with contextlib.redirect_stdout(io.StringIO()):
                        finish.known_main(args, seed, Path('/tii'), Path('/lora'), [], 'sha')
            self.assertEqual(run.call_count, 3)
            for call in run.call_args_list:
                self.assertIn(expected, str(call.args[0]))
                self.assertFalse(call.kwargs['execute'])

    def test_modes_have_expected_run_counts(self):
        row = dict.fromkeys(finish.METRICS, 1.0)
        for mode, count in (('audit', 0), ('core', 36), ('weights', 4), ('ssl', 8), ('ssl', 2)):
            argv = ['check', '--mode', mode, '--output-root', '/out',
                    '--data-root', '/data']
            if count == 2:
                argv += ['--ssl-backbones', 'mocov3']
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(sys, 'argv', argv))
                stack.enter_context(patch.object(Path, 'is_dir', return_value=True))
                stack.enter_context(patch.object(finish.verify, 'source_digest', return_value='sha'))
                stack.enter_context(patch.object(finish.verify, 'resolve_imr_data_path', return_value=Path('/data')))
                stack.enter_context(patch.object(finish, 'manifest', return_value=[]))
                stack.enter_context(patch.object(finish, 'known_main', return_value={
                    arm: row for arm in ('baseline', 'identity', 'full')}))
                run = stack.enter_context(patch.object(finish, 'run_variant', return_value=row))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                finish.main()
            self.assertEqual(run.call_count, count)
            for call in run.call_args_list:
                self.assertFalse(call.args[0].run)
                if count == 2:
                    self.assertEqual(call.args[4], 'vit_base_patch16_224_mocov3')

    def test_read_only_variant_never_launches(self):
        args = self.args()
        args.run, args.tag = False, 'tag'
        with patch.object(finish.verify, 'source_digest', return_value='sha'):
            with patch.object(finish, 'file_hash', return_value='driver'):
                with patch.object(finish.verify, 'run_or_read', return_value={10: {}}) as run:
                    with contextlib.redirect_stdout(io.StringIO()):
                        finish.run_variant(args, 43, Path('/tii'), Path('/lora'),
                                           'vit_base_patch16_224', 'rp_only', [], 'sha')
        self.assertFalse(run.call_args.kwargs['execute'])


if __name__ == '__main__':
    unittest.main()
