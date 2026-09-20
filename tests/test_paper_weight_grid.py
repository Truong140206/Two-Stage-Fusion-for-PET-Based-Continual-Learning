"""Dependency-free guards for verifying the existing paper weight grid."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import verify_paper_weight_grid as grid


class GridTests(unittest.TestCase):
    def args(self):
        return SimpleNamespace(dataset='imr', batch_size=24, port=29558,
                               resolved_data_path=Path('/data'), output_root=Path('/out'))

    def test_exactly_seven_new_cells(self):
        self.assertEqual(len(grid.CELLS), 9)
        self.assertEqual(len(set(grid.CELLS) - grid.REUSE), 7)

    def test_commands_keep_everything_except_two_weights(self):
        base = grid.verify.command(self.args(), 42, 'full', Path('/tii'), Path('/lora'))
        for w, beta in grid.CELLS:
            cmd = grid.command(self.args(), Path('/tii'), Path('/lora'), w, beta)
            grid.finish.set_flag(cmd, '--rp_route_fusion_weight', '0.7')
            grid.finish.set_flag(cmd, '--rp_class_fusion_weight', '0.5')
            self.assertEqual(cmd, base)

    def test_unknown_cell_rejected(self):
        with self.assertRaises(ValueError):
            grid.command(self.args(), Path('/tii'), Path('/lora'), '0.9', '0.5')

    def test_historical_filename_is_exact_paper_configuration(self):
        name = grid.historical_name('imr_lora_rank8_baseline_10tasks_seed42', '0.6', '0.3')
        self.assertIn('f1d1w0p6lsw0p0c0ca0cw0p3sh1p0m1gmargin.log', name)
        self.assertNotIn('__', name)

    def test_reused_main_never_runs(self):
        with patch.object(grid.verify, 'run_or_read', return_value={}) as run:
            grid.reuse_cell(self.args(), Path('/tii'), Path('/lora'), [], '0.7', '0.5')
        self.assertFalse(run.call_args.kwargs['execute'])
        self.assertIn('maskfix_verify_v2', str(run.call_args.args[0]))
        self.assertNotIn('driver_sha256', run.call_args.args[2])

    def test_w06_driver_is_pinned_to_original_git_version(self):
        with patch.object(grid.subprocess, 'check_output', return_value=b'original driver') as git:
            with patch.object(grid.verify, 'run_or_read', return_value={}) as run:
                grid.reuse_cell(self.args(), Path('/tii'), Path('/lora'), [], '0.6', '0.5')
        self.assertIn(grid.LEGACY_DRIVER_COMMIT, git.call_args.args[0][-1])
        self.assertEqual(run.call_args.args[2]['driver_sha256'],
                         grid.hashlib.sha256(b'original driver').hexdigest())
        self.assertFalse(run.call_args.kwargs['execute'])


if __name__ == '__main__':
    unittest.main()
