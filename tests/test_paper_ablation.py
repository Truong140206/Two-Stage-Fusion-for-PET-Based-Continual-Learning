"""Dependency-free ablation command/provenance tests; no GPU required."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from tools import verify_paper_ablation as ab


class AblationTests(unittest.TestCase):
    def cmd(self, bb, arm):
        return ab.command(bb, arm, Path('/tii'), Path('/lora'), Path('/data'))

    def test_exact_fifteen_configurations(self):
        self.assertEqual(len(ab.ARMS) * len(ab.grid.BACKBONES), 15)
        for bb in ab.grid.BACKBONES:
            for arm in ab.ARMS:
                cfg = ab.grid.parsed_config(self.cmd(bb, arm))
                self.assertTrue(cfg.eval and cfg.strict_exemplar_free)
                self.assertEqual(cfg.seed, 42)
                self.assertEqual(cfg.num_tasks, 10)
                self.assertEqual(cfg.rp_class_fusion_gate, 'none')
                self.assertEqual(cfg.rp_route_fusion_weight, 1.0 if arm == 'class_plain' else .7)
                self.assertEqual(cfg.rp_class_fusion_weight, 0.0 if arm == 'route_only' else .5)

    def test_class_plain_command_matches_original_driver(self):
        args = SimpleNamespace(dataset='imr', batch_size=24, port=29558,
                               resolved_data_path=Path('/data'))
        for bb, (model, _, _) in ab.grid.BACKBONES.items():
            old = ab.finish.variant_command(args, 42, Path('/tii'), Path('/lora'), model, 'class_plain')
            self.assertEqual(self.cmd(bb, 'class_plain'), old)

    def test_full_plain_changes_only_gate(self):
        for bb in ab.grid.BACKBONES:
            cmd = self.cmd(bb, 'full_plain')
            ab.finish.set_flag(cmd, '--rp_class_fusion_gate', 'margin')
            self.assertEqual(cmd, ab.grid.command(None, 'imr', bb, 'full',
                                                 Path('/tii'), Path('/lora'), Path('/data')))

    def test_unknown_arm_rejected(self):
        with self.assertRaises(ValueError):
            self.cmd('sup', 'full_gated')

    def test_old_driver_hash_pins_git_blob(self):
        with patch.object(ab.subprocess, 'check_output', return_value=b'legacy') as git:
            digest = ab.original_driver_hash()
        self.assertIn(ab.grid.LEGACY_DRIVER_COMMIT, git.call_args.args[0][-1])
        self.assertEqual(digest, ab.hashlib.sha256(b'legacy').hexdigest())

    def test_ssl_baseline_uses_original_grid_metadata(self):
        args = SimpleNamespace(output_root=Path('/out'))
        with patch.object(ab.finish, 'file_hash', return_value='hash'), \
                patch.object(ab.grid, 'read_verified', return_value={}) as read:
            path, _ = ab.baseline(args, 'dino', Path('/lora'), Path('/tii'), Path('/data'), [])
        self.assertIn('__paper_backbone_verify_v1__identity.log', str(path))
        meta = read.call_args.args[1]
        self.assertEqual(set(meta['driver_sha256']), set(ab.grid.DRIVER_FILES))
        self.assertNotIn('verify_paper_ablation.py', meta['driver_sha256'])

    def test_historical_lookup_never_uses_gated_full(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = ab.historical(Path(tmp), 'dino', Path('/lora'), 'full_plain', {})
            self.assertEqual(record['status'], 'NOT_FOUND')
            self.assertIn('w0p7lsw0p0c0ca0cw0p5sh1p0m1bdino.log', record['expected_log'])
            self.assertNotIn('gmargin', record['expected_log'])

    def test_missing_notmnist_does_not_create_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                result = ab.audit_notmnist(root)
            self.assertEqual(len(result), 2)
            self.assertTrue(all(not r['disk'] and not r['archive'] for r in result))
            self.assertEqual(list(root.iterdir()), [])

    def test_broken_image_archive_audit_never_repairs_or_extracts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = 'F/Q3Jvc3NvdmVyIEJvbGRPYmxpcXVlLnR0Zg==.png'
            path = root / 'notMNIST' / 'Train' / name
            path.parent.mkdir(parents=True)
            path.write_bytes(b'broken')
            archive = root / 'notMNIST.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('notMNIST/Train/' + name, b'broken')
            before = archive.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                item = ab.audit_notmnist(root)[0]
            self.assertEqual(item['disk'][0]['status'], 'UNREADABLE')
            self.assertEqual(item['disk'][0]['sha256'], item['archive'][0]['sha256'])
            self.assertEqual(path.read_bytes(), b'broken')
            self.assertEqual(archive.read_bytes(), before)

    def test_end_to_end_runs_fourteen_reuses_one_and_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(output_root=root, data_root=root, run=True, tag='test',
                                   audit_notmnist=False)
            rows = {i: dict(zip(ab.finish.METRICS, (80, 75, 90, 1, 3, -2))) for i in range(1, 11)}
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(ab.verify, 'source_digest', return_value=ab.grid.PAPER_SOURCE))
                stack.enter_context(patch.object(ab.finish, 'file_hash', return_value='hash'))
                stack.enter_context(patch.object(ab.grid, 'resolve_data', return_value=root))
                stack.enter_context(patch.object(ab.grid, 'historical_pair',
                                                return_value=({}, root/'lora', root/'tii')))
                stack.enter_context(patch.object(ab.grid, 'manifest', return_value=([], {})))
                stack.enter_context(patch.object(ab, 'baseline', return_value=(root/'baseline.log', rows)))
                stack.enter_context(patch.object(ab, 'original_driver_hash', return_value='legacy'))
                stack.enter_context(patch.object(ab.grid, 'read_verified', return_value=rows))
                run = stack.enter_context(patch.object(ab.grid, 'run_or_read', return_value=rows))
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    ab.execute(args)
            self.assertEqual(run.call_count, 14)
            self.assertIn('arms=15 reused=1 missing=14', output.getvalue())
            self.assertIn('PAPER_ABLATION_COMPLETE=15/15', output.getvalue())
            report = json.loads((root/'test__summary.json').read_text())
            self.assertEqual(len(report['results']), 15)
            self.assertEqual(len(report['contrasts']), 15)


if __name__ == '__main__':
    unittest.main()
