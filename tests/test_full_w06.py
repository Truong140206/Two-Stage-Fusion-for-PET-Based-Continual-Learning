"""CPU-only protocol tests; no claim of CUDA evaluation."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from tools import check_full_w06 as w06


class FullW06Tests(unittest.TestCase):
    def test_only_weight_changes_all_twelve_configs(self):
        for ds in w06.verify.DATASETS:
            for seed in w06.SEEDS:
                base = SimpleNamespace(dataset=ds, resolved_data_path=Path('/data'),
                                       batch_size=24, port=29558)
                cmd = w06.command(base, seed, Path('/tii'), Path('/lora'))
                cfg = w06.grid.parsed_config(cmd)
                self.assertTrue(cfg.eval and cfg.rp_pin_extractor and cfg.strict_exemplar_free)
                self.assertEqual((cfg.seed, cfg.rp_route_fusion_weight,
                                  cfg.rp_class_fusion_weight, cfg.rp_class_fusion_gate),
                                 (seed, .6, .5, 'margin'))
                w06.finish.set_flag(cmd, '--rp_route_fusion_weight', '0.7')
                self.assertEqual(cmd, w06.verify.command(base, seed, 'full',
                                                         Path('/tii'), Path('/lora')))

    def test_tags(self):
        self.assertEqual(w06.reference_tag('cub200', 43), 'paperfix_verify_v3')
        self.assertEqual(w06.reference_tag('cub200', 42), 'maskfix_verify_v2')
        self.assertEqual(w06.reference_tag('imr', 45), 'maskfix_verify_v2')

    def results(self):
        return {d: {s: {arm: dict.fromkeys(w06.finish.METRICS, float(s)+delta)
                        for arm, delta in (('baseline', 0), ('w07', 1), ('w06', 1.2))}
                    for s in w06.SEEDS} for d in w06.verify.DATASETS}

    def test_paired_not_unpaired_interval(self):
        result = w06.summarise(self.results())['imr']
        contrast = result['contrasts']['w06_minus_w07']['Acc@1']
        self.assertAlmostEqual(contrast['mean'], .2)
        self.assertAlmostEqual(contrast['sd'], 0)
        self.assertAlmostEqual(contrast['ci95_low'], .2)
        self.assertGreater(result['arms']['w06']['Acc@1']['sd'], 1)

    def test_missing_seed_rejected(self):
        result = self.results()
        del result['imr'][45]
        with self.assertRaises(ValueError):
            w06.summarise(result)

    def test_summary_reuse_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'summary.json'
            report = w06.summarise(self.results())
            w06.save_summary(p, report)
            original = p.read_bytes()
            w06.save_summary(p, report)
            self.assertEqual(p.read_bytes(), original)
            with self.assertRaises(ValueError):
                w06.save_summary(p, {'different': True})
            self.assertEqual(json.loads(original), json.loads(json.dumps(report)))

    def exercise(self, run=False, missing_reference=False, disk_free=2*1024**3,
                 existing=False):
        args = SimpleNamespace(data_root=Path('/data'), output_root=Path('/out'),
                               tag='full_w06_main_v1', run=run)
        rows = {i: dict.fromkeys(w06.finish.METRICS, float(i)) for i in range(1, 11)}
        def read(path, meta, tasks):
            if missing_reference:
                raise ValueError('bad original provenance')
            if '__soict_final_v1__w06' in str(path):
                self.assertEqual(meta['driver_sha256'], 'legacy')
            return rows
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(w06.verify, 'source_digest', return_value=w06.grid.PAPER_SOURCE))
            stack.enter_context(patch.object(w06.finish, 'file_hash', return_value='current'))
            stack.enter_context(patch.object(w06, 'old_w06_hash', return_value='legacy'))
            stack.enter_context(patch.object(w06.verify, 'resolve_imr_data_path', return_value=Path('/data')))
            stack.enter_context(patch.object(w06.grid, 'manifest', return_value=([], {})))
            reader = stack.enter_context(patch.object(w06.grid, 'read_verified', side_effect=read))
            runner = stack.enter_context(patch.object(w06.grid, 'run_or_read', return_value=rows))
            stack.enter_context(patch.object(w06.grid, 'assert_unchanged'))
            stack.enter_context(patch.object(Path, 'exists', return_value=existing))
            stack.enter_context(patch.object(w06.shutil, 'disk_usage',
                                            return_value=SimpleNamespace(free=disk_free)))
            save = stack.enter_context(patch.object(w06, 'save_summary'))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            w06.execute(args)
        return reader, runner, save

    def test_preflight_no_gpu_no_summary(self):
        reader, runner, save = self.exercise()
        self.assertEqual(reader.call_count, 28)
        runner.assert_not_called()
        save.assert_not_called()

    def test_runs_only_eight_missing_not_imr(self):
        _, runner, save = self.exercise(run=True)
        self.assertEqual(runner.call_count, 8)
        save.assert_called_once()
        for c in runner.call_args_list:
            self.assertNotIn('imr_lora', str(c.args[0]))
            self.assertTrue(c.args[4])
            self.assertIn('__full_w06_main_v1__w06.log', str(c.args[0]))

    def test_completed_batch_reused_without_gpu(self):
        _, runner, save = self.exercise(existing=True)
        runner.assert_not_called()
        save.assert_called_once()

    def test_missing_reference_stops_before_launch(self):
        with self.assertRaisesRegex(ValueError, 'provenance'):
            self.exercise(run=True, missing_reference=True)

    def test_low_disk_stops(self):
        with self.assertRaisesRegex(RuntimeError, 'Disk'):
            self.exercise(run=True, disk_free=1024)


if __name__ == '__main__':
    unittest.main()
