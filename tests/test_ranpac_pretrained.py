"""CPU-only protocol, argument and preservation tests; no GPU claim."""
import ast
import json
import math
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools import try_ranpac_pretrained as trial


class TrialTests(unittest.TestCase):
    def test_fixed_training_budget_and_seed(self):
        for phase, epochs, batch in (("tii", "20", "128"), ("lora", "50", "24")):
            cmd = trial.cli_arguments(phase, Path("/trial"), Path("/data"))
            self.assertEqual(cmd[cmd.index("--epochs") + 1], epochs)
            self.assertEqual(cmd[cmd.index("--batch-size") + 1], batch)
            self.assertEqual(cmd[cmd.index("--seed") + 1], "1")
            self.assertIn("--strict_exemplar_free", cmd)
            self.assertNotIn("--rp_head", cmd)
            self.assertNotIn("--eval", cmd)

    def test_eval_same_checkpoint_and_fixed_fusion(self):
        commands = {p: trial.cli_arguments(p, Path("/trial"), Path("/data"))
                    for p in ("baseline", "full")}
        for flag in ("--output_dir", "--trained_original_model", "--model", "--seed"):
            self.assertEqual(*(c[c.index(flag) + 1] for c in commands.values()))
        self.assertNotIn("--rp_head", commands["baseline"])
        full = commands["full"]
        for flag, value in (("--rp_route_fusion_weight", "0.7"),
                            ("--rp_class_fusion_weight", "0.5"),
                            ("--rp_class_fusion_gate", "margin"),
                            ("--rp_dim", "10000"), ("--rp_lambda", "10000")):
            self.assertEqual(full[full.index(flag) + 1], value)

    def test_arguments_parse_in_existing_configs(self):
        import argparse
        from configs import imr_lora, imr_hideprompt_5e
        for phase in trial.PHASES:
            parser = argparse.ArgumentParser()
            (imr_hideprompt_5e if phase == "tii" else imr_lora).get_args_parser(parser)
            args = parser.parse_args(trial.cli_arguments(phase, Path("/trial"), Path("/data"))[1:])
            self.assertEqual(args.num_tasks, 10)
            self.assertFalse(getattr(args, "crct_real_feature_replay", False))

    def test_actual_split_function_with_explicit_order(self):
        # Compile the real split function without importing torch/torchvision.
        source = ast.parse((trial.ROOT / "datasets.py").read_text(encoding="utf-8"))
        fn = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "split_single_dataset")
        module = SimpleNamespace(random=random, math=math,
                                 Subset=lambda ds, indexes: indexes)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "datasets.py", "exec"), module.__dict__)
        original = module.split_single_dataset
        order = list(reversed(range(200)))
        ds = SimpleNamespace(classes=list(range(200)), targets=list(range(200)) * 2)
        args = SimpleNamespace(num_tasks=10, shuffle=True)
        rng_before = random.getstate()
        splits, mask, mapping = trial.ordered_split(module, original, order)(ds, ds, args)
        self.assertEqual(mask[0], order[:20])
        self.assertEqual(mapping[199], 0)
        self.assertEqual(mapping[0], 9)
        self.assertEqual([len(s[0]) for s in splits], [40] * 10)
        self.assertEqual(random.getstate(), rng_before)
        self.assertIs(module.random, random)
        self.assertTrue(args.shuffle)

    def test_split_restores_state_on_failure(self):
        module = SimpleNamespace(random=random)
        args = SimpleNamespace(shuffle=False)
        def fail(*pos):
            raise RuntimeError("stop")
        with self.assertRaises(RuntimeError):
            trial.ordered_split(module, fail, list(range(200)))(None, None, args)
        self.assertIs(module.random, random)
        self.assertFalse(args.shuffle)

    def test_native_summary_wrong_pretrained_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "summary.json"
            data = {"metadata": {"config": trial.native.EXPECTED,
                                 "revision": trial.common.REVISION,
                                 "class_order": list(range(200)), "pretrained_sha256": "wrong"},
                    "stages": [{}] * 10, "final": {"stage": 10}}
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                trial.checked_summary(path)

    def test_checkpoint_old_path_and_existing_file_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / "new"
            folder.mkdir()
            path = folder / "existing.pth"
            path.write_bytes(b"keep")
            save = Mock()
            wrapped = trial.atomic_checkpoint(save, folder)
            for target in (path, Path(d) / "old.pth"):
                with self.assertRaises(ValueError):
                    wrapped({}, target)
            save.assert_not_called()
            self.assertEqual(path.read_bytes(), b"keep")

    def test_atomic_save_success(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            path = folder / "new.pth"
            def save(state, target):
                self.assertEqual(target.suffix, ".partial")
                self.assertFalse(path.exists())
                target.write_bytes(b"complete")
            with patch.object(trial.shutil, "disk_usage", return_value=SimpleNamespace(free=20 * 1024**3)):
                trial.atomic_checkpoint(save, folder)({}, path)
            self.assertEqual(path.read_bytes(), b"complete")
            self.assertFalse(path.with_suffix(".pth.partial").exists())

    def test_failed_save_keeps_partial_not_final(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            path = folder / "new.pth"
            def save(state, target):
                target.write_bytes(b"partial")
                raise RuntimeError("interrupted")
            with patch.object(trial.shutil, "disk_usage", return_value=SimpleNamespace(free=20 * 1024**3)):
                with self.assertRaises(RuntimeError):
                    trial.atomic_checkpoint(save, folder)({}, path)
            self.assertFalse(path.exists())
            self.assertTrue(path.with_suffix(".pth.partial").exists())

    def test_disk_guard_before_save(self):
        with tempfile.TemporaryDirectory() as d:
            save = Mock()
            with patch.object(trial.shutil, "disk_usage", return_value=SimpleNamespace(free=1024)):
                with self.assertRaises(RuntimeError):
                    trial.atomic_checkpoint(save, Path(d))({}, Path(d) / "new.pth")
            save.assert_not_called()

    def test_resume_requires_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            log = folder / "tii.log"
            log.write_text("complete")
            trial.save_json_exclusive(folder / "tii_done.json",
                                      {"files": {"tii.log": trial.common.sha(log)}})
            self.assertTrue(trial.validate_done(folder, "tii"))
            log.write_text("changed")
            with self.assertRaises(ValueError):
                trial.validate_done(folder, "tii")

    def test_no_done_means_no_resume(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(trial.validate_done(Path(d), "tii"))

    def test_no_overwrite_summary(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "summary.json"
            trial.save_json_exclusive(path, {"a": 1})
            with self.assertRaises(FileExistsError):
                trial.save_json_exclusive(path, {"a": 2})


if __name__ == "__main__":
    unittest.main()
