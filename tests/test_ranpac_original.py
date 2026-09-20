"""Dependency-free safety/entry-point/metric tests, not a GPU reproduction."""
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools import run_ranpac_original as native


class OriginalTests(unittest.TestCase):
    def test_official_config_exact(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "args").mkdir()
            with (root / "args/imagenetr_publish.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=native.EXPECTED)
                writer.writeheader()
                writer.writerow(native.EXPECTED)
            self.assertEqual(native.config_row(root), native.EXPECTED)

    def test_changed_seed_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "args").mkdir()
            with (root / "args/imagenetr_publish.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=native.EXPECTED)
                writer.writeheader()
                writer.writerow(dict(native.EXPECTED, seed="42"))
            with self.assertRaises(ValueError):
                native.config_row(root)

    def test_environment_isolation(self):
        with patch.dict(os.environ, {"PYTHONPATH": "bad", "PYTHONHOME": "bad",
                                      "VIRTUAL_ENV": "old", "LD_LIBRARY_PATH": "old"}):
            env = native.isolated_env(Path("/private"), 4)
        for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "LD_LIBRARY_PATH"):
            self.assertNotIn(key, env)
        self.assertEqual(env["OMP_NUM_THREADS"], "4")
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")
        self.assertIn("torch-cache", env["TORCH_HOME"])

    def test_existing_directory_never_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            folder = root / "occupied"
            folder.mkdir()
            with self.assertRaisesRegex(ValueError, "preserved"):
                native.link_existing(folder, root)
            self.assertTrue(folder.is_dir())

    def test_existing_matching_link_reused(self):
        link = Mock()
        link.is_symlink.return_value = True
        link.resolve.return_value = Path("/target").resolve()
        native.link_existing(link, Path("/target"))
        link.symlink_to.assert_not_called()

    def test_wrong_link_rejected(self):
        link = Mock()
        link.is_symlink.return_value = True
        link.resolve.return_value = Path("/elsewhere")
        with self.assertRaises(ValueError):
            native.link_existing(link, Path("/target"))
        link.symlink_to.assert_not_called()

    def test_checksum_prevents_extraction_execution(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "micromamba-2.3.2-0.tar.bz2").write_bytes(b"wrong")
            with patch.object(native.tarfile, "open") as extract:
                with self.assertRaisesRegex(ValueError, "checksum"):
                    native.download_mamba(root)
                extract.assert_not_called()

    def test_existing_environment_does_not_reinstall(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "env-py39/bin").mkdir(parents=True)
            (root / "env-py39/bin/python").touch()
            (root / "environment-ready.json").write_text("{}")
            with patch.object(native.subprocess, "run") as run:
                self.assertEqual(native.prepare(root, {}), root / "env-py39/bin/python")
                run.assert_not_called()

    def test_native_calls_original_main_with_original_arguments(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            upstream = root / "_ranpac_support/upstream"
            upstream.mkdir(parents=True)
            marker = root / "entry.json"
            (upstream / "main.py").write_text(
                "import json, sys\nfrom pathlib import Path\n"
                "Path(" + repr(str(marker)) + ").write_text(json.dumps(sys.argv))\n")
            original_path, original_argv = sys.path[:], sys.argv[:]
            try:
                with patch.object(native.common, "check_upstream"):
                    native.child(SimpleNamespace(output_root=root, child="native"))
            finally:
                sys.path[:] = original_path
                sys.argv[:] = original_argv
            argv = json.loads(marker.read_text())
            self.assertEqual(argv[1:], ["-i", "7", "-d", "imagenetr"])

    def fixtures(self, root):
        folder = root / "results/class_preds"
        folder.mkdir(parents=True)
        path = folder / "imagenetr_class_preds_publish_7.csv"
        fields = [f"{k}_task_{i}" for i in range(10) for k in ("pred", "true")]
        # Official padding: stage s contains (s+1)*600 predictions.
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for j in range(6000):
                row = {}
                for i in range(10):
                    target = j // 30 if j < (i + 1) * 600 else -1
                    row[f"pred_task_{i}"] = target
                    row[f"true_task_{i}"] = target
                writer.writerow(row)
        with (root / "results/imagenetr_publish_7.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["top1_total", "ave_acc"])
            writer.writeheader()
            writer.writerows([{"top1_total": 100, "ave_acc": 100}] * 10)
        return path

    def test_exact_metrics_from_native_saved_predictions(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixtures(root)
            result = native.summarize(root)
            self.assertEqual(len(result["stages"]), 10)
            self.assertEqual(result["final"]["Acc@1"], 100)
            self.assertEqual(result["final"]["Forgetting"], 0)
            self.assertEqual(result["final"]["Backward"], 0)

    def test_partial_results_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = self.fixtures(root)
            path.write_text("pred_task_0,true_task_0\n1,1\n")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                native.summarize(root)

    def test_official_curve_disagreement_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixtures(root)
            (root / "results/imagenetr_publish_7.csv").write_text(
                "top1_total,ave_acc\n" + "99,99\n" * 10)
            with self.assertRaisesRegex(ValueError, "disagree"):
                native.summarize(root)

    def test_subprocess_lock_inherited_and_exit_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            process = Mock()
            process.wait.return_value = 0
            process.poll.return_value = 0
            lock = Mock()
            lock.fileno.return_value = 13
            with patch.object(native.subprocess, "Popen", return_value=process) as popen:
                native.run_bounded(["python", "original"], root, {}, lock, root / "run.log", 60)
            self.assertEqual(popen.call_args.kwargs["pass_fds"], (13,))
            self.assertIn("EXIT_CODE=0", (root / "run.log").read_text())

    def test_existing_log_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "run.log"
            path.write_text("keep")
            with patch.object(native.subprocess, "Popen") as popen:
                with self.assertRaises(FileExistsError):
                    native.run_bounded([], Path(d), {}, Mock(), path, 60)
                popen.assert_not_called()
            self.assertEqual(path.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
