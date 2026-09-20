"""CPU-only tests; these do not certify a GPU RanPAC run."""
import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from tools import run_ranpac_baseline as runner


class RanPACTests(unittest.TestCase):
    def test_existing_unlocked_file_is_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / ".paper_backbone_verifier.lock"
            path.touch()
            before = path.stat().st_ino
            locker = Mock(LOCK_EX=2, LOCK_NB=4)
            stream = runner.acquire_shared_lock(path, locker)
            locker.flock.assert_called_once_with(stream, 6)
            stream.close()
            self.assertTrue(path.exists())
            self.assertEqual(path.stat().st_ino, before)

    def test_existing_lock_contents_not_truncated(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "lock"
            path.write_text("legacy metadata", encoding="utf-8")
            stream = runner.acquire_shared_lock(path, Mock(LOCK_EX=2, LOCK_NB=4))
            stream.close()
            self.assertEqual(path.read_text(encoding="utf-8"), "legacy metadata")

    def test_busy_lock_closes_handle_and_preserves_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "lock"
            locker = Mock(LOCK_EX=2, LOCK_NB=4)
            locker.flock.side_effect = BlockingIOError("held")
            with self.assertRaisesRegex(RuntimeError, "do not delete"):
                runner.acquire_shared_lock(path, locker)
            self.assertTrue(locker.flock.call_args.args[0].closed)
            self.assertTrue(path.exists())

    def test_other_lock_error_closes_handle(self):
        with tempfile.TemporaryDirectory() as folder:
            locker = Mock(LOCK_EX=2, LOCK_NB=4)
            locker.flock.side_effect = OSError("filesystem error")
            with self.assertRaises(OSError):
                runner.acquire_shared_lock(Path(folder) / "lock", locker)
            self.assertTrue(locker.flock.call_args.args[0].closed)

    def test_lock_can_be_reopened_after_release(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "lock"
            locker = Mock(LOCK_EX=2, LOCK_NB=4)
            first = runner.acquire_shared_lock(path, locker)
            first.close()
            second = runner.acquire_shared_lock(path, locker)
            self.assertFalse(second.closed)
            second.close()
            self.assertEqual(locker.flock.call_count, 2)

    @unittest.skipUnless(__import__("os").name == "posix", "Real flock requires POSIX")
    def test_real_flock_exclusion_and_release(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "lock"
            first = runner.acquire_shared_lock(path)
            try:
                with self.assertRaises(RuntimeError):
                    runner.acquire_shared_lock(path)
            finally:
                first.close()
            second = runner.acquire_shared_lock(path)
            second.close()
            self.assertTrue(path.exists())

    def test_unshuffled_order(self):
        self.assertEqual(runner.class_order(42, False), list(range(200)))

    def test_paper_python_shuffle(self):
        expected = list(range(200))
        random.Random(42).shuffle(expected)
        self.assertEqual(runner.class_order(42, True), expected)

    def test_order_preserves_global_rng(self):
        state = random.getstate()
        runner.class_order(42, True)
        self.assertEqual(state, random.getstate())

    def test_ambiguous_shuffle_rejected(self):
        with self.assertRaises(ValueError):
            runner.class_order(42, "False")

    def test_task_balanced_not_pooled(self):
        result = runner.exact_metrics([0, 0, 0, 0], [0, 0, 0, 20], 2)
        self.assertEqual(result["Acc@1"], 50)
        self.assertEqual(result["pooled_acc"], 75)
        self.assertEqual(result["task_counts"], [3, 1])

    def test_missing_task_rejected(self):
        with self.assertRaises(ValueError):
            runner.exact_metrics([0], [0], 2)

    def test_empty_predictions_rejected(self):
        with self.assertRaises(ValueError):
            runner.exact_metrics([], [], 1)

    def test_unseen_prediction_rejected(self):
        with self.assertRaises(ValueError):
            runner.exact_metrics([20], [0], 1)

    def test_retention_includes_final_excludes_last_task(self):
        rows = [{"per_task": [80]}, {"per_task": [85, 90]},
                {"per_task": [82, 91, 60]}]
        self.assertEqual(runner.retention(rows), {"Forgetting": 1.5, "Backward": 1.5})

    def test_bad_triangle_rejected(self):
        with self.assertRaises(ValueError):
            runner.retention([{"per_task": [80]}, {"per_task": [80]}])

    def test_single_task_retention_absent(self):
        self.assertEqual(runner.retention([{"per_task": [80]}]), {})

    def test_no_dataset_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(ValueError):
                runner.resolve_data(root)
            self.assertEqual(list(root.iterdir()), [])

    def test_complete_split_and_ambiguity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "imagenet-r"
            for split in ("train", "test"):
                for i in range(200):
                    (data / split / str(i)).mkdir(parents=True)
            self.assertEqual(runner.resolve_data(root), data)
            for split in ("train", "test"):
                (root / split).mkdir()
            with self.assertRaises(ValueError):
                runner.resolve_data(root)
            self.assertEqual(runner.resolve_data(root, data), data)

    def test_changed_upstream_rejected(self):
        with patch.object(runner.subprocess, "check_output",
                          side_effect=[runner.REVISION, " M RanPAC.py"]):
            with self.assertRaises(ValueError):
                runner.check_upstream(Path("dummy"))

    def test_wrong_revision_rejected(self):
        with patch.object(runner.subprocess, "check_output", side_effect=["wrong", ""]):
            with self.assertRaises(ValueError):
                runner.check_upstream(Path("dummy"))

    def test_busy_gpu_stops_before_run(self):
        with patch.object(runner.subprocess, "check_output", return_value="12345\n"):
            with self.assertRaisesRegex(RuntimeError, "no run log"):
                runner.idle_gpu_preflight()

    def test_low_gpu_memory_rejected(self):
        with patch.object(runner.subprocess, "check_output", side_effect=["", "1000"]):
            with self.assertRaisesRegex(RuntimeError, "16 GiB"):
                runner.idle_gpu_preflight()


if __name__ == "__main__":
    unittest.main()
