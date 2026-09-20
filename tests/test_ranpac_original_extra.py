"""CPU-only safety and metric tests for the two-dataset RanPAC launcher."""
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools import run_ranpac_original_extra as extra


class OriginalExtraTests(unittest.TestCase):
    def test_scope_excludes_five_datasets(self):
        self.assertEqual(set(extra.SPECS), {"cifar100", "ima"})
        self.assertEqual(extra.SPECS["cifar100"]["increment"], 10)
        self.assertEqual(extra.SPECS["ima"]["increment"], 20)

    def test_official_configs_are_dataset_specific(self):
        self.assertEqual(extra.SPECS["cifar100"]["expected"]["model_name"], "adapter")
        self.assertIn("in21k_adapter", extra.SPECS["cifar100"]["expected"]["convnet_type"])
        self.assertEqual(extra.SPECS["ima"]["expected"]["model_name"], "ssf")
        self.assertTrue(extra.SPECS["ima"]["expected"]["convnet_type"].endswith("_ssf"))

    def write_config(self, root, name, **changes):
        spec = extra.SPECS[name]
        (root / "args").mkdir(exist_ok=True)
        row = dict(spec["expected"], **changes)
        with (root / "args" / spec["csv"]).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=spec["expected"])
            writer.writeheader()
            writer.writerow(row)

    def test_exact_official_config_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.write_config(root, "cifar100")
            self.assertEqual(extra.config_row(root, "cifar100"),
                             extra.SPECS["cifar100"]["expected"])

    def test_changed_official_config_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.write_config(root, "ima", seed="42")
            with self.assertRaisesRegex(ValueError, "differs"):
                extra.config_row(root, "ima")

    def test_pretrained_source_handles_adapter_without_default_cfg(self):
        url = ("https://storage.googleapis.com/vit_models/augreg/" +
               extra.SPECS["cifar100"]["checkpoint_file"])
        timm = SimpleNamespace(models=SimpleNamespace(
            vision_transformer=SimpleNamespace(default_cfgs={
                "vit_base_patch16_224_in21k": {"url": url}})))
        self.assertEqual(extra.pretrained_source("cifar100", object(), timm), url)

    def test_pretrained_source_uses_ssf_model_metadata(self):
        url = ("https://storage.googleapis.com/vit_models/augreg/" +
               extra.SPECS["ima"]["checkpoint_file"])
        model = SimpleNamespace(pretrained_cfg={"url": url})
        self.assertEqual(extra.pretrained_source("ima", model, Mock()), url)

    def test_wrong_pretrained_source_rejected(self):
        model = SimpleNamespace(pretrained_cfg={"url": "https://example.invalid/model.npz"})
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            extra.pretrained_source("ima", model, Mock())

    def test_generic_metrics_use_dataset_increment(self):
        cifar = extra.exact_metrics(list(range(20)), list(range(20)), 2, 10)
        imagenet = extra.exact_metrics(list(range(40)), list(range(40)), 2, 20)
        self.assertEqual(cifar["task_counts"], [10, 10])
        self.assertEqual(imagenet["task_counts"], [20, 20])
        self.assertEqual(cifar["Acc@1"], 100)
        with self.assertRaisesRegex(ValueError, "outside"):
            extra.exact_metrics([20], [0], 2, 10)

    def test_extra_dependency_installed_in_private_environment(self):
        with tempfile.TemporaryDirectory() as d:
            support = Path(d)
            python = support / "env-py39/bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
            missing = Mock(returncode=1)
            with patch.object(extra.subprocess, "run",
                              side_effect=[missing, Mock(returncode=0), Mock(returncode=0),
                                           Mock(returncode=0)]) as run:
                extra.ensure_extra_environment(support, {}, python, install=True)
            install = run.call_args_list[1].args[0]
            self.assertIn("easydict==1.13", install)
            self.assertEqual(json.loads(
                (support / "environment-extra-ready.json").read_text()),
                {"easydict": "1.13"})

    def test_missing_extra_dependency_requires_prepare(self):
        with tempfile.TemporaryDirectory() as d:
            support = Path(d)
            python = support / "python"
            python.touch()
            with patch.object(extra.subprocess, "run", return_value=Mock(returncode=1)):
                with self.assertRaisesRegex(ValueError, "--prepare"):
                    extra.ensure_extra_environment(support, {}, python, install=False)

    def make_results(self, root, name):
        spec = extra.SPECS[name]
        result_dir = root / "results"
        pred_dir = result_dir / "class_preds"
        pred_dir.mkdir(parents=True)
        fields = [f"{kind}_task_{stage}" for stage in range(10)
                  for kind in ("pred", "true")]
        total = spec["classes"]
        with (pred_dir / f"{spec['official']}_class_preds_publish_7.csv").open(
                "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for target in range(total):
                row = {}
                for stage in range(10):
                    value = target if target < (stage + 1) * spec["increment"] else -1
                    row[f"pred_task_{stage}"] = value
                    row[f"true_task_{stage}"] = value
                writer.writerow(row)
        with (result_dir / f"{spec['official']}_publish_7.csv").open(
                "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["top1_total", "ave_acc"])
            writer.writeheader()
            writer.writerows([{"top1_total": 100, "ave_acc": 100}] * 10)
        return {
            "dataset": name,
            "counts": {"train": [1] * total, "test": [1] * total},
            "class_order": list(range(total)),
        }

    def test_summarize_perfect_cifar_and_imageneta(self):
        for name in extra.SPECS:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                metadata = self.make_results(root, name)
                result = extra.summarize(root, name, metadata)
                self.assertEqual(result["final"]["Acc@1"], 100)
                self.assertEqual(result["final"]["Forgetting"], 0)
                self.assertEqual(result["final"]["Backward"], 0)

    def test_partial_predictions_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            metadata = self.make_results(root, "cifar100")
            spec = extra.SPECS["cifar100"]
            path = root / "results/class_preds" / (
                spec["official"] + "_class_preds_publish_7.csv")
            path.write_text("pred_task_0,true_task_0\n1,1\n")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                extra.summarize(root, "cifar100", metadata)


if __name__ == "__main__":
    unittest.main()
