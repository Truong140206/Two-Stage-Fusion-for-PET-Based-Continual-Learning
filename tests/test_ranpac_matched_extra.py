"""CPU-only protocol tests for the CIFAR/ImageNet-A matched runner."""
import argparse
import json
import math
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

from tools import run_ranpac_matched_extra as runner


class MatchedExtraTests(unittest.TestCase):
    def test_exact_protocol_arguments(self):
        for name, expected_dataset in (("cifar100", "Split-CIFAR100"),
                                       ("ima", "Split-Imagenet-A")):
            commands = {p: runner.cli_arguments(name, p, Path("/out"), Path("/data"))
                        for p in runner.PHASES}
            for phase, epochs in (("tii", "20"), ("lora", "50")):
                cmd = commands[phase]
                self.assertEqual(cmd[cmd.index("--epochs") + 1], epochs)
                self.assertEqual(cmd[cmd.index("--seed") + 1], "1")
                self.assertEqual(cmd[cmd.index("--dataset") + 1], expected_dataset)
                if phase == "lora":
                    self.assertIn("--strict_exemplar_free", cmd)
            full = commands["full"]
            self.assertEqual(full[full.index("--rp_route_fusion_weight") + 1], "0.7")
            self.assertEqual(full[full.index("--rp_class_fusion_weight") + 1], "0.5")
            self.assertEqual(full[full.index("--rp_class_fusion_gate") + 1], "margin")
            self.assertNotIn("--rp_head", commands["baseline"])

    def test_arguments_parse_in_real_configs(self):
        from configs import (cifar100_hideprompt_5e, cifar100_lora,
                             ima_hideprompt_5e, ima_lora)
        modules = {
            ("cifar100", "tii"): cifar100_hideprompt_5e,
            ("cifar100", "lora"): cifar100_lora,
            ("ima", "tii"): ima_hideprompt_5e,
            ("ima", "lora"): ima_lora,
        }
        for (name, phase), module in modules.items():
            parser = argparse.ArgumentParser()
            module.get_args_parser(parser)
            args = parser.parse_args(
                runner.cli_arguments(name, phase, Path("/out"), Path("/data"))[1:])
            self.assertEqual(args.seed, 1)
            self.assertEqual(args.num_tasks, 10)

    def test_fixed_order_split_for_both_class_counts(self):
        import ast
        source = ast.parse((runner.ROOT / "datasets.py").read_text(encoding="utf-8"))
        fn = next(n for n in source.body
                  if isinstance(n, ast.FunctionDef) and n.name == "split_single_dataset")
        module = SimpleNamespace(random=random, math=math,
                                 Subset=lambda ds, indexes: indexes)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "datasets.py", "exec"),
             module.__dict__)
        for classes in (100, 200):
            order = list(reversed(range(classes)))
            ds = SimpleNamespace(classes=list(range(classes)), targets=list(range(classes)) * 2)
            args = SimpleNamespace(num_tasks=10, shuffle=False)
            rng_before = random.getstate()
            _, masks, mapping = runner.ordered_split(
                module, module.split_single_dataset, order, classes)(ds, ds, args)
            self.assertEqual(masks[0], order[:classes // 10])
            self.assertEqual(mapping[classes - 1], 0)
            self.assertEqual(random.getstate(), rng_before)
            self.assertFalse(args.shuffle)

    def test_source_metadata_rejects_wrong_training_seed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root / "_ranpac_support/original/preflight-extra-cifar100/metadata.json"
            path.parent.mkdir(parents=True)
            metadata = {"dataset": "cifar100", "revision": runner.common.REVISION,
                        "torch_seed_from_original_trainer": 2,
                        "class_order": list(range(100)),
                        "counts": {"train": [1] * 100, "test": [1] * 100}}
            path.write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                runner.source_metadata(root, "cifar100")

    def test_backbone_matches_each_native_ranpac_dataset(self):
        digest = "a" * 64
        for name in ("cifar100", "ima"):
            filename, actual_digest = runner.native_backbone(
                name, {"pretrained_sha256": digest})
            self.assertEqual(filename, runner.extra.SPECS[name]["checkpoint_file"])
            self.assertEqual(actual_digest, digest)
        self.assertNotEqual(runner.extra.SPECS["cifar100"]["checkpoint_file"],
                            runner.extra.SPECS["ima"]["checkpoint_file"])

    def test_backbone_rejects_missing_digest(self):
        with self.assertRaises(ValueError):
            runner.native_backbone("cifar100", {})


if __name__ == "__main__":
    unittest.main()
