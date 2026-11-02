#!/usr/bin/env python3
"""Planning arithmetic only; these tests do not validate model inference."""

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "layer_plan", ROOT / "scripts/plan-layer-streaming.py"
)
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)


class LayerPlanTests(unittest.TestCase):
    def tensors(self):
        return [{"name": "token_embd.weight", "bytes": 100}] + [
            {"name": f"blk.{i}.ffn.weight", "bytes": 10 + i} for i in range(64)
        ]

    def plan(self, **kwargs):
        args = dict(ram_bytes=1000, runtime_bytes=200, read_gbs=2.0)
        args.update(kwargs)
        return planner.make_plan(self.tensors(), **args)

    def test_resident_plus_double_buffer_plus_runtime(self):
        plan = self.plan()
        self.assertEqual(plan["resident_tensor_bytes"], 100)
        self.assertEqual(plan["largest_layer_bytes"], 73)
        self.assertEqual(plan["planned_bytes"], 446)
        self.assertTrue(plan["budget_fits_arithmetically"])
        self.assertFalse(self.plan(ram_bytes=445)["budget_fits_arithmetically"])
        self.assertTrue(self.plan(ram_bytes=446)["budget_fits_arithmetically"])

    def test_decode_io_is_all_blocks_not_largest_block(self):
        plan = self.plan()
        streamed = sum(range(10, 74))
        self.assertEqual(plan["streamed_bytes_per_decode_token"], streamed)
        self.assertEqual(plan["io_only_decode_tok_s_upper_bound"], 2e9 / streamed)
        self.assertEqual(plan["tensor_bytes"], streamed + 100)

    def test_aggregates_tensors_in_a_block(self):
        tensors = self.tensors() + [{"name": "blk.63.attn.weight", "bytes": 27}]
        plan = planner.make_plan(tensors, ram_bytes=1000, runtime_bytes=0, read_gbs=1)
        self.assertEqual(plan["largest_layer_bytes"], 100)

    def test_rejects_missing_blocks_and_duplicate_names(self):
        for tensors in (self.tensors()[:-1], self.tensors() + self.tensors()[:1]):
            with self.assertRaises(ValueError):
                planner.make_plan(tensors, ram_bytes=1000, runtime_bytes=0, read_gbs=1)

    def test_rejects_invalid_budgets_or_bandwidth(self):
        for kwargs in ({"ram_bytes": 0}, {"runtime_bytes": -1}, {"read_gbs": 0},
                       {"read_gbs": float("nan")}, {"read_gbs": float("inf")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.plan(**kwargs)

    def test_reader_checks_shards_and_model_shape(self):
        def field(value):
            return SimpleNamespace(contents=lambda: value)

        def reader(shard=0, count=1, arch="qwen35"):
            return SimpleNamespace(
                fields={
                    "general.architecture": field(arch),
                    "qwen35.block_count": field(64),
                    "qwen35.embedding_length": field(5120),
                    "split.count": field(count),
                    "split.no": field(shard),
                },
                tensors=[SimpleNamespace(name=f"blk.{shard}.weight", n_bytes=128)],
            )

        fake = SimpleNamespace(GGUFReader=lambda path, mode: reader())
        with patch.dict(sys.modules, {"gguf": fake}):
            self.assertEqual(planner.inspect_ggufs([Path("a")])[0]["bytes"], 128)
            fake.GGUFReader = lambda path, mode: reader(count=2)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                planner.inspect_ggufs([Path("a")])
            with self.assertRaisesRegex(ValueError, "duplicate shard"):
                planner.inspect_ggufs([Path("a"), Path("b")])
            fake.GGUFReader = lambda path, mode: reader(arch="llama")
            with self.assertRaisesRegex(ValueError, "architecture"):
                planner.inspect_ggufs([Path("a")])
            fake.GGUFReader = lambda path, mode: reader(int(Path(path).name), 2)
            self.assertEqual(len(planner.inspect_ggufs([Path("0"), Path("1")])), 2)

    def test_pinned_remote_report_can_be_planned_without_weights(self):
        report = json.loads((ROOT / "bench/plans/qwen38-27b-q4_k_m-header.json").read_text())
        tensors = planner.tensors_from_report(report)
        plan = planner.make_plan(tensors, ram_bytes=4 * 1024**3,
                                 runtime_bytes=1024**3, read_gbs=2)
        # These are synthetic budgets and bandwidth, NOT Series X measurements.
        self.assertEqual(plan["tensor_bytes"], report["tensor_bytes"])
        self.assertEqual(plan["largest_layer_bytes"], report["largest_block_bytes"])
        self.assertEqual(plan["streamed_bytes_per_decode_token"], report["block_bytes_per_decode_token"])
        self.assertAlmostEqual(plan["io_only_decode_tok_s_upper_bound"], 0.11624696, places=6)
        report["tensor_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            planner.tensors_from_report(report)


if __name__ == "__main__":
    unittest.main()
