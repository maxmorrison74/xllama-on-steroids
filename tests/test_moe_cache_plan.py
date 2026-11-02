#!/usr/bin/env python3
"""Header/layout arithmetic tests; no model execution or cache performance claims."""

import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


planner = load("moe_plan", "plan-moe-cache.py")
inspector = load("moe_metadata", "inspect-remote-gguf.py")


class ExpertLayoutTests(unittest.TestCase):
    def metadata(self):
        return {"general.architecture": "qwen3moe", "qwen3moe.expert_count": 16,
                "qwen3moe.expert_used_count": 2, "qwen3moe.block_count": 1}

    def tensors(self):
        return [{"name": f"blk.0.ffn_{kind}_exps.weight", "shape": [256, 2, 16],
                 "bytes": 4608, "type": 12, "offset": i * 4608}
                for i, kind in enumerate(("gate", "up", "down"))] + [
                    {"name": "output_norm.weight", "shape": [25], "bytes": 100,
                     "type": 0, "offset": 13824}]

    def test_payload_and_proposed_packed_alignment_are_distinct(self):
        plan = inspector.routed_expert_layout(self.tensors(), self.metadata(), 4096)
        self.assertEqual(plan["common_tensor_bytes"], 100)
        self.assertEqual(plan["layers"][0]["expert_payload_bytes"], 864)
        self.assertEqual(plan["proposed_bytes_per_slot_across_all_layers"], 3 * 4096)
        self.assertEqual(plan["proposed_no_cache_expert_read_bytes_per_decode_token"], 6 * 4096)
        self.assertEqual(plan["layers"][0]["projections"][1]["source_tensor_file_offset"], 4096 + 4608)

    def test_rejects_incomplete_or_inconsistent_expert_layout(self):
        tensors = self.tensors()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            inspector.routed_expert_layout(tensors[1:], self.metadata(), 0)
        tensors[0]["shape"][2] = 15
        with self.assertRaisesRegex(ValueError, "third tensor axis"):
            inspector.routed_expert_layout(tensors, self.metadata(), 0)

    def test_bounds_metadata_and_leaves_dense_models_alone(self):
        self.assertIsNone(inspector.routed_expert_layout([], {"general.architecture": "qwen35"}, 0))
        for value in (0, -1, True, 10**20):
            metadata = self.metadata()
            metadata["qwen3moe.block_count"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                inspector.routed_expert_layout(self.tensors(), metadata, 0)


class CachePlanTests(unittest.TestCase):
    def report(self):
        return json.loads((ROOT / "bench/plans/qwen3-30b-a3b-instruct-2507-q4_k_m-header.json").read_text())

    def plan(self, **kwargs):
        params = dict(ram_mib=4096, runtime_mib=768, slots=16, read_gbs=1.57,
                      assumed_byte_hit_rate=0.5)
        params.update(kwargs)
        return planner.make_plan(self.report(), **params)

    def test_pinned_geometry_and_explicit_budget_fit(self):
        # Layout arithmetic, not measured Xbox model allocations.
        plan = self.plan()
        self.assertEqual(plan["common_tensor_bytes"], 1073051648)
        self.assertEqual(plan["expert_cache_bytes"], 16 * 137134080)
        self.assertEqual(plan["expert_read_bytes_per_token_without_cache"], 1097072640)
        self.assertTrue(plan["budget_fits_arithmetically"])
        self.assertFalse(self.plan(slots=20)["budget_fits_arithmetically"])

    def test_hit_rate_is_an_explicit_byte_weighted_scenario(self):
        miss = self.plan(assumed_byte_hit_rate=0)
        half = self.plan(assumed_byte_hit_rate=0.5)
        all_hit = self.plan(assumed_byte_hit_rate=1)
        self.assertEqual(half["scenario_missed_expert_bytes_per_token"], 0.5 * miss["scenario_missed_expert_bytes_per_token"])
        self.assertEqual(all_hit["io_service_seconds_per_token_at_assumptions"], 0)
        self.assertTrue(all_hit["limitations"])

    def test_rejects_invalid_or_unbounded_inputs(self):
        for kwargs in ({"slots": 7}, {"slots": 129}, {"slots": True}, {"ram_mib": 0},
                       {"runtime_mib": -1}, {"read_gbs": float("inf")},
                       {"assumed_byte_hit_rate": float("nan")}, {"assumed_byte_hit_rate": 1.01}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.plan(**kwargs)

    def test_rejects_corrupt_header_report_totals(self):
        report = self.report()
        for key in ("proposed_bytes_per_slot_across_all_layers",
                    "proposed_no_cache_expert_read_bytes_per_decode_token"):
            damaged = copy.deepcopy(report)
            damaged["routed_experts"][key] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                planner.make_plan(damaged, ram_mib=4096, runtime_mib=768, slots=16,
                                  read_gbs=1.57, assumed_byte_hit_rate=0)


if __name__ == "__main__":
    unittest.main()
