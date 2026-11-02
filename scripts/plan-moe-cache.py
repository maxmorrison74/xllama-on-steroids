#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
#
# Built on top of Xllama by Gianluca Mazza (https://github.com/gianlucamazza/xllama)
"""Cost an expert cache from a captured GGUF header; no runtime performance claims."""

import argparse
import json
import math
from pathlib import Path

MIB = 1024 * 1024


def make_plan(report, *, ram_mib, runtime_mib, slots, read_gbs, assumed_byte_hit_rate):
    moe = report["routed_experts"]
    experts = moe["experts_per_layer"]
    active = moe["active_experts_per_token_per_layer"]
    if not all(isinstance(v, int) and not isinstance(v, bool)
               for v in (ram_mib, runtime_mib, slots, experts, active)):
        raise ValueError("budgets, slots and expert counts must be integers")
    if ram_mib <= 0 or runtime_mib < 0 or not 0 < active <= slots <= experts:
        raise ValueError("invalid budget or slot count; need active <= slots <= experts")
    if not math.isfinite(read_gbs) or read_gbs <= 0:
        raise ValueError("read GB/s must be finite and positive")
    if not math.isfinite(assumed_byte_hit_rate) or not 0 <= assumed_byte_hit_rate <= 1:
        raise ValueError("assumed byte-weighted hit rate must be in [0,1]")
    strides = [layer["proposed_expert_stride_bytes"] for layer in moe["layers"]]
    if not strides or any(not isinstance(size, int) or size <= 0 or size % 4096 for size in strides):
        raise ValueError("invalid page-aligned expert strides")
    per_slot = sum(strides)
    if per_slot != moe["proposed_bytes_per_slot_across_all_layers"]:
        raise ValueError("inconsistent slot byte totals")
    common = moe["common_tensor_bytes"]
    if not isinstance(common, int) or common <= 0:
        raise ValueError("invalid common tensor bytes")
    reads = per_slot * active
    if reads != moe["proposed_no_cache_expert_read_bytes_per_decode_token"]:
        raise ValueError("inconsistent expert read byte totals")
    cache = per_slot * slots
    total = common + cache + runtime_mib * MIB
    missed = reads * (1 - assumed_byte_hit_rate)
    return {
        "status": "scenario_only_no_uwp_moe_engine_or_measured_xbox_cache_hit_rate",
        "source": report.get("source"),
        "header_sha256": report.get("header_sha256"),
        "common_tensor_bytes": common,
        "slots_per_layer": slots,
        "expert_cache_bytes": cache,
        "runtime_reserve_bytes_assumed": runtime_mib * MIB,
        "planning_cap_bytes": ram_mib * MIB,
        "planned_bytes": total,
        "budget_fits_arithmetically": total <= ram_mib * MIB,
        "expert_read_bytes_per_token_without_cache": reads,
        "assumed_byte_weighted_hit_rate": assumed_byte_hit_rate,
        "scenario_missed_expert_bytes_per_token": missed,
        "assumed_effective_read_gbs": read_gbs,
        "io_service_seconds_per_token_at_assumptions": missed / (read_gbs * 1e9),
        "limitations": [
            "runtime reserve is an assumption, not measured model memory",
            "RAM cap must not be confused with the headless allocation envelope",
            "cache hit rate is hypothetical, not borrowed from another machine",
            "probe rate may not transfer to packed expert reads",
            "excludes compute, scheduling, cache planning, copies and prefill",
            "not an end-to-end token rate or a correctness/admission verdict",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata_report", type=Path)
    parser.add_argument("--ram-mib", type=int, required=True)
    parser.add_argument("--runtime-mib", type=int, required=True)
    parser.add_argument("--slots", type=int, required=True)
    parser.add_argument("--read-gbs", type=float, required=True)
    parser.add_argument("--assumed-byte-hit-rate", type=float, required=True)
    args = parser.parse_args()
    try:
        plan = make_plan(json.loads(args.metadata_report.read_text()), ram_mib=args.ram_mib,
                         runtime_mib=args.runtime_mib, slots=args.slots, read_gbs=args.read_gbs,
                         assumed_byte_hit_rate=args.assumed_byte_hit_rate)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(plan, indent=2, allow_nan=False))
    return 0 if plan["budget_fits_arithmetically"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
