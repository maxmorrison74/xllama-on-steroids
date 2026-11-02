#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Inspect a text-only Qwen3.8-27B GGUF; do not load weights or run inference.

Host-only: uses gguf-py's read-only mapping to inspect tensor descriptors.
Pass every shard of a split GGUF. Budgets are caller-provided, not Xbox specs.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIB = 1024 * 1024


def make_plan(tensors, *, ram_bytes, runtime_bytes, read_gbs):
    """Plan whole-layer double buffering, keeping non-block tensors resident.

    This deliberately conservative policy keeps full embeddings/output resident.
    runtime_bytes must separately cover KV/recurrent state, compute, UI and other
    allocations. A successful arithmetic fit is NOT runtime admission.
    """
    if ram_bytes <= 0 or runtime_bytes < 0:
        raise ValueError("RAM must be positive and runtime reserve non-negative")
    if not math.isfinite(read_gbs) or read_gbs <= 0:
        raise ValueError("read GB/s must be finite and positive")
    layers = {}
    resident = 0
    names = set()
    for tensor in tensors:
        name, size = tensor["name"], tensor["bytes"]
        if name in names:
            raise ValueError(f"duplicate tensor: {name}; pass each shard once")
        names.add(name)
        if size <= 0:
            raise ValueError(f"invalid tensor size: {name}")
        match = re.match(r"^blk\.(\d+)\.", name)
        if match:
            layer = int(match[1])
            layers[layer] = layers.get(layer, 0) + size
        else:
            resident += size
    if sorted(layers) != list(range(64)):
        raise ValueError("expected all 64 decoder blocks for Qwen3.8-27B")
    largest = max(layers.values())
    streamed = sum(layers.values())
    required = resident + 2 * largest + runtime_bytes
    return {
        "model": "Qwen/Qwen3.8-27B",
        "status": "planning_only_no_uwp_streaming_engine",
        "policy": "resident_non_block_tensors_double_buffered_decoder_layers",
        "tensor_bytes": resident + streamed,
        "resident_tensor_bytes": resident,
        "streamed_bytes_per_decode_token": streamed,
        "largest_layer_bytes": largest,
        "double_buffer_bytes": 2 * largest,
        "runtime_reserve_bytes": runtime_bytes,
        "ram_budget_bytes": ram_bytes,
        "planned_bytes": required,
        "budget_fits_arithmetically": required <= ram_bytes,
        "assumed_read_gbs": read_gbs,
        "io_only_decode_tok_s_upper_bound": read_gbs * 1e9 / streamed,
        "layers": [{"index": i, "bytes": layers[i]} for i in sorted(layers)],
    }


def inspect_ggufs(paths):
    sys.path.insert(0, str(ROOT / "llama.cpp" / "gguf-py"))
    try:
        from gguf import GGUFReader
    except ImportError as exc:
        raise ValueError(
            "initialize llama.cpp and install its gguf-py dependencies on the host"
        ) from exc
    tensors = []
    shard_ids = set()
    shard_counts = set()
    for path in paths:
        reader = GGUFReader(str(path), "r")
        fields = reader.fields
        arch = fields.get("general.architecture")
        if arch is None or arch.contents() != "qwen35":
            raise ValueError(f"{path}: expected qwen35 architecture")
        for key, expected in (("qwen35.block_count", 64), ("qwen35.embedding_length", 5120)):
            if key not in fields or fields[key].contents() != expected:
                raise ValueError(f"{path}: expected {key}={expected}")
        count = int(fields["split.count"].contents()) if "split.count" in fields else 1
        shard_id = int(fields["split.no"].contents()) if "split.no" in fields else 0
        if count <= 0 or not 0 <= shard_id < count or shard_id in shard_ids:
            raise ValueError(f"{path}: invalid or duplicate shard number")
        shard_counts.add(count)
        shard_ids.add(shard_id)
        tensors.extend({"name": t.name, "bytes": int(t.n_bytes)} for t in reader.tensors)
        del reader
    if shard_counts != {len(paths)} or shard_ids != set(range(len(paths))):
        raise ValueError("incomplete or inconsistent GGUF shard set")
    return tensors


def tensors_from_report(report):
    """Consume a metadata-only report without requiring NumPy or model weights."""
    metadata = report["metadata"]
    if (metadata.get("general.architecture") != "qwen35"
            or metadata.get("qwen35.block_count") != 64
            or metadata.get("qwen35.embedding_length") != 5120):
        raise ValueError("metadata report does not match Qwen3.8-27B shape")
    if metadata.get("split.count", 1) != 1:
        raise ValueError("remote report mode currently requires an unsplit GGUF")
    tensors = [{"name": f"blk.{layer['index']}.aggregate", "bytes": layer["bytes"]}
               for layer in report["layers"]]
    resident = report["non_block_tensor_bytes"]
    if resident < 0 or sum(t["bytes"] for t in tensors) + resident != report["tensor_bytes"]:
        raise ValueError("inconsistent metadata report byte totals")
    if resident:
        tensors.append({"name": "resident.aggregate", "bytes": resident})
    return tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", nargs="*", type=Path)
    parser.add_argument("--metadata-report", type=Path, help="inspect-remote-gguf.py JSON output")
    parser.add_argument("--ram-mib", required=True, type=int, help="measured app budget")
    parser.add_argument("--runtime-mib", required=True, type=int, help="explicit runtime reserve")
    parser.add_argument("--read-gbs", required=True, type=float, help="measured SSD GB/s, decimal")
    args = parser.parse_args()
    if bool(args.gguf) == bool(args.metadata_report):
        parser.error("provide either local GGUF files or --metadata-report")
    try:
        report = json.loads(args.metadata_report.read_text()) if args.metadata_report else None
        plan = make_plan(
            tensors_from_report(report) if report is not None else inspect_ggufs(args.gguf),
            ram_bytes=args.ram_mib * MIB,
            runtime_bytes=args.runtime_mib * MIB,
            read_gbs=args.read_gbs,
        )
        if report is not None:
            plan["source"] = report.get("source")
            plan["header_sha256"] = report.get("header_sha256")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(plan, indent=2, allow_nan=False))
    return 0 if plan["budget_fits_arithmetically"] else 1


if __name__ == "__main__":
    sys.exit(main())
