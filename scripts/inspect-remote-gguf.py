#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Read a bounded GGUF header from a pinned Hugging Face artifact, not its weights.

No third-party dependencies. Reports descriptor sizes, not runtime compatibility
or full-file integrity. The server MUST honor the HTTP range request.
"""

import argparse
import hashlib
import io
import json
import math
import re
import runpy
import struct
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 16 * 1024 * 1024
KEEP = {"general.architecture", "general.name", "general.alignment",
        "qwen35.block_count", "qwen35.embedding_length", "split.count", "split.no"}
KEEP.update(f"{arch}.{key}" for arch in ("qwen3moe", "qwen35moe")
            for key in ("block_count", "embedding_length", "expert_count", "expert_used_count"))
SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
           7: "?", 10: "Q", 11: "q", 12: "d"}


class Header:
    def __init__(self, data):
        self.stream = io.BytesIO(data)
        self.size = len(data)

    def read(self, count):
        if count < 0 or count > self.size - self.stream.tell():
            raise ValueError("GGUF header is truncated or exceeds download limit")
        return self.stream.read(count)

    def number(self, fmt):
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))[0]

    def string(self, keep=True):
        data = self.read(self.number("Q"))
        return data.decode("utf-8") if keep else None

    def value(self, kind, keep=False, depth=0):
        if kind in SCALARS:
            value = self.number(SCALARS[kind])
            return value if keep else None
        if kind == 8:
            return self.string(keep)
        if kind == 9:
            if depth >= 2:
                raise ValueError("excessive GGUF array nesting")
            subtype, count = self.number("I"), self.number("Q")
            if count > self.size:
                raise ValueError("excessive GGUF array count")
            if keep:
                raise ValueError("expected scalar metadata, got array")
            if subtype in SCALARS:
                self.read(count * struct.calcsize("<" + SCALARS[subtype]))
            else:
                for _ in range(count):
                    self.value(subtype, depth=depth + 1)
            return None
        raise ValueError(f"unknown GGUF metadata type {kind}")


def routed_expert_layout(tensors, metadata, data_offset):
    """Cost a proposed page-aligned expert layout; do not repack or execute it."""
    arch = metadata.get("general.architecture")
    if arch not in ("qwen3moe", "qwen35moe"):
        return None
    experts = metadata.get(f"{arch}.expert_count")
    active = metadata.get(f"{arch}.expert_used_count")
    blocks = metadata.get(f"{arch}.block_count")
    if (not all(isinstance(v, int) and not isinstance(v, bool) and 0 < v <= 100000
                for v in (experts, active, blocks)) or active > experts):
        raise ValueError("missing or invalid MoE expert metadata")
    layers = {}
    for tensor in tensors:
        match = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down|gate_up)_exps\.weight", tensor["name"])
        if not match:
            if "_exps" in tensor["name"]:
                raise ValueError("unsupported routed-expert tensor layout")
            continue
        if len(tensor["shape"]) != 3 or tensor["shape"][2] != experts:
            raise ValueError("expected contiguous expert slabs on the third tensor axis")
        if tensor["bytes"] % experts:
            raise ValueError("expert tensor size is not divisible by expert count")
        layer = int(match[1])
        slab = tensor["bytes"] // experts
        layers.setdefault(layer, []).append({
            "projection": match[2], "bytes_per_expert": slab,
            "ggml_type": tensor["type"],
            "source_tensor_file_offset": data_offset + tensor["offset"],
        })
    if sorted(layers) != list(range(blocks)):
        raise ValueError("missing MoE decoder expert blocks")
    descriptions = []
    for layer, projections in sorted(layers.items()):
        kinds = {p["projection"] for p in projections}
        if kinds not in ({"gate", "up", "down"}, {"gate_up", "down"}):
            raise ValueError("incomplete expert FFN projections")
        descriptions.append({
            "index": layer,
            "expert_payload_bytes": sum(p["bytes_per_expert"] for p in projections),
            "proposed_expert_stride_bytes": sum((p["bytes_per_expert"] + 4095) // 4096 * 4096
                                                for p in projections),
            "projections": projections,
        })
    per_slot = sum(layer["proposed_expert_stride_bytes"] for layer in descriptions)
    expert_bytes = sum(layer["expert_payload_bytes"] for layer in descriptions) * experts
    return {
        "status": "layout_arithmetic_only_not_runtime_or_cache_measurement",
        "experts_per_layer": experts, "active_experts_per_token_per_layer": active,
        "common_tensor_bytes": sum(t["bytes"] for t in tensors) - expert_bytes,
        "all_routed_expert_tensor_bytes": expert_bytes,
        "proposed_bytes_per_slot_across_all_layers": per_slot,
        "proposed_no_cache_expert_read_bytes_per_decode_token": per_slot * active,
        "packing_assumption": "one expert blob with each projection padded to 4096 bytes; original quantized bytes unchanged",
        "layers": descriptions,
    }


def parse_header(data, file_bytes, quant_sizes):
    header = Header(data)
    if header.read(4) != b"GGUF" or header.number("I") != 3:
        raise ValueError("expected little-endian GGUF v3")
    n_tensors, n_fields = header.number("Q"), header.number("Q")
    if n_tensors > 100000 or n_fields > 100000:
        raise ValueError("excessive GGUF descriptor count")
    metadata, keys = {}, set()
    for _ in range(n_fields):
        key = header.string()
        if key in keys:
            raise ValueError(f"duplicate metadata key {key}")
        keys.add(key)
        value = header.value(header.number("I"), keep=key in KEEP)
        if key in KEEP:
            metadata[key] = value
    tensors, names = [], set()
    alignment = metadata.get("general.alignment", 32)
    if not isinstance(alignment, int) or alignment <= 0 or alignment > 1024 * 1024:
        raise ValueError("invalid GGUF alignment")
    for _ in range(n_tensors):
        name, n_dims = header.string(), header.number("I")
        if name in names or not 1 <= n_dims <= 4:
            raise ValueError("duplicate tensor or invalid tensor dimensions")
        names.add(name)
        shape = [header.number("Q") for _ in range(n_dims)]
        kind, offset = header.number("I"), header.number("Q")
        if kind not in quant_sizes:
            raise ValueError(f"unknown GGML quantization type {kind}")
        block, size = quant_sizes[kind]
        if min(shape) <= 0 or shape[0] % block or offset % alignment:
            raise ValueError(f"invalid tensor shape/alignment: {name}")
        tensors.append({"name": name, "bytes": math.prod(shape) // block * size,
                        "offset": offset, "shape": shape, "type": kind})
    data_offset = (header.stream.tell() + alignment - 1) // alignment * alignment
    header.read(data_offset - header.stream.tell())
    end = 0
    for tensor in sorted(tensors, key=lambda tensor: tensor["offset"]):
        if tensor["offset"] < end:
            raise ValueError("overlapping tensor data ranges")
        end = tensor["offset"] + tensor["bytes"]
        if data_offset + end > file_bytes:
            raise ValueError("tensor data range exceeds remote file size")
    layers, resident = {}, 0
    for tensor in tensors:
        match = re.match(r"^blk\.(\d+)\.", tensor["name"])
        if match:
            layer = int(match[1])
            layers[layer] = layers.get(layer, 0) + tensor["bytes"]
        else:
            resident += tensor["bytes"]
    report = {
        "metadata": metadata,
        "header_bytes": data_offset,
        "header_sha256": hashlib.sha256(data[:data_offset]).hexdigest(),
        "tensor_count": len(tensors),
        "tensor_bytes": sum(t["bytes"] for t in tensors),
        "non_block_tensor_bytes": resident,
        "largest_block_bytes": max(layers.values(), default=0),
        "block_bytes_per_decode_token": sum(layers.values()),
        "layers": [{"index": i, "bytes": layers[i]} for i in sorted(layers)],
    }
    moe = routed_expert_layout(tensors, metadata, data_offset)
    if moe is not None:
        report["routed_experts"] = moe
    return report


def fetch_prefix(url):
    request = Request(url, headers={"Range": f"bytes=0-{LIMIT - 1}",
                                    "Accept-Encoding": "identity"})
    with urlopen(request, timeout=30) as response:
        match = re.fullmatch(r"bytes 0-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
        if response.status != 206 or not match:
            raise ValueError("server did not honor bounded range request; download aborted")
        end, total = map(int, match.groups())
        if end != min(LIMIT, total) - 1 or total <= 0:
            raise ValueError("unexpected Content-Range; download aborted")
        data = response.read(end + 2)
        if len(data) != end + 1:
            raise ValueError("truncated or oversized HTTP range response")
    return data, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True, help="full 40-character commit SHA")
    parser.add_argument("--filename", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-f0-9]{40}", args.revision):
        parser.error("revision must be a pinned commit SHA, not main")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", args.repo) or not args.filename.endswith(".gguf"):
        parser.error("expected owner/repo and a GGUF filename")
    url = (f"https://huggingface.co/{args.repo}/resolve/{args.revision}/"
           f"{quote(args.filename, safe='')}")
    try:
        constants = runpy.run_path(str(ROOT / "llama.cpp/gguf-py/gguf/constants.py"))
        data, total = fetch_prefix(url)
        report = parse_header(data, total, constants["GGML_QUANT_SIZES"])
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    report["source"] = {"repo": args.repo, "revision": args.revision,
                        "filename": args.filename, "file_bytes": total,
                        "fetched_bytes": len(data), "full_file_sha256_verified": False}
    report["status"] = "metadata_only_not_inference_validation"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
