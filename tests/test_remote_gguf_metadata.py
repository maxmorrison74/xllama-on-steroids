#!/usr/bin/env python3
"""Bounded metadata inspection: no network, NumPy, or real weights required."""

import importlib.util
import struct
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("remote_gguf", ROOT / "scripts/inspect-remote-gguf.py")
remote = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote)


def string(value):
    value = value.encode()
    return struct.pack("<Q", len(value)) + value


def fixture(*, offset=0, dims=32, kind=0, alignment=32):
    fields = [
        string("general.architecture") + struct.pack("<I", 8) + string("qwen35"),
        string("general.alignment") + struct.pack("<II", 4, alignment),
        string("ignored.array") + struct.pack("<IIQ", 9, 8, 2) + string("a") + string("b"),
    ]
    tensor = string("blk.0.weight") + struct.pack("<IQIQ", 1, dims, kind, offset)
    data = b"GGUF" + struct.pack("<IQQ", 3, 1, len(fields)) + b"".join(fields) + tensor
    return data + b"\0" * (-len(data) % 32)


class MetadataTests(unittest.TestCase):
    def test_reads_only_header_and_calculates_quantized_bytes(self):
        data = fixture(kind=12, dims=256)
        report = remote.parse_header(data, len(data) + 144, {12: (256, 144)})
        self.assertEqual(report["tensor_count"], 1)
        self.assertEqual(report["tensor_bytes"], 144)
        self.assertEqual(report["header_bytes"], len(data))
        self.assertEqual(report["largest_block_bytes"], 144)
        self.assertNotIn("ignored.array", report["metadata"])

    def test_rejects_truncated_headers(self):
        data = fixture()
        for length in (0, 4, 8, 24, len(data) - 1):
            with self.subTest(length=length), self.assertRaises(ValueError):
                remote.parse_header(data[:length], 4096, {0: (1, 4)})

    def test_rejects_out_of_file_range_and_unknown_types(self):
        data = fixture()
        with self.assertRaisesRegex(ValueError, "exceeds remote"):
            remote.parse_header(data, len(data), {0: (1, 4)})
        with self.assertRaisesRegex(ValueError, "unknown GGML"):
            remote.parse_header(data, 4096, {})

    def test_rejects_invalid_shape_or_alignment(self):
        for data in (fixture(dims=0), fixture(offset=1), fixture(alignment=0),
                     fixture(kind=12, dims=257)):
            with self.assertRaises(ValueError):
                remote.parse_header(data, 4096, {0: (1, 4), 12: (256, 144)})


class HttpRangeTests(unittest.TestCase):
    def response(self, status, content_range, data):
        response = SimpleNamespace(status=status, headers={"Content-Range": content_range},
                                   read=lambda count: data[:count])
        context = unittest.mock.MagicMock()
        context.__enter__.return_value = response
        return context

    def test_rejects_server_ignoring_range_without_reading_body(self):
        response = self.response(200, "", b"")
        response.__enter__.return_value.read = unittest.mock.Mock()
        with patch.object(remote, "urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "did not honor"):
                remote.fetch_prefix("https://example.test/a")
        response.__enter__.return_value.read.assert_not_called()

    def test_accepts_short_complete_file(self):
        with patch.object(remote, "urlopen", return_value=self.response(206, "bytes 0-3/4", b"GGUF")):
            self.assertEqual(remote.fetch_prefix("https://example.test/a"), (b"GGUF", 4))

    def test_rejects_invalid_ranges_and_body_lengths(self):
        for value, body in (("bytes 1-3/4", b"abc"), ("bytes 0-2/4", b"abc"),
                            ("bytes 0-3/4", b"abc"), ("bytes 0-3/4", b"abcde")):
            with patch.object(remote, "urlopen", return_value=self.response(206, value, body)):
                with self.subTest(value=value, body=body), self.assertRaises(ValueError):
                    remote.fetch_prefix("https://example.test/a")


if __name__ == "__main__":
    unittest.main()
