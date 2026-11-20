"""Regression tests for gen_metadata rows stored with size = 0."""

import os
import sqlite3
import tempfile
import unittest

from agy_usage.parser import AntigravityParser


class TestGenMetadataZeroSize(unittest.TestCase):
    """gen_metadata rows with non-empty data must be parsed even when size == 0."""

    @staticmethod
    def _varint(value: int) -> bytes:
        out = bytearray()
        while True:
            b = value & 0x7F
            value >>= 7
            if value:
                out.append(b | 0x80)
            else:
                out.append(b)
                return bytes(out)

    def _pb_str(self, field_num: int, value: str) -> bytes:
        payload = value.encode("utf-8")
        return self._varint((field_num << 3) | 2) + self._varint(len(payload)) + payload

    def _gen_metadata_row(self, model_name="", last_step_idx=None, model_enum=None) -> bytes:
        f20 = b""
        if last_step_idx is not None:
            f20 += self._pb_str(1, "last_step_index") + self._pb_str(2, str(last_step_idx))
        if model_enum is not None:
            f20 += self._pb_str(1, "model_enum") + self._pb_str(2, model_enum)
        parts = []
        if model_name:
            parts.append(self._pb_str(19, model_name))
        if f20:
            parts.append(self._varint((20 << 3) | 2) + self._varint(len(f20)) + f20)
        f1 = b"".join(parts)
        return self._varint((1 << 3) | 2) + self._varint(len(f1)) + f1

    def _step_metadata(self, prompt: int, output: int) -> bytes:
        ts = bytes([0x08]) + self._varint(1_751_234_567)
        usage = self._varint(2 << 3) + self._varint(prompt)
        usage += self._varint(3 << 3) + self._varint(output)
        return bytes([0x0A, len(ts)]) + ts + bytes([0x4A, len(usage)]) + usage

    def test_zero_size_gen_metadata_with_data_is_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "zero-size.db")
            conn = sqlite3.connect(path)
            try:
                conn.execute(
                    "CREATE TABLE steps (idx INTEGER PRIMARY KEY, step_type TEXT, "
                    "step_payload BLOB, metadata BLOB)"
                )
                conn.execute(
                    "CREATE TABLE gen_metadata (idx INTEGER PRIMARY KEY, data BLOB, size INTEGER)"
                )
                blob = self._gen_metadata_row(
                    model_name="gemini-4-hydrogen",
                    last_step_idx=1,
                    model_enum="MODEL_PLACEHOLDER_M462",
                )
                conn.execute(
                    "INSERT INTO gen_metadata (idx, data, size) VALUES (?, ?, 0)",
                    (0, blob),
                )
                conn.execute(
                    "INSERT INTO gen_metadata (idx, data, size) VALUES (?, ?, 0)",
                    (1, b""),
                )
                conn.execute(
                    "INSERT INTO gen_metadata (idx, data, size) VALUES (?, NULL, 0)",
                    (2,),
                )
                conn.execute(
                    "INSERT INTO steps (idx, step_type, step_payload, metadata) "
                    "VALUES (?, 'user', '', ?)",
                    (1, self._step_metadata(1000, 200)),
                )
                conn.commit()
            finally:
                conn.close()

            summary = AntigravityParser.parse_session_db(path)
            self.assertIsNotNone(summary)
            self.assertEqual(summary.models, ["gemini-4-hydrogen"])
            self.assertEqual(summary.steps[0].model, "gemini-4-hydrogen")

    def test_known_enum_m462_resolves_without_field_19(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "enum-m462.db")
            conn = sqlite3.connect(path)
            try:
                conn.execute(
                    "CREATE TABLE steps (idx INTEGER PRIMARY KEY, step_type TEXT, "
                    "step_payload BLOB, metadata BLOB)"
                )
                conn.execute(
                    "CREATE TABLE gen_metadata (idx INTEGER PRIMARY KEY, data BLOB, size INTEGER)"
                )
                blob = self._gen_metadata_row(
                    last_step_idx=1,
                    model_enum="MODEL_PLACEHOLDER_M462",
                )
                conn.execute(
                    "INSERT INTO gen_metadata (idx, data, size) VALUES (?, ?, 0)",
                    (0, blob),
                )
                conn.execute(
                    "INSERT INTO steps (idx, step_type, step_payload, metadata) "
                    "VALUES (?, 'user', '', ?)",
                    (1, self._step_metadata(500, 100)),
                )
                conn.commit()
            finally:
                conn.close()

            summary = AntigravityParser.parse_session_db(path)
            self.assertIsNotNone(summary)
            self.assertEqual(summary.models, ["gemini-4-hydrogen"])


if __name__ == "__main__":
    unittest.main()
