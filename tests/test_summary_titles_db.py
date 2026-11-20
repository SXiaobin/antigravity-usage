"""Tests for reading conversation titles from sibling conversation_summaries.db."""

import os
import sqlite3
import tempfile
import unittest

from agy_usage.parser import AntigravityParser


class TestSummaryTitlesDb(unittest.TestCase):
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

    def _step_metadata(self, prompt: int, output: int) -> bytes:
        ts = bytes([0x08]) + self._varint(1_751_234_567)
        usage = self._varint(2 << 3) + self._varint(prompt)
        usage += self._varint(3 << 3) + self._varint(output)
        return bytes([0x0A, len(ts)]) + ts + bytes([0x4A, len(usage)]) + usage

    def _write_session_db(self, path: str, prompt_text: str) -> None:
        conn = sqlite3.connect(path)
        try:
            conn.execute(
                "CREATE TABLE steps (idx INTEGER PRIMARY KEY, step_type TEXT, "
                "step_payload BLOB, metadata BLOB)"
            )
            conn.execute(
                "INSERT INTO steps (idx, step_type, step_payload, metadata) "
                "VALUES (0, 'user', ?, ?)",
                (self._pb_str(1, prompt_text), self._step_metadata(100, 20)),
            )
            conn.commit()
        finally:
            conn.close()

    def test_prefers_conversation_summaries_title_and_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            conv_dir = os.path.join(tmp, "conversations")
            os.makedirs(conv_dir)
            self._write_session_db(os.path.join(conv_dir, "sess-1.db"), "very long first prompt")
            self._write_session_db(os.path.join(conv_dir, "sess-2.db"), "another prompt")
            self._write_session_db(os.path.join(conv_dir, "sess-3.db"), "fallback prompt title")

            sum_db = os.path.join(tmp, "conversation_summaries.db")
            conn = sqlite3.connect(sum_db)
            try:
                conn.execute(
                    "CREATE TABLE conversation_summaries "
                    "(conversation_id TEXT PRIMARY KEY, title TEXT, preview TEXT)"
                )
                conn.execute(
                    "INSERT INTO conversation_summaries VALUES (?, ?, ?)",
                    ("sess-1", "Clean Summary Title", "Ignored Preview"),
                )
                conn.execute(
                    "INSERT INTO conversation_summaries VALUES (?, ?, ?)",
                    ("sess-2", "", "Preview Fallback Title"),
                )
                conn.commit()
            finally:
                conn.close()

            sessions = {
                s.session_id: s
                for s in AntigravityParser.scan_directory(conv_dir, source="cli")
            }
            self.assertEqual(sessions["sess-1"].title, "Clean Summary Title")
            self.assertEqual(sessions["sess-2"].title, "Preview Fallback Title")
            self.assertEqual(sessions["sess-3"].title, "fallback prompt title")


if __name__ == "__main__":
    unittest.main()
