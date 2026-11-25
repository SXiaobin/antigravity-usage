"""Tests for request-level deduplication across branched / forked sessions."""

import os
import sqlite3
import tempfile
import unittest
from datetime import datetime

from agy_usage.aggregator import UsageAggregator
from agy_usage.cache import summary_from_dict, summary_to_dict
from agy_usage.limits import LimitsCalculator
from agy_usage.models import CostBreakdown, SessionSummary, StepRecord, TokenUsage
from agy_usage.parser import (
    AntigravityParser,
    extract_usage_request_id,
)


def _encode_varint(n: int) -> bytes:
    buf = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            buf.append(b | 0x80)
        else:
            buf.append(b)
            break
    return bytes(buf)


def _make_field(fnum: int, wtype: int, val: bytes) -> bytes:
    header = _encode_varint((fnum << 3) | wtype)
    return header + val


def _make_bytes_field(fnum: int, b: bytes) -> bytes:
    length = _encode_varint(len(b))
    return _make_field(fnum, 2, length + b)


def _make_varint_field(fnum: int, n: int) -> bytes:
    return _make_field(fnum, 0, _encode_varint(n))


class TestBranchRequestDeduplication(unittest.TestCase):
    """Verify deduplication of LLM requests across branched sessions."""

    def test_extract_usage_request_id(self):
        # Field 7: bot-uuid
        usage_bytes = _make_bytes_field(7, b"bot-00000000-0000-0000-0000-000000000001")
        self.assertEqual(
            extract_usage_request_id(usage_bytes),
            "bot-00000000-0000-0000-0000-000000000001",
        )

        # Field 11 fallback when Field 7 is absent
        usage_bytes_f11 = _make_bytes_field(11, b"test-fallback-request-id-1")
        self.assertEqual(
            extract_usage_request_id(usage_bytes_f11),
            "test-fallback-request-id-1",
        )

        # Empty / no id
        self.assertIsNone(extract_usage_request_id(b""))

    def test_step_record_to_dict_and_cache_roundtrip(self):
        now = datetime(2026, 11, 24, 12, 0, 0)
        step = StepRecord(
            step_idx=1,
            timestamp=now,
            model="gemini-3.8-flash",
            usage=TokenUsage(prompt_tokens=100, output_tokens=50),
            request_id="bot-test-uuid-1",
        )
        d = step.to_dict()
        self.assertEqual(d["request_id"], "bot-test-uuid-1")

        sess = SessionSummary(
            session_id="sess-1",
            source="cli",
            file_path="/tmp/sess1.db",
            steps=[step],
            usage=step.usage,
            step_count=1,
        )
        cached = summary_to_dict(sess)
        self.assertEqual(cached["steps"][0][9], "bot-test-uuid-1")

        # Roundtrip
        restored = summary_from_dict(cached)
        self.assertEqual(len(restored.steps), 1)
        self.assertEqual(restored.steps[0].request_id, "bot-test-uuid-1")

        # Backwards compatibility: restore from 9-element step array (legacy cache)
        legacy_cached = dict(cached)
        legacy_cached["steps"] = [cached["steps"][0][:9]]
        restored_legacy = summary_from_dict(legacy_cached)
        self.assertEqual(len(restored_legacy.steps), 1)
        self.assertIsNone(restored_legacy.steps[0].request_id)

    def test_cross_session_aggregation_dedup(self):
        ts1 = datetime(2026, 11, 24, 10, 0, 0)
        ts2 = datetime(2026, 11, 24, 10, 5, 0)
        ts3 = datetime(2026, 11, 24, 10, 10, 0)

        # Parent session: steps 1 and 2
        step1_parent = StepRecord(
            step_idx=1,
            timestamp=ts1,
            model="gemini-3.8-flash",
            usage=TokenUsage(prompt_tokens=1000, output_tokens=200),
            request_id="bot-req-1",
        )
        step2_parent = StepRecord(
            step_idx=2,
            timestamp=ts2,
            model="gemini-3.8-flash",
            usage=TokenUsage(prompt_tokens=2000, output_tokens=300),
            request_id="bot-req-2",
        )
        sess_parent = SessionSummary(
            session_id="parent-sess",
            source="cli",
            file_path="/tmp/parent.db",
            start_time=ts1,
            end_time=ts2,
            step_count=2,
            usage=step1_parent.usage.add(step2_parent.usage),
            steps=[step1_parent, step2_parent],
        )

        # Child session: inherited step 1 and 2, plus newly executed step 3
        step1_child = StepRecord(
            step_idx=1,
            timestamp=ts1,
            model="gemini-3.8-flash",
            usage=TokenUsage(prompt_tokens=1000, output_tokens=200),
            request_id="bot-req-1",
        )
        step2_child = StepRecord(
            step_idx=2,
            timestamp=ts2,
            model="gemini-3.8-flash",
            usage=TokenUsage(prompt_tokens=2000, output_tokens=300),
            request_id="bot-req-2",
        )
        step3_child = StepRecord(
            step_idx=3,
            timestamp=ts3,
            model="gemini-3.8-flash",
            usage=TokenUsage(prompt_tokens=3000, output_tokens=500),
            request_id="bot-req-3",
        )
        sess_child = SessionSummary(
            session_id="child-sess",
            source="cli",
            file_path="/tmp/child.db",
            start_time=ts1,
            end_time=ts3,
            step_count=3,
            usage=step1_child.usage.add(step2_child.usage).add(step3_child.usage),
            steps=[step1_child, step2_child, step3_child],
        )

        sessions = [sess_parent, sess_child]

        # 1. Daily aggregation
        daily = UsageAggregator.aggregate_daily(sessions)
        d_usage = daily[ts1.date()]
        # Expected: 1000+2000+3000 = 6000 prompt, 200+300+500 = 1000 output.
        # Without dedup, it would be 9000 prompt, 1500 output.
        self.assertEqual(d_usage.usage.prompt_tokens, 6000)
        self.assertEqual(d_usage.usage.output_tokens, 1000)
        self.assertEqual(d_usage.steps_count, 3)

        # 2. Models aggregation
        models = UsageAggregator.aggregate_models(sessions)
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].usage.prompt_tokens, 6000)
        self.assertEqual(models[0].usage.output_tokens, 1000)
        self.assertEqual(models[0].request_count, 3)

        # 3. compute_summary
        summary = UsageAggregator.compute_summary(sessions, daily_data=daily)
        self.assertEqual(summary["step_count"], 3)
        self.assertEqual(summary["total_usage"].prompt_tokens, 6000)
        self.assertEqual(summary["total_usage"].output_tokens, 1000)
        self.assertEqual(summary["total_cost"].prompt_cost, d_usage.cost.prompt_cost)

        # 4. Limits pool items
        pool_items = list(LimitsCalculator._iter_pool_items(sessions))
        self.assertEqual(len(pool_items), 3)

    def test_sqlite_parsing_extracts_request_id(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "test_session.db")
            conn = sqlite3.connect(db_path)
            conn.execute(
                "CREATE TABLE steps (idx INTEGER PRIMARY KEY, step_type INTEGER, metadata BLOB, step_payload BLOB)"
            )

            # Build metadata:
            # Field 1 (step timestamp): Seconds 1764000000
            ts_payload = _make_varint_field(1, 1764000000)
            f1_ts = _make_bytes_field(1, ts_payload)

            # Field 9 (UsageMetadata):
            # Field 2 (prompt_tokens): 500
            # Field 3 (output_tokens): 100
            # Field 7 (request_id): b"bot-test-12345"
            u_f2 = _make_varint_field(2, 500)
            u_f3 = _make_varint_field(3, 100)
            u_f7 = _make_bytes_field(7, b"bot-test-12345")
            usage_blob = u_f2 + u_f3 + u_f7
            f9_usage = _make_bytes_field(9, usage_blob)

            meta_blob = f1_ts + f9_usage
            conn.execute(
                "INSERT INTO steps (idx, step_type, metadata) VALUES (1, 1, ?)",
                (meta_blob,),
            )
            conn.commit()
            conn.close()

            summary, complete = AntigravityParser._parse_session_db_checked(db_path, source="cli")
            self.assertIsNotNone(summary)
            self.assertEqual(len(summary.steps), 1)
            self.assertEqual(summary.steps[0].request_id, "bot-test-12345")

    def test_sqlite_parsing_parent_references_filters_inherited_steps(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "child_session.db")
            conn = sqlite3.connect(db_path)
            conn.execute(
                "CREATE TABLE steps (idx INTEGER PRIMARY KEY, step_type INTEGER, metadata BLOB, step_payload BLOB)"
            )
            conn.execute("CREATE TABLE parent_references (val BLOB)")

            # Insert 5 steps with distinct timestamps and usages
            # Steps 1..3: inherited from parent
            # Steps 4..5: new in child
            for s_idx in range(1, 6):
                # Timestamp: 1764000000 + s_idx * 100
                ts_payload = _make_varint_field(1, 1764000000 + s_idx * 100)
                f1_ts = _make_bytes_field(1, ts_payload)

                u_f2 = _make_varint_field(2, 1000)
                u_f3 = _make_varint_field(3, 200)
                u_f7 = _make_bytes_field(7, f"bot-req-{s_idx}".encode())
                usage_blob = u_f2 + u_f3 + u_f7
                f9_usage = _make_bytes_field(9, usage_blob)

                meta_blob = f1_ts + f9_usage
                conn.execute(
                    "INSERT INTO steps (idx, step_type, metadata) VALUES (?, 1, ?)",
                    (s_idx, meta_blob),
                )

            # Insert parent_references:
            # 1. Placeholder with uint64 -1 (18446744073709551615) - should be ignored
            f6_p1 = _make_bytes_field(6, b"dummy-parent")
            f2_p1 = _make_varint_field(2, 18446744073709551615)
            conn.execute("INSERT INTO parent_references (val) VALUES (?)", (f6_p1 + f2_p1,))

            # 2. Self reference - should be ignored
            f6_p2 = _make_bytes_field(6, b"child_session")
            f2_p2 = _make_varint_field(2, 2)
            conn.execute("INSERT INTO parent_references (val) VALUES (?)", (f6_p2 + f2_p2,))

            # 3. Valid parent reference: parent_id = "parent-sess-uuid", step cutoff = 3
            f6_p3 = _make_bytes_field(6, b"parent-sess-uuid")
            f2_p3 = _make_varint_field(2, 3)
            conn.execute("INSERT INTO parent_references (val) VALUES (?)", (f6_p3 + f2_p3,))

            conn.commit()
            conn.close()

            summary, complete = AntigravityParser._parse_session_db_checked(db_path, source="cli")
            self.assertIsNotNone(summary)
            self.assertTrue(complete)
            self.assertEqual(summary.parent_session_id, "parent-sess-uuid")
            self.assertEqual(summary.parent_step_index, 3)
            self.assertEqual(summary.inherited_step_count, 3)
            # Only steps 4 and 5 should be kept in summary
            self.assertEqual(summary.step_count, 2)
            self.assertEqual(len(summary.steps), 2)
            self.assertEqual(summary.steps[0].step_idx, 4)
            self.assertEqual(summary.steps[1].step_idx, 5)
            self.assertEqual(summary.usage.prompt_tokens, 2000)
            self.assertEqual(summary.usage.output_tokens, 400)
            self.assertEqual(summary.start_time, datetime.fromtimestamp(1764000400))
            self.assertEqual(summary.end_time, datetime.fromtimestamp(1764000500))

            # Verify cache serialization roundtrip
            cached = summary_to_dict(summary)
            self.assertEqual(cached["parent_session_id"], "parent-sess-uuid")
            self.assertEqual(cached["parent_step_index"], 3)
            self.assertEqual(cached["inherited_step_count"], 3)

            restored = summary_from_dict(cached)
            self.assertEqual(restored.parent_session_id, "parent-sess-uuid")
            self.assertEqual(restored.parent_step_index, 3)
            self.assertEqual(restored.inherited_step_count, 3)
            self.assertEqual(restored.step_count, 2)
            self.assertEqual(len(restored.steps), 2)


if __name__ == "__main__":
    unittest.main()
