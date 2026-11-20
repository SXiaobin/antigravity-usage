"""Tests for live monitor time-range filtering, step-level slicing, and watch mode."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta
import io
import json
import unittest
from unittest import mock

from agy_usage.aggregator import (
    compute_live_window_start,
    filter_sessions_since,
)
from agy_usage.cli import (
    cmd_live,
    main as cli_main,
    parse_args,
)
from agy_usage.display import Colors, strip_ansi
from agy_usage.models import (
    SessionSummary,
    StepRecord,
    TokenUsage,
)


def _make_step(
    step_idx: int,
    timestamp: datetime,
    model: str = "gemini-3.8-flash",
    prompt_tokens: int = 1000,
    cached_tokens: int = 200,
    output_tokens: int = 100,
) -> StepRecord:
    usage = TokenUsage(
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        output_tokens=output_tokens,
        thought_tokens=0,
        candidate_tokens=output_tokens,
    )
    return StepRecord(
        step_idx=step_idx,
        timestamp=timestamp,
        model=model,
        usage=usage,
    )


def _make_session(
    session_id: str,
    steps: list[StepRecord] | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    models: list[str] | None = None,
    workspace: str = "workspace-test",
    title: str = "Test Session",
    usage: TokenUsage | None = None,
    step_count: int | None = None,
) -> SessionSummary:
    if steps:
        total_usage = TokenUsage()
        for st in steps:
            total_usage = total_usage.add(st.usage)
        s_models = models or list(dict.fromkeys(st.model for st in steps))
        s_start = start_time or steps[0].timestamp
        s_end = end_time or steps[-1].timestamp
        s_count = step_count if step_count is not None else len(steps)
        return SessionSummary(
            session_id=session_id,
            source="cli",
            file_path=f"/fake/{session_id}.db",
            title=title,
            workspace=workspace,
            models=s_models,
            start_time=s_start,
            end_time=s_end,
            step_count=s_count,
            usage=usage or total_usage,
            steps=list(steps),
        )
    return SessionSummary(
        session_id=session_id,
        source="cli",
        file_path=f"/fake/{session_id}.db",
        title=title,
        workspace=workspace,
        models=models or (["gemini-3.8-flash"] if usage else []),
        start_time=start_time,
        end_time=end_time or start_time,
        step_count=step_count or 0,
        usage=usage or TokenUsage(),
        steps=[],
    )


class TestWindowBoundaryCalculations(unittest.TestCase):
    """Test start datetime calculation for today, week (Wednesday-anchored), and all."""

    def test_today_boundary_at_various_times(self):
        fixed_dt = datetime(2026, 11, 20, 15, 45, 30, 123456)
        start = compute_live_window_start("today", now=fixed_dt)
        self.assertEqual(start, datetime(2026, 11, 20, 0, 0, 0, 0))

        midnight = datetime(2026, 11, 20, 0, 0, 0, 0)
        self.assertEqual(compute_live_window_start("today", now=midnight), midnight)

        end_of_day = datetime(2026, 11, 20, 23, 59, 59, 999999)
        self.assertEqual(
            compute_live_window_start("today", now=end_of_day),
            datetime(2026, 11, 20, 0, 0, 0, 0),
        )

    def test_today_boundary_defaults_to_now(self):
        start = compute_live_window_start("today")
        now = datetime.now()
        self.assertEqual(start.date(), now.date())
        self.assertEqual(start.hour, 0)
        self.assertEqual(start.minute, 0)
        self.assertEqual(start.second, 0)
        self.assertEqual(start.microsecond, 0)

    def test_all_boundary_returns_none(self):
        self.assertIsNone(compute_live_window_start("all"))
        self.assertIsNone(compute_live_window_start("all", now=datetime(2026, 11, 20)))

    def test_week_boundary_wednesday_anchored(self):
        # 2026-11-18 is Wednesday (weekday 2)
        wed = datetime(2026, 11, 18, 14, 30, 0)
        expected_wed = datetime(2026, 11, 18, 0, 0, 0)
        self.assertEqual(compute_live_window_start("week", now=wed), expected_wed)

        # 2026-11-19 is Thursday (weekday 3) -> resolves to Wednesday 2026-11-18 00:00:00
        thu = datetime(2026, 11, 19, 9, 0, 0)
        self.assertEqual(compute_live_window_start("week", now=thu), expected_wed)

        # 2026-11-20 is Friday (weekday 4) -> resolves to Wednesday 2026-11-18 00:00:00
        fri = datetime(2026, 11, 20, 23, 59, 59)
        self.assertEqual(compute_live_window_start("week", now=fri), expected_wed)

        # 2026-11-21 is Saturday (weekday 5) -> resolves to Wednesday 2026-11-18 00:00:00
        sat = datetime(2026, 11, 21, 12, 0, 0)
        self.assertEqual(compute_live_window_start("week", now=sat), expected_wed)

        # 2026-11-22 is Sunday (weekday 6) -> resolves to Wednesday 2026-11-18 00:00:00
        sun = datetime(2026, 11, 22, 8, 15, 0)
        self.assertEqual(compute_live_window_start("week", now=sun), expected_wed)

        # 2026-11-23 is Monday (weekday 0) -> resolves to preceding Wednesday 2026-11-18 00:00:00
        mon = datetime(2026, 11, 23, 10, 0, 0)
        self.assertEqual(compute_live_window_start("week", now=mon), expected_wed)

        # 2026-11-24 is Tuesday (weekday 1) -> resolves to preceding Wednesday 2026-11-18 00:00:00
        tue = datetime(2026, 11, 24, 23, 59, 59)
        self.assertEqual(compute_live_window_start("week", now=tue), expected_wed)

        # 2026-11-25 is the next Wednesday -> resolves to its own 00:00:00
        next_wed = datetime(2026, 11, 25, 0, 0, 1)
        self.assertEqual(
            compute_live_window_start("week", now=next_wed),
            datetime(2026, 11, 25, 0, 0, 0),
        )

    def test_invalid_range_mode_raises(self):
        with self.assertRaises(ValueError):
            compute_live_window_start("monthly")


class TestStepLevelSlicing(unittest.TestCase):
    """Test read-only, non-mutating session and step slicing."""

    def test_pass_through_when_started_within_window(self):
        start_dt = datetime(2026, 11, 20, 9, 0, 0)
        st = _make_step(1, datetime(2026, 11, 20, 10, 0, 0))
        sess = _make_session("s-in-window", steps=[st])

        sliced = filter_sessions_since([sess], start_dt)
        self.assertEqual(len(sliced), 1)
        # Identity check: zero-copy pass-through
        self.assertIs(sliced[0], sess)

    def test_fast_skip_when_ended_before_window(self):
        start_dt = datetime(2026, 11, 20, 9, 0, 0)
        st = _make_step(1, datetime(2026, 11, 20, 8, 30, 0))
        sess = _make_session("s-before-window", steps=[st])

        sliced = filter_sessions_since([sess], start_dt)
        self.assertEqual(len(sliced), 0)

    def test_boundary_crossing_session_slicing_and_immutability(self):
        start_dt = datetime(2026, 11, 20, 9, 0, 0)

        st1 = _make_step(1, datetime(2026, 11, 20, 8, 15, 0), model="gemini-3.8-flash", prompt_tokens=1000, output_tokens=100)
        st2 = _make_step(2, datetime(2026, 11, 20, 9, 30, 0), model="claude-3-5-sonnet", prompt_tokens=4000, output_tokens=400)
        st3 = _make_step(3, datetime(2026, 11, 20, 10, 0, 0), model="gemini-3.8-flash", prompt_tokens=1500, output_tokens=150)

        orig_sess = _make_session("s-cross", steps=[st1, st2, st3])
        orig_steps_count = len(orig_sess.steps)
        orig_prompt_tokens = orig_sess.usage.prompt_tokens

        sliced = filter_sessions_since([orig_sess], start_dt)
        self.assertEqual(len(sliced), 1)
        new_sess = sliced[0]

        # 1. Must be a newly constructed instance, not mutating the original
        self.assertIsNot(new_sess, orig_sess)
        self.assertEqual(len(orig_sess.steps), orig_steps_count)
        self.assertEqual(orig_sess.usage.prompt_tokens, orig_prompt_tokens)

        # 2. Recomputed step count & steps
        self.assertEqual(new_sess.step_count, 2)
        self.assertEqual(len(new_sess.steps), 2)
        self.assertEqual(new_sess.steps[0].step_idx, 2)
        self.assertEqual(new_sess.steps[1].step_idx, 3)

        # 3. Start time & end time recomputed to window bounds
        self.assertEqual(new_sess.start_time, datetime(2026, 11, 20, 9, 30, 0))
        self.assertEqual(new_sess.end_time, datetime(2026, 11, 20, 10, 0, 0))

        # 4. Usage recomputed from filtered steps only (st2 + st3: 4000 + 1500 = 5500 prompt)
        self.assertEqual(new_sess.usage.prompt_tokens, 5500)
        self.assertEqual(new_sess.usage.output_tokens, 550)

        # 5. Models sorted descending by volume in filtered steps
        # claude: 4000 + 200 + 400 = 4600; gemini: 1500 + 200 + 150 = 1850
        self.assertEqual(new_sess.models, ["claude-3-5-sonnet", "gemini-3.8-flash"])

        # 6. Cached property cost on new instance accurately reflects filtered steps
        expected_cost = st2.cost.total + st3.cost.total
        self.assertAlmostEqual(new_sess.cost.total, expected_cost, places=6)

    def test_stepless_session_handling(self):
        start_dt = datetime(2026, 11, 20, 9, 0, 0)
        s_after = _make_session("s-stepless-after", steps=[], start_time=datetime(2026, 11, 20, 9, 30, 0))
        s_before = _make_session("s-stepless-before", steps=[], start_time=datetime(2026, 11, 20, 8, 30, 0))

        sliced = filter_sessions_since([s_after, s_before], start_dt)
        self.assertEqual(len(sliced), 1)
        self.assertEqual(sliced[0].session_id, "s-stepless-after")

    def test_all_mode_returns_all_sessions(self):
        s1 = _make_session("s1", start_time=datetime(2026, 1, 1))
        s2 = _make_session("s2", start_time=datetime(2026, 11, 20))
        sliced = filter_sessions_since([s1, s2], None)
        self.assertEqual(len(sliced), 2)


class TestLiveDashboardRender(unittest.TestCase):
    """Test text rendering, headers, recent sessions table, and empty window behavior."""

    def test_empty_window_renders_zero_kpi_cards_and_waiting_hint(self):
        """When no sessions fall into the active window, render 0-value cards and waiting hint."""
        # Sessions exist on disk, but only from yesterday
        yesterday_step = _make_step(1, datetime.now() - timedelta(days=1))
        sess = _make_session("s-old", steps=[yesterday_step])

        args = argparse.Namespace(
            interval=1,
            json=False,
            csv=False,
            today=True,
            week=False,
            all=False,
        )

        out_buf = io.StringIO()
        err_buf = io.StringIO()
        with mock.patch("agy_usage.cli.get_sessions", return_value=[sess]), \
             mock.patch("agy_usage.cli.time.sleep", side_effect=KeyboardInterrupt()), \
             redirect_stdout(out_buf), redirect_stderr(err_buf):
            rc = cmd_live(args)

        self.assertEqual(rc, 0)
        out = strip_ansi(out_buf.getvalue())
        self.assertIn("ANTIGRAVITY LIVE MONITOR", out)
        self.assertIn("Today:", out)
        self.assertIn("TOTAL PROCESSED", out)
        self.assertIn("0 in / 0 out", out)
        self.assertIn("0 sessions", out)
        self.assertIn("No activity recorded in this window yet. Waiting for new prompts...", out)
        # Heatmap must NOT be rendered in live mode
        self.assertNotIn("ACTIVITY HEATMAP", out)
        self.assertEqual(err_buf.getvalue(), "")

    def test_recent_sessions_table_limit_10_and_cost_column(self):
        """Live dashboard shows up to 10 sessions, with Est. Cost before Started, sorted by end_time desc."""
        base_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        sessions = []
        for i in range(12):
            t_start = base_time + timedelta(minutes=i * 2)
            t_end = base_time + timedelta(minutes=i * 2 + 1)
            st = _make_step(1, t_end, prompt_tokens=1000 * (i + 1), output_tokens=100 * (i + 1))
            s = _make_session(
                f"sess-{i:02d}",
                steps=[st],
                start_time=t_start,
                end_time=t_end,
                title=f"Session {i}",
            )
            sessions.append(s)

        args = argparse.Namespace(
            interval=1,
            json=False,
            csv=False,
            today=True,
            week=False,
            all=False,
        )

        out_buf = io.StringIO()
        with mock.patch("agy_usage.cli.get_sessions", return_value=sessions), \
             mock.patch("agy_usage.cli.time.sleep", side_effect=KeyboardInterrupt()), \
             redirect_stdout(out_buf):
            rc = cmd_live(args)

        self.assertEqual(rc, 0)
        raw_out = strip_ansi(out_buf.getvalue())

        # Check table headers: Est. Cost must precede Started
        self.assertIn("Est. Cost", raw_out)
        self.assertIn("Started", raw_out)
        header_idx_cost = raw_out.index("Est. Cost")
        header_idx_started = raw_out.index("Started")
        self.assertLess(header_idx_cost, header_idx_started)

        # Recent sessions limit: 12 sessions passed in, exactly 10 displayed
        self.assertIn("sess-11", raw_out)
        self.assertIn("sess-10", raw_out)
        self.assertIn("sess-02", raw_out)
        # sess-00 and sess-01 should be truncated out by limit=10
        self.assertNotIn("sess-01", raw_out)
        self.assertNotIn("sess-00", raw_out)

        # Check sorting: sess-11 (most recent end_time) appears before sess-10
        pos_11 = raw_out.index("sess-11")
        pos_10 = raw_out.index("sess-10")
        self.assertLess(pos_11, pos_10)

    def test_header_window_labels(self):
        """Verify window label formats for --today, --week, and --all."""
        fixed_dt = datetime(2026, 11, 20, 14, 0, 0)
        sess = _make_session("s1", steps=[_make_step(1, fixed_dt)])

        modes = [
            ("today", "[Today: 2026-11-20 00:00 ~ now]"),
            ("week", "[Week (Wed+): 2026-11-18 00:00 ~ now]"),
            ("all", "[All Time]"),
        ]

        for mode, expected_label in modes:
            with self.subTest(mode=mode):
                args = argparse.Namespace(
                    interval=1,
                    json=False,
                    csv=False,
                    today=(mode == "today"),
                    week=(mode == "week"),
                    all=(mode == "all"),
                )
                out_buf = io.StringIO()
                with mock.patch("agy_usage.cli.get_sessions", return_value=[sess]), \
                     mock.patch("agy_usage.cli.datetime") as mock_dt, \
                     mock.patch("agy_usage.cli.time.sleep", side_effect=KeyboardInterrupt()), \
                     redirect_stdout(out_buf):
                    mock_dt.now.return_value = fixed_dt
                    mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
                    rc = cmd_live(args)

                self.assertEqual(rc, 0)
                out = strip_ansi(out_buf.getvalue())
                self.assertIn(expected_label, out)
                self.assertIn("Last updated: 14:00:00", out)


class TestCliFlags(unittest.TestCase):
    """Test CLI argument parsing and mutual exclusivity of time range flags."""

    def test_live_flags_mutually_exclusive(self):
        args_today = parse_args(["live", "--today"])
        self.assertTrue(args_today.today)
        self.assertFalse(args_today.week)
        self.assertFalse(args_today.all)

        args_week = parse_args(["live", "--week"])
        self.assertFalse(args_week.today)
        self.assertTrue(args_week.week)
        self.assertFalse(args_week.all)

        args_all = parse_args(["live", "--all"])
        self.assertFalse(args_all.today)
        self.assertFalse(args_all.week)
        self.assertTrue(args_all.all)

        with self.assertRaises(SystemExit):
            with redirect_stderr(io.StringIO()):
                parse_args(["live", "--today", "--week"])

        with self.assertRaises(SystemExit):
            with redirect_stderr(io.StringIO()):
                parse_args(["live", "--today", "--all"])

    def test_live_json_mode_respects_time_slicing(self):
        """live --json emits sliced sessions matching the active window."""
        now = datetime.now()
        s_old = _make_session("s-old", steps=[_make_step(1, now - timedelta(days=2))])
        s_today = _make_session("s-today", steps=[_make_step(1, now)])

        args = argparse.Namespace(
            interval=1,
            json=True,
            csv=False,
            today=True,
            week=False,
            all=False,
        )

        out_buf = io.StringIO()
        with mock.patch("agy_usage.cli.get_sessions", return_value=[s_old, s_today]), \
             mock.patch("agy_usage.cli.time.sleep", side_effect=KeyboardInterrupt()), \
             redirect_stdout(out_buf):
            rc = cmd_live(args)

        self.assertEqual(rc, 0)
        lines = [ln.strip() for ln in out_buf.getvalue().splitlines() if ln.strip()]
        self.assertEqual(len(lines), 1)
        data = json.loads(lines[0])
        self.assertEqual(data["summary"]["session_count"], 1)
        self.assertEqual(data["sessions"][0]["session_id"], "s-today")

    def test_live_limit_flag_controls_recent_sessions(self):
        """live -n / --limit controls how many recent sessions are displayed (0 = all)."""
        args_default = parse_args(["live"])
        self.assertEqual(args_default.limit, 10)
        args_custom = parse_args(["live", "-n", "3"])
        self.assertEqual(args_custom.limit, 3)
        args_all = parse_args(["live", "--limit", "0"])
        self.assertEqual(args_all.limit, 0)

        now = datetime.now()
        sessions = [
            _make_session(f"lim-{i:02d}", steps=[_make_step(1, now - timedelta(minutes=i))])
            for i in range(12)
        ]

        # limit=3 shows only 3
        out_3 = io.StringIO()
        args = argparse.Namespace(interval=1, json=False, csv=False, today=True, week=False, all=False, limit=3)
        with mock.patch("agy_usage.cli.get_sessions", return_value=sessions), \
             mock.patch("agy_usage.cli.time.sleep", side_effect=KeyboardInterrupt()), \
             redirect_stdout(out_3):
            cmd_live(args)
        text_3 = out_3.getvalue()
        self.assertIn("lim-00", text_3)
        self.assertIn("lim-02", text_3)
        self.assertNotIn("lim-03", text_3)

        # limit=0 shows all 12
        out_0 = io.StringIO()
        args.limit = 0
        with mock.patch("agy_usage.cli.get_sessions", return_value=sessions), \
             mock.patch("agy_usage.cli.time.sleep", side_effect=KeyboardInterrupt()), \
             redirect_stdout(out_0):
            cmd_live(args)
        text_0 = out_0.getvalue()
        self.assertIn("lim-00", text_0)
        self.assertIn("lim-11", text_0)


class TestSessionWatch(unittest.TestCase):
    """Test session --watch live refreshing and boundary behaviors."""

    def test_session_watch_csv_rejected_exit_2(self):
        err_buf = io.StringIO()
        with redirect_stderr(err_buf):
            rc = cli_main(["session", "--watch", "--csv"])
        self.assertEqual(rc, 2)
        self.assertIn("error: --csv is not supported by '--watch'", err_buf.getvalue())

    def test_session_watch_unknown_session_exit_1(self):
        st = _make_step(1, datetime.now())
        sess = _make_session("valid-sess-1", steps=[st])

        err_buf = io.StringIO()
        with mock.patch("agy_usage.cli.get_sessions", return_value=[sess]), \
             redirect_stderr(err_buf):
            rc = cli_main(["session", "unknown-session-id", "--watch"])

        self.assertEqual(rc, 1)
        self.assertIn("Session 'unknown-session-id' not found.", err_buf.getvalue())

    def test_session_watch_loop_and_interrupt(self):
        now = datetime.now()
        st = _make_step(1, now)
        sess = _make_session("watch-sess-1", steps=[st], title="Watch Target")

        out_buf = io.StringIO()
        with mock.patch("agy_usage.cli.get_sessions", return_value=[sess]), \
             mock.patch("agy_usage.cli.time.sleep", side_effect=KeyboardInterrupt()), \
             redirect_stdout(out_buf):
            rc = cli_main(["session", "watch-sess-1", "--watch", "--interval", "2"])

        self.assertEqual(rc, 0)
        raw_out = strip_ansi(out_buf.getvalue())
        self.assertIn("ANTIGRAVITY SESSION WATCH", raw_out)
        self.assertIn("(every 2s)", raw_out)
        self.assertIn("SESSION TOTALS: watch-sess-1", raw_out)
        self.assertIn("Session watch stopped.", raw_out)


if __name__ == "__main__":
    unittest.main()
