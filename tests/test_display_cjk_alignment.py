"""Tests for East Asian wide-character display width and table alignment."""

import unittest

from agy_usage.display import display_width, render_table, strip_ansi, truncate_ansi


class TestDisplayCjkAlignment(unittest.TestCase):
    def test_display_width_counts_cjk_as_two(self):
        self.assertEqual(display_width("abc"), 3)
        self.assertEqual(display_width("中文"), 4)
        self.assertEqual(display_width("中文abc"), 7)

    def test_render_table_aligns_mixed_cjk_and_ascii_rows(self):
        table = render_table(
            ["Title", "Model"],
            [
                ["测试标题", "gemini-4-hydrogen"],
                ["plain-ascii", "gemini-3.8-flash"],
            ],
        )
        lines = [strip_ansi(line) for line in table.splitlines()]
        row_widths = [display_width(line) for line in lines]
        self.assertEqual(len(set(row_widths)), 1)

    def test_truncate_ansi_respects_cjk_width(self):
        truncated = truncate_ansi("中文标题很长", 6)
        self.assertLessEqual(display_width(truncated), 6)
        self.assertTrue(truncated.endswith("…"))


if __name__ == "__main__":
    unittest.main()
