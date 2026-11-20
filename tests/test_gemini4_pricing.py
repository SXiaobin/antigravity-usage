"""Tests for Gemini 4 family Non-Global Region API pricing schedules."""

from datetime import date
import unittest

from agy_usage.models import get_model_rates


class TestGemini4Pricing(unittest.TestCase):
    def test_gemini_4_neon_rates(self):
        self.assertEqual(get_model_rates("gemini-4-neon"), (3.30, 0.165, 16.50))

    def test_gemini_4_hydrogen_rates(self):
        self.assertEqual(get_model_rates("gemini-4-hydrogen"), (1.65, 0.0825, 8.25))

    def test_gemini_4_carbon_schedule(self):
        self.assertEqual(
            get_model_rates("gemini-4-carbon", on_date=date(2026, 11, 30)),
            (2.20, 0.11, 11.00),
        )
        self.assertEqual(
            get_model_rates("gemini-4-carbon", on_date=date(2026, 12, 1)),
            (4.40, 0.22, 22.00),
        )

    def test_family_fallbacks(self):
        self.assertEqual(get_model_rates("gemini-4-hydrogen-medium"), (1.65, 0.0825, 8.25))
        self.assertEqual(get_model_rates("gemini-4-neon-high"), (3.30, 0.165, 16.50))
        self.assertEqual(
            get_model_rates("gemini-4-carbon-cyber-permissive", on_date=date(2026, 12, 1)),
            (4.40, 0.22, 22.00),
        )


if __name__ == "__main__":
    unittest.main()
