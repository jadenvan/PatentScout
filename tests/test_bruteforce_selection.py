"""
Test: Brute-Force Selection

Assert that:
1. config/active_config.json exists and contains chosen_trial_id.
2. A reports/bruteforce_run_*.json exists and contains chosen_trial_id.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "active_config.json")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")


class TestBruteforceSelection(unittest.TestCase):
    """Verify experiment runner produced proper selection artifacts."""

    def test_active_config_exists(self):
        """config/active_config.json must exist after a successful run."""
        self.assertTrue(
            os.path.exists(CONFIG_PATH),
            f"active_config.json not found at {CONFIG_PATH}",
        )

    def test_active_config_has_chosen_trial(self):
        """active_config.json must contain a chosen_trial_id."""
        if not os.path.exists(CONFIG_PATH):
            self.skipTest("active_config.json not found")
        with open(CONFIG_PATH) as f:
            data = json.load(f)
        self.assertIn("chosen_trial_id", data)
        self.assertTrue(data["chosen_trial_id"], "chosen_trial_id is empty")

    def test_active_config_has_config_and_metrics(self):
        """active_config.json must have config and metrics sections."""
        if not os.path.exists(CONFIG_PATH):
            self.skipTest("active_config.json not found")
        with open(CONFIG_PATH) as f:
            data = json.load(f)
        self.assertIn("config", data)
        self.assertIn("metrics", data)
        self.assertIn("thresholds", data)

    def test_bruteforce_report_exists(self):
        """At least one reports/bruteforce_run_*.json must exist."""
        pattern = os.path.join(REPORTS_DIR, "bruteforce_run_*.json")
        matches = glob.glob(pattern)
        self.assertGreater(
            len(matches), 0,
            "No bruteforce_run_*.json found in reports/",
        )

    def test_bruteforce_report_has_chosen_trial(self):
        """The latest bruteforce report must contain chosen_trial_id."""
        pattern = os.path.join(REPORTS_DIR, "bruteforce_run_*.json")
        matches = sorted(glob.glob(pattern))
        if not matches:
            self.skipTest("No bruteforce report found")
        with open(matches[-1]) as f:
            data = json.load(f)
        self.assertIn("chosen_trial_id", data)
        self.assertTrue(data["chosen_trial_id"], "chosen_trial_id is empty in report")

    def test_bruteforce_report_has_trials(self):
        """The report must contain trial data."""
        pattern = os.path.join(REPORTS_DIR, "bruteforce_run_*.json")
        matches = sorted(glob.glob(pattern))
        if not matches:
            self.skipTest("No bruteforce report found")
        with open(matches[-1]) as f:
            data = json.load(f)
        self.assertIn("trials", data)
        self.assertGreater(len(data["trials"]), 0, "No trials in report")


if __name__ == "__main__":
    unittest.main()
