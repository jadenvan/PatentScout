"""
Test: Match Deduplication

Assert no duplicate (patent, element) pairs in similarity_results["matches"]
when the chosen dedupe policy is applied.
"""
from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SESSION_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples", "solar_charger_session.json",
)
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "active_config.json",
)


class TestMatchDedup(unittest.TestCase):
    """Verify deduplication invariants on cached session matches."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(SESSION_PATH):
            raise unittest.SkipTest("Cached session not found")
        with open(SESSION_PATH) as f:
            cls.session = json.load(f)

        cls.active_config = None
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH) as f:
                cls.active_config = json.load(f)

    def _get_dedupe_policy(self):
        if self.active_config:
            return self.active_config.get("config", {}).get("dedupe_policy", "patent_element")
        return "patent_element"

    def test_no_duplicate_patent_element_pairs(self):
        """With patent_element policy, no (patent, element) pair should repeat."""
        policy = self._get_dedupe_policy()
        if policy != "patent_element":
            self.skipTest(f"Dedupe policy is '{policy}', not patent_element")

        matches = self.session.get("similarity_results", {}).get("matches", [])
        if not matches:
            self.skipTest("No matches in session data")

        # Apply dedupe
        from tools.experiment_runner import _apply_dedupe
        deduped = _apply_dedupe(matches, "patent_element")

        seen = set()
        for m in deduped:
            key = (m.get("patent_number", ""), m.get("element_id", ""))
            self.assertNotIn(
                key, seen,
                f"Duplicate (patent, element) found: {key}",
            )
            seen.add(key)

    def test_dedupe_keeps_highest_score(self):
        """Deduplication should keep the highest-scoring match per key."""
        matches = self.session.get("similarity_results", {}).get("matches", [])
        if not matches:
            self.skipTest("No matches in session data")

        from tools.experiment_runner import _apply_dedupe
        deduped = _apply_dedupe(matches, "patent_element")

        # Build lookup to verify
        best_scores: dict[tuple, float] = {}
        for m in matches:
            key = (m.get("patent_number", ""), m.get("element_id", ""))
            best_scores[key] = max(best_scores.get(key, 0), m.get("similarity_score", 0))

        for m in deduped:
            key = (m.get("patent_number", ""), m.get("element_id", ""))
            if key in best_scores:
                self.assertAlmostEqual(
                    m["similarity_score"],
                    best_scores[key],
                    places=3,
                    msg=f"Deduped match for {key} does not have highest score",
                )


if __name__ == "__main__":
    unittest.main()
