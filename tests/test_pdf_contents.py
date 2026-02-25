"""
Smoke test for PatentScout PDF report generation.

Loads the cached solar charger session, generates a PDF via ReportGenerator,
and validates basic structural expectations (file size, key text presence).
"""

import json
import os
import re
import sys
import tempfile
import unittest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.report_generator import ReportGenerator

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


SESSION_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples",
    "solar_charger_session.json",
)


def _load_session() -> dict:
    if not os.path.exists(SESSION_PATH):  # pragma: no cover
        raise unittest.SkipTest(f"Session file not found: {SESSION_PATH}")
    with open(SESSION_PATH, "r") as f:
        return json.load(f)


class TestPdfContents(unittest.TestCase):
    """Smoke tests for the generated PDF report."""

    @classmethod
    def setUpClass(cls):
        cls.session = _load_session()
        rg = ReportGenerator()
        cls.pdf_bytes = rg.generate(cls.session)

        # Write to a temp file for inspection
        cls.tmp_dir = tempfile.mkdtemp(prefix="patentscout_test_")
        cls.pdf_path = os.path.join(cls.tmp_dir, "test_report.pdf")
        with open(cls.pdf_path, "wb") as f:
            f.write(cls.pdf_bytes)

        # Extract text using pdfplumber if available
        cls._full_text = None
        if HAS_PDFPLUMBER:
            with pdfplumber.open(cls.pdf_path) as pdf:
                cls._full_text = "\n".join(
                    (page.extract_text() or "") for page in pdf.pages
                )

    def _pdf_text(self) -> str:
        """Return extracted text from the PDF."""
        if self._full_text is not None:
            return self._full_text
        # Fallback: simple binary decode
        return self.pdf_bytes.decode("latin-1", errors="replace")

    # ------------------------------------------------------------------ #
    # File-size checks                                                     #
    # ------------------------------------------------------------------ #
    def test_pdf_exists_and_not_empty(self):
        self.assertTrue(os.path.exists(self.pdf_path))
        size = os.path.getsize(self.pdf_path)
        self.assertGreater(size, 10_000, "PDF too small — likely corrupt")

    def test_pdf_under_5mb(self):
        size = os.path.getsize(self.pdf_path)
        self.assertLess(size, 5_000_000, "PDF exceeds 5 MB limit")

    # ------------------------------------------------------------------ #
    # Content checks                                                       #
    # ------------------------------------------------------------------ #
    def test_prior_art_table_has_patents(self):
        text = self._pdf_text()
        patents = re.findall(r"US[\-]?\d{5,}", text)
        self.assertGreaterEqual(
            len(patents), 5,
            f"Expected >=5 patent numbers in PDF; found {len(patents)}",
        )

    def test_match_detail_has_feature_label(self):
        text = self._pdf_text()
        self.assertIn("Feature", text, "Missing 'Feature' label in match details")

    def test_match_detail_has_claim_element(self):
        text = self._pdf_text()
        self.assertIn("Claim Element", text, "Missing 'Claim Element' label in match details")

    def test_has_table_of_contents(self):
        text = self._pdf_text()
        self.assertIn("Table of Contents", text)

    def test_has_page_numbers(self):
        text = self._pdf_text()
        self.assertIn("Page", text)

    def test_has_recommendation_text(self):
        text = self._pdf_text()
        self.assertIn("Recommendation", text)

    def test_has_embedding_similarity(self):
        text = self._pdf_text()
        self.assertIn("Embedding Similarity", text)


if __name__ == "__main__":
    unittest.main()
