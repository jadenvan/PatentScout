"""
Unit tests for modules/report_helpers.py
"""

import unittest

from modules.report_helpers import (
    format_google_patent_url,
    highlight_snippet,
    safe_text_for_pdf,
)


class TestFormatGooglePatentUrl(unittest.TestCase):
    def test_standard_number(self):
        url = format_google_patent_url("US-7479949-B2")
        self.assertEqual(url, "https://patents.google.com/patent/US7479949B2")

    def test_already_clean(self):
        url = format_google_patent_url("US7479949B2")
        self.assertEqual(url, "https://patents.google.com/patent/US7479949B2")

    def test_empty_returns_empty(self):
        self.assertEqual(format_google_patent_url(""), "")
        self.assertEqual(format_google_patent_url(None), "")

    def test_strips_whitespace(self):
        url = format_google_patent_url("  US-12345678-A1  ")
        self.assertEqual(url, "https://patents.google.com/patent/US12345678A1")


class TestHighlightSnippet(unittest.TestCase):
    def test_basic_highlight(self):
        text = "a solar panel connected to a battery"
        result = highlight_snippet(text, ["solar", "battery"])
        self.assertIn("<b>solar</b>", result)
        self.assertIn("<b>battery</b>", result)

    def test_truncation(self):
        text = "x" * 300
        result = highlight_snippet(text, [], max_len=100)
        # 100 chars + "..."
        self.assertEqual(len(result), 103)

    def test_empty_input(self):
        self.assertEqual(highlight_snippet("", ["a"]), "")
        self.assertEqual(highlight_snippet(None, ["a"]), "")

    def test_case_insensitive(self):
        text = "Solar and SOLAR and solar"
        result = highlight_snippet(text, ["solar"])
        self.assertEqual(result.count("<b>"), 3)

    def test_newlines_collapsed(self):
        text = "hello\nworld\n  yes"
        result = highlight_snippet(text, [])
        self.assertNotIn("\n", result)

    def test_empty_terms_ignored(self):
        text = "some text"
        result = highlight_snippet(text, ["", None])
        self.assertEqual(result, "some text")


class TestSafeTextForPdf(unittest.TestCase):
    def test_none_returns_fallback(self):
        self.assertEqual(safe_text_for_pdf(None), "N/A")

    def test_empty_returns_fallback(self):
        self.assertEqual(safe_text_for_pdf(""), "N/A")

    def test_custom_fallback(self):
        self.assertEqual(safe_text_for_pdf("", fallback="—"), "—")

    def test_control_chars_replaced(self):
        result = safe_text_for_pdf("hello\x00world\x01!")
        self.assertEqual(result, "hello world !")

    def test_normal_text_unchanged(self):
        self.assertEqual(safe_text_for_pdf("hello world"), "hello world")


if __name__ == "__main__":
    unittest.main()
