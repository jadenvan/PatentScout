"""
PatentScout — Report Helpers

Utility functions for formatting patent URLs, highlighting text snippets,
and sanitising strings for ReportLab PDF generation.
"""

from __future__ import annotations

import re
from urllib.parse import quote_plus


def format_google_patent_url(publication_number: str) -> str:
    """Convert a publication number like 'US-7479949-B2' to a Google Patents URL."""
    if not publication_number:
        return ""
    clean = publication_number.replace("-", "").strip()
    return f"https://patents.google.com/patent/{quote_plus(clean)}"


def highlight_snippet(
    element_text: str,
    primary_terms: list[str],
    max_len: int = 160,
) -> str:
    """
    Return a short snippet of *element_text* with *primary_terms* wrapped
    in ``<b>`` tags suitable for ReportLab Paragraph markup.
    """
    if not element_text:
        return ""
    s = element_text.replace("\n", " ").strip()
    s = " ".join(s.split())  # collapse whitespace
    snippet = s[:max_len]
    if len(s) > max_len:
        snippet += "..."
    for term in sorted((t for t in set(primary_terms) if t), key=len, reverse=True):
        if not term:
            continue
        try:
            snippet = re.sub(
                r"(?i)(" + re.escape(term) + r")",
                r"<b>\1</b>",
                snippet,
            )
        except re.error:
            continue
    return snippet


def safe_text_for_pdf(s: str | None, fallback: str = "N/A") -> str:
    """
    Sanitise a string for use in ReportLab Paragraphs.

    Replaces control characters (below space) with a space and returns
    *fallback* when the input is empty/None.
    """
    if not s:
        return fallback
    return "".join(ch if ch >= " " else " " for ch in str(s))
