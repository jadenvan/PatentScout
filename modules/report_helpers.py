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


def format_patent_date(date_val) -> str:
    """Convert BigQuery integer date (YYYYMMDD) to human-readable format."""
    import pandas as pd
    if pd.isna(date_val) or date_val is None:
        return "Date unknown"
    try:
        date_int = int(date_val)
        if date_int <= 0:
            return "Date unknown"
        date_str = str(date_int)
        if len(date_str) != 8:
            return str(date_int)
        from datetime import datetime
        dt = datetime.strptime(date_str, '%Y%m%d')
        return dt.strftime('%B %d, %Y')
    except (ValueError, TypeError):
        return str(date_val)


def format_patent_year(date_val) -> str:
    """Short year format for tables."""
    import pandas as pd
    if pd.isna(date_val) or date_val is None:
        return "N/A"
    try:
        val = int(date_val)
        if val <= 0:
            return "N/A"
        return str(val)[:4]
    except (ValueError, TypeError):
        return "N/A"


def group_matches_by_patent(matches: list[dict]) -> list[dict]:
    """
    Group match entries by patent number and return one entry per patent,
    sorted by the best (highest) overall confidence → score.

    Each returned dict has:
      - All fields from the best match entry for that patent
      - ``all_features``: list of dicts with feature_label, similarity_score,
            overall_confidence for every feature matched against this patent
      - ``feature_count``: how many features matched
      - ``best_score``: max similarity_score across all features

    This eliminates the "same patent repeated N times" problem in the
    Prior Art table and enriched claim view.
    """
    from collections import defaultdict

    by_patent: dict[str, list[dict]] = defaultdict(list)
    for m in matches:
        key = m.get("patent_number", m.get("publication_number", "UNKNOWN"))
        by_patent[key].append(m)

    _conf_order = {"HIGH": 0, "MODERATE": 1, "LOW": 2}

    grouped: list[dict] = []
    for pat_num, entries in by_patent.items():
        # Sort entries for this patent: best overall → highest score
        entries_sorted = sorted(
            entries,
            key=lambda e: (
                _conf_order.get(e.get("overall_confidence", e.get("similarity_level", "LOW")), 2),
                -float(e.get("similarity_score", 0)),
            ),
        )
        best = dict(entries_sorted[0])   # copy so we don't mutate original
        best["all_features"] = [
            {
                "feature_label": e.get("feature_label", ""),
                "similarity_score": e.get("similarity_score", 0),
                "overall_confidence": e.get("overall_confidence", e.get("similarity_level", "LOW")),
                "element_text": e.get("element_text", ""),
                "claim_number": e.get("claim_number", ""),
            }
            for e in entries_sorted
        ]
        best["feature_count"] = len(set(e.get("feature_label", "") for e in entries))
        best["best_score"] = max(float(e.get("similarity_score", 0)) for e in entries)
        grouped.append(best)

    # Sort grouped list: best overall confidence → highest score → most features
    grouped.sort(
        key=lambda g: (
            _conf_order.get(g.get("overall_confidence", g.get("similarity_level", "LOW")), 2),
            -g["best_score"],
            -g["feature_count"],
        ),
    )
    return grouped
