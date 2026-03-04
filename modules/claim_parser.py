"""
PatentScout — Claim Parser Module

Parses raw patent claim text (retrieved from BigQuery) into structured dicts
ready for element-level comparison.  Uses a cascade of regex strategies with
an optional Gemini fallback for hard-to-parse claims.

Key design decisions
--------------------
* Claims text from BigQuery is a single string containing all claims.
* Multiple UNNEST of nested BigQuery fields can produce a cartesian product,
  so claims arrive as one blob we must split here.
* Up to 3 claims are batched per Gemini call to reduce latency.
* Semicolons inside parentheses are NOT used as element delimiters.
* Element count is capped at 15 per claim.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional

import pandas as pd

from config.prompts import CLAIM_PARSING_PROMPT

logger = logging.getLogger(__name__)


# Constants

_TRANSITIONAL_PHRASES = [
    "consisting essentially of",
    "consisting of",
    "comprising",
    "including",
    "characterized by",
    "wherein the improvement comprises",
    "having",
    "which comprises",
    "that comprises",
]

# Dependent-claim reference pattern
_DEP_PATTERN = re.compile(
    r"(?:of|in|to|according\s+to|as\s+(?:set\s+forth|defined|claimed|described)\s+in)"
    r"\s+claim\s+\d+",
    re.IGNORECASE,
)

_MAX_ELEMENTS = 15
_BATCH_SIZE   = 3     # max claims per Gemini call
_GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
]



# Helpers

def _sanitise(text: str) -> str:
    """Normalise encoding, collapse whitespace, strip OCR garbage."""
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    # Remove long runs of non-alphanumeric chars that look like OCR noise
    text = re.sub(r"[^\x20-\x7E\n]{2,}", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_claims(text: str) -> list[tuple[int, str]]:
    """
    Split a raw claims blob into ``[(claim_number, claim_text), ...]``.

    Tries three progressively looser split patterns; returns whichever
    produces the most plausible results (each segment ≥ 20 characters).
    """
    candidates: list[list[tuple[int, str]]] = []

    # Pattern A: newline + digit(s) + period
    for pat in (
        r"\n\s*(\d{1,3})\.\s+",
        r"(?<=[.;])\s*(\d{1,3})\.\s+",
        r"\b(\d{1,3})\.\s+",
    ):
        parts = re.split(pat, text)
        # parts: [pre, num, text, num, text, ...]
        result: list[tuple[int, str]] = []
        i = 1
        while i + 1 < len(parts):
            try:
                num = int(parts[i])
                body = parts[i + 1].strip()
                if len(body) >= 20:
                    result.append((num, body))
            except (ValueError, IndexError):
                pass
            i += 2
        if result:
            candidates.append(result)

    if not candidates:
        return []
    # Pick the split that produced the most claims with sequential numbering
    def _score(c: list[tuple[int, str]]) -> int:
        nums = [n for n, _ in c]
        sequential = sum(1 for a, b in zip(nums, nums[1:]) if b == a + 1)
        return sequential * 10 + len(c)

    return max(candidates, key=_score)


def _find_transition(text: str) -> tuple[str, int]:
    """
    Return (transitional_phrase, index_in_text) for the first match.
    Returns ('', -1) if none found.
    """
    lower = text.lower()
    best = ("", -1)
    for phrase in _TRANSITIONAL_PHRASES:
        idx = lower.find(phrase)
        if idx != -1:
            if best[1] == -1 or idx < best[1]:
                best = (phrase, idx)
    return best


def _split_elements(body: str) -> list[str]:
    """
    Split a claim body on semicolons (not inside parens) and
    wherein/whereby/such-that sub-clauses.
    """
    # Split on semicolons not inside parentheses
    parts = re.split(r";\s*(?![^()]*\))", body)
    elements: list[str] = []
    for part in parts:
        # Further split on "wherein"/"whereby"/"such that" if long enough
        sub = re.split(r"\b(?:wherein|whereby|such\s+that)\b", part, flags=re.IGNORECASE)
        for s in sub:
            s = s.strip().rstrip(";., ")
            if len(s) > 10:
                elements.append(s)
    return elements[:_MAX_ELEMENTS]


def _make_element_id(claim_num: int, idx: int) -> str:
    """Return element id like '1a', '1b', '10a', '10b' … '10p'."""
    letter = chr(ord("a") + idx) if idx < 26 else str(idx)
    return f"{claim_num}{letter}"



# Main class


class ClaimParser:
    """
    Parses raw BigQuery patent claims text into structured dicts.

    Usage::

        parser = ClaimParser(gemini_client=genai.Client(api_key="..."))
        result = parser.parse_claims("US-7479949-B2", raw_text)
        results = parser.parse_all(detail_df, max_patents=20)
    """

    def __init__(self, gemini_client=None) -> None:
        self.gemini = gemini_client

    # Public: single patent

    def parse_claims(self, patent_number: str, raw_claims: str) -> dict:
        """
        Parse all claims from *raw_claims* and return a structured dict.

        Returns
        -------
        dict with keys:
            patent_number, total_claims_found, independent_claims,
            parsing_method, parsing_confidence
        """
        if not raw_claims or len(str(raw_claims).strip()) < 30:
            return self._empty_result(patent_number, reason="no_claims_text")

        text = _sanitise(str(raw_claims))

        # Very short text → probably a design patent with no written claims
        if len(text) < 100:
            return self._empty_result(patent_number, reason="design_patent")

        split_claims = _split_claims(text)
        if not split_claims:
            return self._empty_result(patent_number, reason="parse_failed")

        independent: list[dict] = []
        gemini_used = False

        for claim_num, claim_text in split_claims:
            # Is it a dependent claim?
            if _DEP_PATTERN.search(claim_text):
                continue

            parsed = self._parse_single_claim(claim_num, claim_text)

            # Gemini fallback when regex yields < 2 elements
            if len(parsed["elements"]) < 2 and self.gemini:
                gemini_result = self._gemini_parse_batch([(claim_num, claim_text)])
                if gemini_result:
                    parsed = gemini_result[0]
                    gemini_used = True

            independent.append(parsed)

        if not independent:
            return self._empty_result(patent_number, reason="no_independent_claims")

        # Confidence scoring
        avg_elements = sum(len(c["elements"]) for c in independent) / len(independent)
        if avg_elements >= 3 and len(independent) >= 2 and not gemini_used:
            confidence = "HIGH"
        elif avg_elements >= 2:
            confidence = "MODERATE"
        else:
            confidence = "LOW"

        method = "gemini_fallback" if gemini_used else "regex"
        if gemini_used and len(independent) > 1:
            method = "mixed"

        return {
            "patent_number": patent_number,
            "total_claims_found": len(split_claims),
            "independent_claims": independent,
            "parsing_method": method,
            "parsing_confidence": confidence,
        }

    # Public: batch over DataFrame

    def parse_all(
        self,
        detail_df: pd.DataFrame,
        max_patents: int = 20,
    ) -> dict:
        """
        Parse claims for the top *max_patents* rows of *detail_df*.

        Returns
        -------
        dict with keys:
            results   — list of parsed claim dicts (one per patent)
            summary   — {attempted, successful, skipped, failed}
        """
        if detail_df is None or detail_df.empty:
            return {"results": [], "summary": {"attempted": 0, "successful": 0,
                                                "skipped": 0, "failed": 0}}

        top = detail_df.head(max_patents)
        results: list[dict] = []
        attempted = skipped = failed = 0

        for _, row in top.iterrows():
            pub_num = str(row.get("publication_number", "UNKNOWN"))
            claims_text = row.get("claims_text", None)

            # Skip rows with no claims text
            if not claims_text or str(claims_text).strip() in ("", "nan", "None"):
                skipped += 1
                continue

            attempted += 1
            try:
                parsed = self.parse_claims(pub_num, str(claims_text))
                if parsed.get("independent_claims"):
                    results.append(parsed)
                else:
                    failed += 1
            except Exception as exc:
                logger.warning("claim parsing failed for %s: %s", pub_num, exc)
                failed += 1

        return {
            "results": results,
            "summary": {
                "attempted": attempted,
                "successful": len(results),
                "skipped": skipped,
                "failed": failed,
            },
        }

    # Internal: single-claim regex parsing

    def _parse_single_claim(self, claim_num: int, text: str) -> dict:
        transition, idx = _find_transition(text)
        if idx == -1:
            # No transitional phrase — treat entire text as preamble, no elements
            return {
                "claim_number": claim_num,
                "full_text": text,
                "preamble": text.strip(),
                "transitional_phrase": "",
                "elements": [],
                "plain_english": "",
            }

        preamble = text[:idx].strip()
        body = text[idx + len(transition):].strip()
        elements_raw = _split_elements(body)
        elements = [
            {"id": _make_element_id(claim_num, i), "text": e}
            for i, e in enumerate(elements_raw)
        ]

        # Deduplicate elements by normalised text
        _seen_el: set[str] = set()
        _unique_el: list[dict] = []
        for el in elements:
            norm = el["text"].strip().lower()
            if norm not in _seen_el:
                _seen_el.add(norm)
                _unique_el.append(el)
        elements = _unique_el

        return {
            "claim_number": claim_num,
            "full_text": text,
            "preamble": preamble,
            "transitional_phrase": transition,
            "elements": elements,
            "plain_english": "",
        }

    # Internal: Gemini batch parsing

    def _gemini_parse_batch(
        self, claims: list[tuple[int, str]]
    ) -> list[dict]:
        """
        Send up to _BATCH_SIZE claims to Gemini and return parsed dicts.
        Returns empty list on any failure.
        """
        if not self.gemini:
            return []

        batch = claims[:_BATCH_SIZE]
        delimited = "\n".join(
            f"===CLAIM START===\n{num}. {text}\n===CLAIM END==="
            for num, text in batch
        )
        prompt = CLAIM_PARSING_PROMPT + f"\n\nCLAIMS TO PARSE:\n{delimited}"

        last_exc: Exception = RuntimeError("no models tried")
        for model in _GEMINI_MODELS:
            try:
                resp = self.gemini.models.generate_content(
                    model=model,
                    contents=[prompt],
                )
                time.sleep(0.5)
                raw = resp.text.strip()
                # Strip markdown fences if present
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
                parsed_list = json.loads(raw)
                if not isinstance(parsed_list, list):
                    continue

                results: list[dict] = []
                for item in parsed_list:
                    cnum = item.get("claim_number", 0)
                    elems = item.get("elements", [])[:_MAX_ELEMENTS]
                    results.append({
                        "claim_number": cnum,
                        "full_text": next(
                            (t for n, t in batch if n == cnum), ""
                        ),
                        "preamble": item.get("preamble", ""),
                        "transitional_phrase": item.get("transitional_phrase", ""),
                        "elements": elems,
                        "plain_english": item.get("plain_english", ""),
                    })
                return results
            except Exception as exc:
                last_exc = exc
                continue

        logger.warning("Gemini claim parsing failed: %s", last_exc)
        return []

    # Internal: empty result helper

    @staticmethod
    def _empty_result(patent_number: str, reason: str = "") -> dict:
        return {
            "patent_number": patent_number,
            "total_claims_found": 0,
            "independent_claims": [],
            "parsing_method": "skipped",
            "parsing_confidence": "LOW",
            "skip_reason": reason,
        }
