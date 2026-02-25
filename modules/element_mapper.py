"""
PatentScout — Element Mapper Module  (Phase 6)

Takes the HIGH / MODERATE similarity matches produced by EmbeddingEngine and
runs a second, contextual analysis layer via Gemini.  Results are merged to
produce the final two-layer comparison matrix.

API surface
-----------
engine = ElementMapper(gemini_client)
enriched = engine.analyze_matches(similarity_results)  # list[dict]
"""

from __future__ import annotations

import json
import logging
import time
from typing import Generator

from config.prompts import CLAIM_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema defaults — used when Gemini omits a field
# ---------------------------------------------------------------------------
_FIELD_DEFAULTS: dict = {
    "claim_element_explanation": "Explanation unavailable.",
    "similarity_assessment": "Assessment unavailable.",
    "key_distinctions": ["Unable to determine from automated analysis."],
    "cannot_determine": "Full legal analysis requires professional patent review.",
    "confidence": "LOW",
}

# Map level strings to numeric for averaging
_LEVEL_NUM = {"HIGH": 3, "MODERATE": 2, "LOW": 1}


def chunk_list(lst: list, n: int) -> Generator[list, None, None]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _validate_result(raw: dict) -> dict:
    """
    Ensure a Gemini-returned analysis object has all required fields.
    Missing fields are filled with safe defaults.
    """
    result: dict = {}
    for field, default in _FIELD_DEFAULTS.items():
        val = raw.get(field, default)
        # key_distinctions must be a non-empty list
        if field == "key_distinctions":
            if not isinstance(val, list) or not val:
                val = [str(val)] if val else default
        # confidence must be one of the three levels
        elif field == "confidence":
            if str(val).upper() not in ("HIGH", "MODERATE", "LOW"):
                val = "LOW"
            else:
                val = str(val).upper()
        result[field] = val
    return result


def _parse_gemini_response(text: str, expected: int) -> list[dict]:
    """
    Parse Gemini text response into a list of analysis dicts.

    Strategy:
    1. Try to parse the entire text as a JSON array.
    2. If that fails, extract the first JSON array substring.
    3. If the array has the wrong length, attempt individual object extraction.
    4. Return a list of defaults for any missing positions.
    """
    text = text.strip()

    # --- attempt 1: full parse -------------------------------------------------
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [_validate_result(r) for r in parsed]
        if isinstance(parsed, dict):
            return [_validate_result(parsed)]
    except json.JSONDecodeError:
        pass

    # --- attempt 2: find the outermost [...] -----------------------------------
    start = text.find("[")
    end   = text.rfind("]")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, list):
                return [_validate_result(r) for r in parsed]
        except json.JSONDecodeError:
            pass

    # --- attempt 3: extract individual {...} objects in order -----------------
    results: list[dict] = []
    cursor = 0
    while cursor < len(text):
        obj_start = text.find("{", cursor)
        if obj_start == -1:
            break
        # find matching closing brace
        depth = 0
        for k, ch in enumerate(text[obj_start:], obj_start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[obj_start : k + 1])
                    results.append(_validate_result(obj))
                except json.JSONDecodeError:
                    pass
                cursor = k + 1
                break
        else:
            break

    if results:
        return results

    # --- fallback: return defaults for all expected positions -----------------
    logger.warning("Could not parse Gemini response; using defaults for %d pairs.", expected)
    return [dict(_FIELD_DEFAULTS) for _ in range(expected)]


class ElementMapper:
    """
    Enriches HIGH / MODERATE embedding similarity matches by running each
    feature-element pair through Gemini for contextual analysis, then merges
    the two layers to detect divergences and compute an overall confidence
    score.
    """

    def __init__(self, gemini_client) -> None:
        """
        Args:
            gemini_client: An initialised ``google.genai.Client`` instance.
        """
        self.gemini = gemini_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_matches(self, similarity_results: dict) -> list[dict]:
        """
        Run Gemini contextual analysis on HIGH / MODERATE embedding matches.

        Args:
            similarity_results: Dict returned by
                ``EmbeddingEngine.compute_similarity_matrix``.

        Returns:
            List of enriched match dicts, each containing all original
            embedding fields plus Gemini fields, divergence detection, and
            an ``overall_confidence`` rating.
        """
        if not similarity_results or not similarity_results.get("matches"):
            return []

        # Filter and sort ------------------------------------------------
        high_mod = [
            m
            for m in similarity_results["matches"]
            if m["similarity_level"] in ("HIGH", "MODERATE")
        ]
        high_mod.sort(key=lambda x: x["similarity_score"], reverse=True)
        pairs_to_analyze = high_mod[:15]

        if not pairs_to_analyze:
            return []

        enriched: list[dict] = []
        batches = list(chunk_list(pairs_to_analyze, 3))

        for batch_idx, batch in enumerate(batches):
            # -- Build the batch prompt text --------------------------------
            pair_lines: list[str] = []
            for i, pair in enumerate(batch, 1):
                pair_lines.append(
                    f"PAIR {i}:\n"
                    f"Feature Label: {pair['feature_label']}\n"
                    f"Feature Description: {pair['feature_description']}\n"
                    f"Claim Element: {pair['element_text']}\n"
                    f"Source: US Patent {pair['patent_number']}, "
                    f"Claim {pair['claim_number']}"
                )

            pairs_text = "\n\n".join(pair_lines)
            prompt = CLAIM_ANALYSIS_PROMPT.format(
                n_pairs=len(batch),
                pairs_text=pairs_text,
            )

            # -- Call Gemini ------------------------------------------------
            batch_results: list[dict] = []
            try:
                response = self.gemini.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
                raw_text = response.text or ""
                batch_results = _parse_gemini_response(raw_text, len(batch))
            except Exception as exc:
                logger.warning(
                    "Gemini batch %d/%d failed (%s); using embedding-only data.",
                    batch_idx + 1,
                    len(batches),
                    exc,
                )
                batch_results = [dict(_FIELD_DEFAULTS) for _ in batch]

            # Pad if Gemini returned fewer objects than expected
            while len(batch_results) < len(batch):
                batch_results.append(dict(_FIELD_DEFAULTS))

            # -- Merge & detect divergence ----------------------------------
            for pair, gemini_result in zip(batch, batch_results):
                embedding_level = pair["similarity_level"]
                gemini_conf     = gemini_result["confidence"]

                divergence_flag = False
                divergence_note = ""

                if embedding_level == "HIGH" and gemini_conf == "LOW":
                    divergence_flag = True
                    divergence_note = (
                        "High textual similarity but contextual analysis identified "
                        "potential technical distinctions — manual review recommended."
                    )
                elif embedding_level == "MODERATE" and gemini_conf == "HIGH":
                    divergence_flag = True
                    divergence_note = (
                        "Moderate textual similarity but contextual analysis found "
                        "strong conceptual overlap — warrants closer examination."
                    )

                # Overall confidence
                avg = (
                    _LEVEL_NUM.get(embedding_level, 1)
                    + _LEVEL_NUM.get(gemini_conf, 1)
                ) / 2
                if divergence_flag:
                    overall = "LOW"
                elif avg >= 2.5:
                    overall = "HIGH"
                elif avg >= 1.5:
                    overall = "MODERATE"
                else:
                    overall = "LOW"

                enriched.append(
                    {
                        **pair,
                        "gemini_explanation": gemini_result[
                            "claim_element_explanation"
                        ],
                        "gemini_assessment": gemini_result["similarity_assessment"],
                        "key_distinctions": gemini_result["key_distinctions"],
                        "cannot_determine": gemini_result["cannot_determine"],
                        "gemini_confidence": gemini_conf,
                        "divergence_flag": divergence_flag,
                        "divergence_note": divergence_note,
                        "overall_confidence": overall,
                    }
                )

            # Rate-limiting: pause between batches
            if batch_idx < len(batches) - 1:
                time.sleep(1)

        return enriched
