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


# Schema defaults — used when Gemini omits a field
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
    logger.warning("could not parse Gemini response; using defaults for %d pairs.", expected)
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

    # Public API

    def analyze_matches(
        self,
        similarity_results: dict,
        invention_description: str = "",
        detail_patents=None,
    ) -> list[dict]:
        """
        Run Gemini contextual analysis on HIGH / MODERATE embedding matches.

        Args:
            similarity_results: Dict returned by
                ``EmbeddingEngine.compute_similarity_matrix``.
            invention_description: The user's full invention description text
                for richer Gemini context.
            detail_patents: DataFrame or list of dicts with patent metadata
                (used to look up patent titles).

        Returns:
            List of enriched match dicts, each containing all original
            embedding fields plus Gemini fields, divergence detection, and
            an ``overall_confidence`` rating.
        """
        if not similarity_results or not similarity_results.get("matches"):
            return []

        # Build patent title lookup from detail_patents
        import pandas as pd
        _title_lookup: dict[str, str] = {}
        if detail_patents is not None:
            if isinstance(detail_patents, pd.DataFrame):
                for _, row in detail_patents.iterrows():
                    pn = row.get("publication_number", "")
                    title = row.get("title", "")
                    if pn and title:
                        _title_lookup[pn] = str(title)
            elif isinstance(detail_patents, list):
                for p in detail_patents:
                    if isinstance(p, dict):
                        pn = p.get("publication_number", "")
                        title = p.get("title", "")
                        if pn and title:
                            _title_lookup[pn] = str(title)

        # Filter and sort ------------------------------------------------
        high_mod = [
            m
            for m in similarity_results["matches"]
            if m["similarity_level"] in ("HIGH", "MODERATE")
        ]
        high_mod.sort(key=lambda x: x["similarity_score"], reverse=True)
        pairs_to_analyze = high_mod[:25]  # increased cap to cover all report matches

        if not pairs_to_analyze:
            return []

        enriched: list[dict] = []
        batches = list(chunk_list(pairs_to_analyze, 3))

        for batch_idx, batch in enumerate(batches):
            # -- Build the batch prompt text --------------------------------
            pair_lines: list[str] = []
            for i, pair in enumerate(batch, 1):
                patent_title = _title_lookup.get(pair['patent_number'], 'Unknown')
                pair_lines.append(
                    f"--- PAIR {i} ---\n"
                    f"Feature Label: {pair['feature_label']}\n"
                    f"Feature Description: {pair['feature_description']}\n"
                    f"Patent: {pair['patent_number']} \u2014 {patent_title}\n"
                    f"Claim Element (Claim {pair['claim_number']}): "
                    f"{pair['element_text']}"
                )

            pairs_text = "\n\n".join(pair_lines)
            prompt = CLAIM_ANALYSIS_PROMPT.format(
                n_pairs=len(batch),
                pairs_text=pairs_text,
                invention_description=invention_description or "Not provided",
            )

            # -- Call Gemini with retry logic --------------------------------
            batch_results: list[dict] = []
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    response = self.gemini.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                    )
                    raw_text = response.text or ""
                    batch_results = _parse_gemini_response(raw_text, len(batch))
                    break  # success
                except Exception as exc:
                    logger.error(
                        "Gemini batch %d/%d attempt %d failed: %s",
                        batch_idx + 1, len(batches), attempt + 1, exc,
                    )
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)
                    else:
                        logger.error(
                            "Batch %d failed after %d attempts. Using defaults.",
                            batch_idx + 1, max_retries + 1,
                        )
                        batch_results = [dict(_FIELD_DEFAULTS) for _ in batch]

            # Pad if Gemini returned fewer objects than expected
            while len(batch_results) < len(batch):
                batch_results.append(dict(_FIELD_DEFAULTS))

            # -- Build index map for pair_index-based matching ----------------
            result_map: dict[int, dict] = {}
            for r in batch_results:
                idx = r.get("pair_index", None)
                if idx is not None:
                    try:
                        result_map[int(idx)] = r
                    except (ValueError, TypeError):
                        pass

            # -- Merge & detect divergence ----------------------------------
            for pair_idx, pair in enumerate(batch):
                # Try index-based match first (1-based), fall back to positional
                gemini_result = result_map.get(pair_idx + 1, None)
                if gemini_result is None:
                    gemini_result = batch_results[pair_idx] if pair_idx < len(batch_results) else dict(_FIELD_DEFAULTS)
                gemini_result = _validate_result(gemini_result)

                # -- Coherence check: verify Gemini response matches the pair --
                feature_words = set(pair["feature_label"].lower().split())
                explanation_lower = gemini_result.get("claim_element_explanation", "").lower()
                if not any(w in explanation_lower for w in feature_words if len(w) > 3):
                    logger.warning(
                        "Gemini response incoherent for '%s'; clearing misaligned data.",
                        pair["feature_label"],
                    )
                    gemini_result = _validate_result({})

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
                time.sleep(1.5)

        # Post-processing: check for duplicate distinctions across all matches
        from collections import Counter, defaultdict
        all_distinctions = [str(e.get("key_distinctions", [])) for e in enriched]
        dist_counts = Counter(all_distinctions)
        for text, count in dist_counts.items():
            if count > 2 and text != "[]":
                logger.warning(
                    "Generic distinction detected (%d occurrences). "
                    "Consider re-querying.", count
                )

        # Post-processing: deduplicate key_distinctions within feature groups
        feature_groups = defaultdict(list)
        for e in enriched:
            feature_groups[e.get("feature_label", "")].append(e)

        for fl, entries in feature_groups.items():
            # Check if all key_distinctions are identical within this feature
            dist_texts = [str(e.get("key_distinctions", [])) for e in entries]
            if len(set(dist_texts)) == 1 and len(entries) > 1:
                # All identical — inject claim-specific context
                for e in entries:
                    claim_text = e.get("element_text", "")
                    patent = e.get("patent_number", "")
                    claim_words = [w for w in claim_text.split() if len(w) > 4][:5]
                    claim_phrase = " ".join(claim_words)

                    existing = e.get("key_distinctions", [])
                    if existing and isinstance(existing, list):
                        modified = list(existing)
                        if modified:
                            modified[0] = (
                                f"{modified[0].rstrip('.')} "
                                f"(specifically contrasting with the claim's language: "
                                f"'{claim_phrase}...')."
                            )
                        e["key_distinctions"] = modified

            # Check if all gemini_explanations are identical within this feature
            exp_texts = [e.get("gemini_explanation", "") for e in entries]
            if len(set(exp_texts)) == 1 and len(entries) > 1 and exp_texts[0]:
                for e in entries:
                    claim_text = e.get("element_text", "")
                    original_exp = e.get("gemini_explanation", "")
                    if claim_text:
                        claim_snippet = claim_text[:120].strip()
                        e["gemini_explanation"] = (
                            f"Regarding the specific claim language \"{claim_snippet}\": "
                            f"{original_exp}"
                        )

        return enriched

    @staticmethod
    def localize_snippet(
        claim_text: str,
        feature_label: str,
        feature_description: str,
        max_len: int = 200,
    ) -> str:
        """
        Find the most relevant snippet within *claim_text* that relates
        to the given feature, using keyword overlap scoring.

        Returns a substring of claim_text (up to *max_len* chars) centred
        on the highest-scoring window.
        """
        if not claim_text or not feature_label:
            return claim_text[:max_len] if claim_text else ""

        # Build keyword set from feature label + description
        import re
        _stop = {"a", "an", "the", "of", "for", "to", "in", "and", "or", "is", "with", "by"}
        words = set()
        for text in (feature_label, feature_description):
            if text:
                for w in re.findall(r"[a-z]+", text.lower()):
                    if w not in _stop and len(w) > 2:
                        words.add(w)

        if not words:
            return claim_text[:max_len]

        # Sliding window scoring
        cleaned = " ".join(claim_text.split())
        best_start, best_score = 0, -1
        step = max(1, max_len // 4)
        for start in range(0, max(1, len(cleaned) - max_len + 1), step):
            window = cleaned[start:start + max_len].lower()
            score = sum(1 for w in words if w in window)
            if score > best_score:
                best_score = score
                best_start = start

        snippet = cleaned[best_start:best_start + max_len]
        if best_start > 0:
            snippet = "..." + snippet
        if best_start + max_len < len(cleaned):
            snippet = snippet + "..."
        return snippet
