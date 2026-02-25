"""
PatentScout — Query Builder Module

Handles Gemini-powered feature extraction from invention descriptions and
optional sketch images.  Produces a structured search strategy dict containing
technical features, predicted CPC codes, and BigQuery-ready regex patterns.
Also builds BigQuery WHERE clause fragments from the extracted strategy.
"""

from __future__ import annotations

import json
import re
import time
from typing import Optional

from google import genai
from google.genai import types

from config.prompts import FEATURE_EXTRACTION_PROMPT, REFORMULATION_PROMPT
from config import settings

# ---------------------------------------------------------------------------
# Models to try in order (first available / non-rate-limited wins).
# gemini-1.5-* are excluded — deprecated and return 404 on v1beta.
# ---------------------------------------------------------------------------
_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

# Required top-level keys in the feature-extraction response
_REQUIRED_KEYS = {"features", "cpc_codes", "search_terms"}

# Required keys per item in each list
_FEATURE_KEYS = {"label", "description", "keywords"}
_CPC_KEYS = {"code", "rationale"}
_TERM_KEYS = {"primary", "synonyms", "bigquery_regex"}


def _detect_mime(image_bytes: bytes) -> str:
    """Return the MIME type for raw image bytes (JPEG or PNG fallback)."""
    if image_bytes[:2] == b"\xff\xd8":
        return "image/jpeg"
    if image_bytes[:4] == b"\x89PNG":
        return "image/png"
    if image_bytes[:4] in (b"GIF8", b"GIF9"):
        return "image/gif"
    return "image/jpeg"  # safe default


def _strip_markdown(text: str) -> str:
    """Remove markdown code fences that Gemini sometimes wraps JSON in."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json(text: str) -> str:
    """
    Extract the first JSON object found in *text*, handling cases where Gemini
    includes surrounding prose.
    """
    cleaned = _strip_markdown(text)
    if cleaned.startswith("{"):
        return cleaned

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)

    raise ValueError(f"No JSON object found in Gemini response:\n{text[:500]}")


def _validate_regex(pattern: str) -> str:
    """
    Test that *pattern* compiles as a valid Python regex.
    Returns the pattern unchanged if valid, or a simple (?i)keyword fallback.
    """
    try:
        re.compile(pattern)
        return pattern
    except re.error:
        words = re.findall(r"[a-zA-Z]{3,}", pattern)
        if words:
            return "(?i)(" + "|".join(words[:5]) + ")"
        return "(?i)invention"


def _sanitize_regex(pattern: str) -> str:
    """
    Simplify a potentially over-complex regex pattern for BigQuery use.

    Strategy:
    1. Validate the regex compiles (fix if not).
    2. Count alternation branches inside the outermost group.  If there are
       more than 7, keep only the first 7 shortest non-empty branches so the
       pattern stays selective without being excessively long.
    3. Return a clean ``(?i)(tok1|tok2|...)`` pattern.

    This prevents Gemini from generating patterns like
    ``(?i)(solar|photovoltaic|pv|...)`` with 20+ branches that cause regex
    engine timeouts or BigQuery slot exhaustion.
    """
    pattern = _validate_regex(pattern)

    # Extract all 4+ letter words from the pattern as the token pool
    words = list(dict.fromkeys(
        w.lower() for w in re.findall(r"[a-zA-Z]{4,}", pattern)
    ))
    if not words:
        return pattern

    # Cap at 7 shortest tokens to keep the alternation lean
    words = sorted(words, key=len)[:7]
    return "(?i)(" + "|".join(words) + ")"


def _validate_structure(data: dict) -> None:
    """Raise ValueError if *data* is missing required keys or sub-keys."""
    missing_top = _REQUIRED_KEYS - set(data.keys())
    if missing_top:
        raise ValueError(f"Gemini response missing required keys: {missing_top}")

    for i, feat in enumerate(data.get("features", [])):
        missing = _FEATURE_KEYS - set(feat.keys())
        if missing:
            raise ValueError(f"Feature[{i}] missing keys: {missing}")

    for i, cpc in enumerate(data.get("cpc_codes", [])):
        missing = _CPC_KEYS - set(cpc.keys())
        if missing:
            raise ValueError(f"cpc_codes[{i}] missing keys: {missing}")

    for i, term in enumerate(data.get("search_terms", [])):
        missing = _TERM_KEYS - set(term.keys())
        if missing:
            raise ValueError(f"search_terms[{i}] missing keys: {missing}")


class QueryBuilder:
    """
    Gemini-powered feature extractor and BigQuery query constructor.

    Usage::

        qb = QueryBuilder(api_key="AIza...")
        strategy = qb.extract_features("A solar-powered wallet charger...")
        where = qb.build_bigquery_where_clause(strategy)
    """

    def __init__(self, api_key: str) -> None:
        """
        Args:
            api_key: Gemini API key (from GEMINI_API_KEY env var).
        """
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file and restart the app."
            )
        self._client = genai.Client(api_key=api_key)
        self._model: Optional[str] = None  # resolved on first call

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_gemini_ordered(self, contents: list, models: "list[str]") -> str:
        """
        Core implementation: try each model in *models* order, retrying on 429.
        Returns the response text of the first successful call.
        """
        last_exc: Exception = RuntimeError("No models tried")
        for model in models:
            for attempt in range(3):  # up to 3 rate-limit retries per model
                try:
                    response = self._client.models.generate_content(
                        model=model,
                        contents=contents,
                    )
                    self._model = model
                    time.sleep(1)  # respect free-tier rate limits
                    return response.text
                except Exception as exc:
                    err_str = str(exc)
                    last_exc = exc
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        wait = 5 * (3 ** attempt)  # 5s, 15s, 45s
                        print(
                            f"[QueryBuilder] {model} rate-limited (429), "
                            f"waiting {wait}s (attempt {attempt+1}/3)..."
                        )
                        time.sleep(wait)
                        continue  # retry same model
                    # Non-429 error (404, 500, …) — skip to next model
                    break
        raise RuntimeError(
            f"All Gemini models failed. Last error: {last_exc}"
        ) from last_exc

    def _call_gemini(self, contents: list) -> str:
        """
        Try each model in *_MODELS* until one responds successfully.
        Returns the text of the first successful response.
        """
        return self._call_gemini_ordered(contents, _MODELS)

    def _parse_response(self, raw: str) -> dict:
        """
        Parse and validate a Gemini response string into a strategy dict.
        Raises ValueError with the raw response attached if parsing fails.
        """
        json_str = _extract_json(raw)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Gemini returned invalid JSON: {exc}\n\nRaw response:\n{raw[:800]}"
            ) from exc

        # Sanitise regex patterns before validation
        for term in data.get("search_terms", []):
            if "bigquery_regex" in term:
                term["bigquery_regex"] = _validate_regex(term["bigquery_regex"])

        _validate_structure(data)
        return data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_features(
        self,
        text: str,
        image_bytes: Optional[bytes] = None,
    ) -> dict:
        """
        Call Gemini with the invention description (and optional sketch image)
        to extract features, CPC codes, and search terms.

        Args:
            text: Free-text invention description from the user.
            image_bytes: Raw bytes of an uploaded sketch, or ``None``.

        Returns:
            Validated strategy dict with keys ``features``, ``cpc_codes``,
            ``search_terms``.

        Raises:
            RuntimeError: If all Gemini models fail.
            ValueError: If the response cannot be parsed after retries.
        """
        prompt_text = FEATURE_EXTRACTION_PROMPT + f"\n\nINVENTION DESCRIPTION:\n{text}"

        if image_bytes:
            mime = _detect_mime(image_bytes)
            image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime)
            contents: list = [image_part, prompt_text]
        else:
            contents = [prompt_text]

        # First attempt
        raw = self._call_gemini(contents)
        try:
            return self._parse_response(raw)
        except ValueError:
            pass  # fall through to retry

        # Single retry with explicit JSON-only instruction.
        # Wait briefly to avoid back-to-back rate-limit hits, then prefer
        # the model that just succeeded (stored in self._model).
        retry_contents = contents[:]
        retry_contents[-1] = prompt_text + "\n\nReturn ONLY valid JSON, no other text."
        print("[QueryBuilder] First parse failed, waiting 20s before JSON-only retry...")
        time.sleep(20)
        # Surface any previously-working model to the front of the list
        preferred = self._model
        if preferred and preferred in _MODELS:
            ordered = [preferred] + [m for m in _MODELS if m != preferred]
        else:
            ordered = _MODELS
        # Temporarily override _MODELS ordering for this call
        raw2 = self._call_gemini_ordered(retry_contents, ordered)
        return self._parse_response(raw2)

    def reformulate_features_for_patent_language(
        self,
        features: list[dict],
    ) -> list[dict]:
        """
        Rewrite each feature description into patent-claim-style language
        using Gemini.

        Args:
            features: List of feature dicts with at least a ``"description"``
                      key (as returned by :meth:`extract_features`).

        Returns:
            The same list with a ``"patent_language"`` key added to each
            feature.  Falls back to the original description if Gemini fails.

        Each feature dict is returned with an added key::

            {"label": "...", "description": "...", ...,
             "patent_language": "comprising a foldable solar panel ..."}
        """
        if not features:
            return features

        # Build the prompt input as a JSON array of descriptions
        descriptions = [
            {"original": f.get("description", f.get("label", ""))}
            for f in features
        ]
        prompt = REFORMULATION_PROMPT + f"\n\nInput: {json.dumps(descriptions)}"

        raw: str = ""
        for attempt in range(settings.REFORMULATION_MAX_RETRIES + 1):
            try:
                raw = self._call_gemini([prompt])
                break
            except Exception as exc:
                wait = 2 ** attempt
                print(
                    f"[QueryBuilder] reformulation Gemini call failed "
                    f"(attempt {attempt + 1}): {exc}. Retrying in {wait}s..."
                )
                time.sleep(wait)
        else:
            # All attempts failed — use originals
            print("[QueryBuilder] Reformulation failed after all retries — using original text")
            for f in features:
                if "patent_language" not in f:
                    f["patent_language"] = f.get("description", "")
            return features

        # Parse Gemini response — multiple fallback extraction strategies
        parsed: list | None = None
        for _try in range(2):
            try:
                cleaned = _strip_markdown(raw)
                # Try direct parse first
                if cleaned.startswith("["):
                    parsed = json.loads(cleaned)
                    break
                # Try to extract JSON array with regex
                match = re.search(r"(\[.*\])", raw, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(1))
                    break
            except (json.JSONDecodeError, ValueError):
                pass

            if _try == 0:
                # Retry Gemini with strict instruction
                strict_prompt = prompt + "\n\nReturn ONLY valid JSON, no other text."
                try:
                    raw = self._call_gemini([strict_prompt])
                except Exception:
                    break

        if not parsed or not isinstance(parsed, list):
            print(
                "[QueryBuilder] Reformulation: could not parse Gemini response — "
                "falling back to original descriptions"
            )
            for f in features:
                if "patent_language" not in f:
                    f["patent_language"] = f.get("description", "")
            return features

        # Apply reformulations
        # Build a lookup: original description → patent_language
        remap: dict[str, str] = {}
        for item in parsed:
            if isinstance(item, dict) and "patent_language" in item:
                orig = item.get("original", "")
                remap[orig] = item["patent_language"]

        for f in features:
            desc = f.get("description", "")
            f["patent_language"] = remap.get(desc, desc)

        print(
            f"[QueryBuilder] Reformulated {len([f for f in features if f.get('patent_language')])} "
            f"features into patent language."
        )
        return features

    def build_bigquery_where_clause(self, search_strategy: dict) -> dict:
        """
        Build BigQuery WHERE clause fragments from a feature-extraction strategy.

        Combines all ``bigquery_regex`` patterns with OR into a text filter,
        and builds a CPC LIKE clause from the predicted ``cpc_codes``.

        Args:
            search_strategy: Dict returned by :meth:`extract_features`.

        Returns:
            Dict with keys ``text_filter``, ``cpc_filter``, and ``combined``.
        """
        search_terms = search_strategy.get("search_terms", [])
        cpc_codes = search_strategy.get("cpc_codes", [])

        # --- text filter ---------------------------------------------------
        regex_clauses: list[str] = []
        for term in search_terms:
            pattern = term.get("bigquery_regex", "")
            if pattern:
                # Sanitize: validate + cap alternations to 7 core tokens
                pattern = _sanitize_regex(pattern)
                safe_pattern = pattern.replace("'", "\\'")
                # Use abstract.text (the scalar field from the UNNEST alias)
                regex_clauses.append(
                    f"REGEXP_CONTAINS(abstract.text, r'{safe_pattern}')"
                )

        if regex_clauses:
            text_filter = "(\n  " + "\n  OR ".join(regex_clauses) + "\n)"
        else:
            text_filter = "TRUE  -- no text patterns generated"

        # --- CPC filter ----------------------------------------------------
        cpc_clauses: list[str] = []
        for cpc in cpc_codes:
            code = cpc.get("code", "").strip()
            if code:
                prefix = re.split(r"[/.]", code)[0]
                cpc_clauses.append(f"cpc_item.code LIKE '{prefix}%'")

        if cpc_clauses:
            cpc_filter = "(\n  " + "\n  OR ".join(cpc_clauses) + "\n)"
        else:
            cpc_filter = "TRUE  -- no CPC codes predicted"

        combined = f"{text_filter}\nAND {cpc_filter}"

        return {
            "text_filter": text_filter,
            "cpc_filter": cpc_filter,
            "combined": combined,
        }
