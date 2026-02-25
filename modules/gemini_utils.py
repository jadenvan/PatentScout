"""
PatentScout — Gemini Utilities

Provides cached Gemini reformulation for features, usable both by the main
pipeline and the experiment runner.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_CACHE_DIR = os.path.join(".cache", "reformulations")


def _cache_path(raw_features: list[dict]) -> str:
    """Deterministic cache path based on SHA-256 of feature descriptions."""
    key = json.dumps(
        [f.get("description", f.get("label", "")) for f in raw_features],
        sort_keys=True,
    )
    sha = hashlib.sha256(key.encode()).hexdigest()
    return os.path.join(_CACHE_DIR, f"{sha}.json")


def generate_reformulations(
    features: list[dict],
    api_key: Optional[str] = None,
    force: bool = False,
) -> list[dict]:
    """
    Reformulate feature descriptions into patent-claim-style language.

    Caches results to ``.cache/reformulations/<sha256>.json``.
    If cached, returns immediately without calling Gemini.

    Returns:
        The feature list with ``patent_language`` keys filled in.
    """
    if not features:
        return features

    cache_file = _cache_path(features)

    # Check cache
    if not force and os.path.exists(cache_file):
        try:
            with open(cache_file) as fh:
                cached = json.load(fh)
            if len(cached) == len(features):
                logger.info("Reformulations loaded from cache: %s", cache_file)
                for f, c in zip(features, cached):
                    f["patent_language"] = c.get("patent_language", f.get("description", ""))
                return features
        except Exception:
            pass

    # Check if patent_language is already populated (from session cache)
    if all(f.get("patent_language") for f in features):
        _save_cache(features, cache_file)
        return features

    # Call Gemini
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        logger.warning("No GEMINI_API_KEY — using original descriptions")
        for f in features:
            f.setdefault("patent_language", f.get("description", ""))
        return features

    try:
        from modules.query_builder import QueryBuilder
        qb = QueryBuilder(api_key=api_key)
        features = qb.reformulate_features_for_patent_language(features)
    except Exception as exc:
        logger.warning("Reformulation failed (%s) — using originals", exc)
        for f in features:
            f.setdefault("patent_language", f.get("description", ""))

    _save_cache(features, cache_file)
    return features


def _save_cache(features: list[dict], cache_file: str) -> None:
    """Persist reformulation results to disk."""
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        data = [
            {
                "label": f.get("label", ""),
                "description": f.get("description", ""),
                "patent_language": f.get("patent_language", ""),
            }
            for f in features
        ]
        with open(cache_file, "w") as fh:
            json.dump(data, fh, indent=2)
    except Exception as exc:
        logger.warning("Could not save reformulation cache: %s", exc)
