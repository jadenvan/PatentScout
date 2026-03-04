"""
PatentScout — Embedding Engine Module

Computes cosine-similarity matrix between invention features and parsed patent
claim elements using sentence-transformers embeddings. Supports optional
patent-language reformulation — takes elementwise max of both similarity scores.
"""

from __future__ import annotations

import logging

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import settings

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Sentence-transformer wrapper for feature-vs-claim cosine similarity."""

    def __init__(self) -> None:
        self.model, self.model_name = self._load_model()

    @staticmethod
    @st.cache_resource
    # load model once per process; fall back to smaller model if needed
    def _load_model() -> tuple[SentenceTransformer, str]:
        primary = settings.EMBEDDING_MODEL_NAME
        fallback = settings.EMBEDDING_MODEL_FALLBACK_NAME

        try:
            m = SentenceTransformer(primary)
            logger.info("embedding model loaded: %s", primary)
            return m, primary
        except Exception as exc:
            logger.warning(
                "primary embedding model '%s' failed (%s) — trying fallback '%s'",
                primary, exc, fallback,
            )

        try:
            m = SentenceTransformer(fallback)
            logger.info("fallback embedding model loaded: %s", fallback)
            return m, fallback
        except Exception as exc2:
            raise RuntimeError(
                f"Both embedding models failed to load.\n"
                f"  Primary:  {primary}\n"
                f"  Fallback: {fallback}\n"
                f"  Last error: {exc2}"
            ) from exc2

    # Public API

    def compute_similarity_matrix(
        self,
        features: list[dict],
        parsed_claims: list[dict],
    ) -> dict:
        """compute feature-vs-claim cosine similarity matrix."""
        # step 1: feature texts (original + optional patent-language)
        feature_texts_orig   = [f.get("description", "") for f in features]
        feature_texts_patent = [
            f.get("patent_language") or f.get("description", "")
            for f in features
        ]
        feature_labels = [f["label"] for f in features]

        uses_reformulation = any(
            f.get("patent_language") and f["patent_language"] != f.get("description", "")
            for f in features
        )

        # step 2: collect claim elements with source tracking
        element_texts: list[str] = []
        element_refs: list[dict] = []

        for patent in parsed_claims:
            for claim in patent.get("independent_claims", []):
                for element in claim.get("elements", []):
                    element_texts.append(element["text"])
                    element_refs.append(
                        {
                            "patent_number": patent["patent_number"],
                            "claim_number": claim["claim_number"],
                            "element_id": element["id"],
                            "full_claim_text": claim.get("full_text", ""),
                        }
                    )

        # edge case: no elements parsed
        if not element_texts:
            return {
                "matrix": None,
                "matches": [],
                "unmatched_features": features,
                "error": "No claim elements were successfully parsed",
            }

        # step 3: encode
        element_embeddings = self.model.encode(
            element_texts, show_progress_bar=False
        )
        orig_embeddings = self.model.encode(
            feature_texts_orig, show_progress_bar=False
        )

        # Cosine similarity using original descriptions (F × E)
        sim_orig = cosine_similarity(orig_embeddings, element_embeddings)

        if uses_reformulation:
            patent_embeddings = self.model.encode(
                feature_texts_patent, show_progress_bar=False
            )
            sim_patent = cosine_similarity(patent_embeddings, element_embeddings)
            # Elementwise maximum — best encoding wins
            sim_matrix = np.maximum(sim_orig, sim_patent)
            logger.info(
                "similarity matrix: max(orig, patent_language), avg delta %.4f",
                float((sim_matrix - sim_orig).mean()),
            )
        else:
            sim_matrix = sim_orig
            logger.info("Similarity matrix: patent_language not available — using original only.")

        # step 4: build structured results
        matches: list[dict] = []
        unmatched_features: list[dict] = []

        for i, feature in enumerate(features):
            feature_matches: list[dict] = []

            for j, ref in enumerate(element_refs):
                score = float(sim_matrix[i][j])
                if score >= settings.SIMILARITY_THRESHOLD_LOW:
                    if score >= settings.SIMILARITY_THRESHOLD_HIGH:
                        level = "HIGH"
                    elif score >= settings.SIMILARITY_THRESHOLD_MODERATE:
                        level = "MODERATE"
                    else:
                        level = "LOW"

                    feature_matches.append(
                        {
                            "feature_label": feature["label"],
                            "feature_description": feature.get("description", ""),
                            "feature_patent_language": feature.get("patent_language", ""),
                            "element_text": element_texts[j],
                            "patent_number": ref["patent_number"],
                            "claim_number": ref["claim_number"],
                            "element_id": ref["element_id"],
                            "similarity_score": round(score, 3),
                            "similarity_level": level,
                        }
                    )

            # Sort matches for this feature by score descending
            feature_matches.sort(
                key=lambda x: x["similarity_score"], reverse=True
            )

            # Flag features with no HIGH or MODERATE match anywhere
            has_strong_match = any(
                m["similarity_level"] in ("HIGH", "MODERATE")
                for m in feature_matches
            )
            if not has_strong_match:
                unmatched_features.append(feature)

            # Keep top-10 matches per feature
            matches.extend(feature_matches[:10])

        # Deduplicate matches: one entry per (feature, patent, element)
        _dedup_seen: set[tuple] = set()
        _dedup_matches: list[dict] = []
        for m in matches:
            key = (
                m["feature_label"],
                m["patent_number"],
                m.get("element_id", ""),
            )
            if key not in _dedup_seen:
                _dedup_seen.add(key)
                _dedup_matches.append(m)
        matches = _dedup_matches

        return {
            "matrix": sim_matrix,
            "feature_labels": feature_labels,
            "element_refs": element_refs,
            "matches": matches,
            "unmatched_features": unmatched_features,
            "embedding_model": self.model_name,
            "uses_reformulation": uses_reformulation,
            "stats": {
                "total_comparisons": len(features) * len(element_refs),
                "high_matches": sum(
                    1 for m in matches if m["similarity_level"] == "HIGH"
                ),
                "moderate_matches": sum(
                    1 for m in matches if m["similarity_level"] == "MODERATE"
                ),
                "low_matches": sum(
                    1 for m in matches if m["similarity_level"] == "LOW"
                ),
            },
        }

    # end of class EmbeddingEngine
