"""
PatentScout — Embedding Engine Module

Loads the sentence-transformers model (cached via st.cache_resource so it is
only downloaded / initialised once per Streamlit process lifetime) and
computes a full cosine-similarity matrix between user invention features and
parsed patent claim elements.

Improvements (optimize/bigquery-embed):
- Accepts feature["patent_language"] for reformulated patent-style text.
- Computes similarity as elementwise max(original_sim, reformulated_sim)
  so the best encoding wins.
- Falls back to a smaller model if the primary model fails to load.
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
    """
    Wraps a sentence-transformers model to encode text into dense embedding
    vectors and compute cosine similarity between invention features and
    patent claim elements.

    The underlying model is loaded once per Streamlit server process via
    :func:`_load_model` which is decorated with ``@st.cache_resource``.
    """

    def __init__(self) -> None:
        self.model, self.model_name = self._load_model()

    @staticmethod
    @st.cache_resource
    def _load_model() -> tuple[SentenceTransformer, str]:
        """Load (or retrieve from cache) the sentence-transformers model.

        Falls back to a smaller model if the primary fails.

        Returns:
            ``(model, model_name_used)``
        """
        primary = settings.EMBEDDING_MODEL_NAME
        fallback = settings.EMBEDDING_MODEL_FALLBACK_NAME

        try:
            m = SentenceTransformer(primary)
            logger.info("Embedding model loaded: %s", primary)
            return m, primary
        except Exception as exc:
            logger.warning(
                "Primary embedding model '%s' failed (%s) — trying fallback '%s'",
                primary, exc, fallback,
            )
            print(
                f"[EmbeddingEngine] Primary model '{primary}' failed: {exc}\n"
                f"  Trying fallback: '{fallback}'"
            )

        try:
            m = SentenceTransformer(fallback)
            logger.info("Fallback embedding model loaded: %s", fallback)
            print(f"[EmbeddingEngine] Fallback model '{fallback}' loaded successfully.")
            return m, fallback
        except Exception as exc2:
            raise RuntimeError(
                f"Both embedding models failed to load.\n"
                f"  Primary:  {primary}\n"
                f"  Fallback: {fallback}\n"
                f"  Last error: {exc2}"
            ) from exc2

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_similarity_matrix(
        self,
        features: list[dict],
        parsed_claims: list[dict],
    ) -> dict:
        """
        Compute cosine similarity between every user feature and every parsed
        claim element.

        If a feature contains a ``"patent_language"`` key (added by
        :meth:`QueryBuilder.reformulate_features_for_patent_language`), both
        the original description *and* the patent-language reformulation are
        encoded.  The similarity score used is the elementwise maximum so the
        better-matching encoding always wins.

        Args:
            features:
                List of feature dicts, each with at least a ``"label"`` and
                ``"description"`` key (as produced by QueryBuilder).
            parsed_claims:
                List of parsed-claim dicts as produced by ClaimParser
                (each must contain ``"patent_number"`` and
                ``"independent_claims"``).

        Returns:
            A dict with keys:
                ``matrix``             – raw numpy similarity matrix (F × E)
                ``feature_labels``     – list of feature label strings
                ``element_refs``       – list of element reference dicts
                ``matches``            – flat list of match dicts (top-10 per feature)
                ``unmatched_features`` – features with no HIGH/MODERATE match
                ``stats``              – summary counts
            On error:
                ``matrix`` is ``None``, ``error`` contains a message.
        """
        # Step 1 — Feature texts (original + optional patent-language)
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

        # Step 2 — Collect all claim elements with source tracking
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

        # Edge-case: no elements could be parsed
        if not element_texts:
            return {
                "matrix": None,
                "matches": [],
                "unmatched_features": features,
                "error": "No claim elements were successfully parsed",
            }

        # Step 3 — Encode
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
                "Similarity matrix: used elementwise max(original, patent_language). "
                "Improvement vs original: %.4f avg",
                float((sim_matrix - sim_orig).mean()),
            )
            print(
                f"[EmbeddingEngine] Used max(original, patent_language) similarity. "
                f"Avg improvement: {(sim_matrix - sim_orig).mean():.4f}"
            )
        else:
            sim_matrix = sim_orig
            logger.info("Similarity matrix: patent_language not available — using original only.")

        # Step 4 — Build structured results
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
