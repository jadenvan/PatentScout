"""
PatentScout — Embedding Engine Module

Loads the sentence-transformers model (cached via st.cache_resource so it is
only downloaded / initialised once per Streamlit process lifetime) and
computes a full cosine-similarity matrix between user invention features and
parsed patent claim elements.
"""

from __future__ import annotations

import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import settings


class EmbeddingEngine:
    """
    Wraps a sentence-transformers model to encode text into dense embedding
    vectors and compute cosine similarity between invention features and
    patent claim elements.

    The underlying model is loaded once per Streamlit server process via
    :func:`_load_model` which is decorated with ``@st.cache_resource``.
    """

    def __init__(self) -> None:
        self.model = self._load_model()

    @staticmethod
    @st.cache_resource
    def _load_model() -> SentenceTransformer:
        """Load (or retrieve from cache) the sentence-transformers model."""
        return SentenceTransformer(settings.EMBEDDING_MODEL_NAME)

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
        # Step 1 — Feature texts
        feature_texts = [f["description"] for f in features]
        feature_labels = [f["label"] for f in features]

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
        feature_embeddings = self.model.encode(
            feature_texts, show_progress_bar=False
        )
        element_embeddings = self.model.encode(
            element_texts, show_progress_bar=False
        )

        # Step 4 — Cosine similarity matrix  (F × E)
        sim_matrix = cosine_similarity(feature_embeddings, element_embeddings)

        # Step 5 — Build structured results
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
                            "feature_description": feature["description"],
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
