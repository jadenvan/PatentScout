"""
PatentScout — White Space Finder Module

Synthesises the comparison matrix and landscape data to identify areas of
lower patent density, flagging them as potential IP white-space opportunities.

Three analysis types are run:
1. Feature Gap — features with no moderate/high prior-art match.
2. Classification Gap — predicted CPC subclasses with few retrieved patents.
3. Combination Novelty — Gemini check whether all features together appear
   in any single patent.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import pandas as pd

from modules.confidence_scorer import ConfidenceScorer

logger = logging.getLogger(__name__)

_COMBINATION_PROMPT = """\
You are a rigorous patent analyst. Your task is to assess whether the specific
*combination* of the following invention features appears together in any single
patent from the abstracts provided.

INVENTION FEATURES:
{features_text}

REPRESENTATIVE PATENT ABSTRACTS (up to 10):
{abstracts_text}

INSTRUCTIONS:
1. Before concluding the combination is novel, specifically check if any single
   patent covers at least 3 of these features simultaneously.
2. Do NOT assess individual features — only the combination matters here.
3. Return ONLY a raw JSON object with exactly these keys:

{{
  "combination_appears_in_prior_art": true | false,
  "closest_patent": "<publication_number or empty string>",
  "explanation": "<2-3 sentence assessment>",
  "confidence": "HIGH" | "MODERATE" | "LOW"
}}

No markdown fences. First character must be '{{', last must be '}}'.
"""


class WhiteSpaceFinder:
    """
    Identifies IP white-space opportunities from similarity results,
    search strategy, and retrieved patent data.
    """

    def __init__(self, gemini_client=None) -> None:
        self.gemini = gemini_client
        self.scorer = ConfidenceScorer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def identify_gaps(
        self,
        features: list[dict],
        similarity_results: dict,
        landscape_df_size: int,
        search_strategy: dict,
        detail_df: Optional[pd.DataFrame] = None,
    ) -> list[dict]:
        """
        Run all three gap analyses and return a ranked list of white-space
        findings (capped at 6, HIGH/MODERATE first).

        Parameters
        ----------
        features:
            Feature dicts from ``search_strategy["features"]``.
        similarity_results:
            Output of ``EmbeddingEngine.compute_similarity_matrix()``.
        landscape_df_size:
            Number of patents in the landscape DataFrame.
        search_strategy:
            Full strategy dict including ``cpc_codes``.
        detail_df:
            Optional detail DataFrame used for CPC density counts.

        Returns
        -------
        list[dict]
            Each item: type, title, description, boundary_patents,
            confidence, data_completeness, disclaimer.
        """
        white_spaces: list[dict] = []

        # 1. Feature gap analysis
        white_spaces.extend(
            self._feature_gaps(similarity_results, landscape_df_size)
        )

        # 2. Classification gap analysis
        white_spaces.extend(
            self._classification_gaps(
                search_strategy, landscape_df_size, detail_df
            )
        )

        # 3. Combination novelty (Gemini)
        combo = self._combination_novelty(
            features, similarity_results, landscape_df_size, detail_df
        )
        if combo:
            white_spaces.append(combo)

        # Rank: HIGH → MODERATE → LOW → INSUFFICIENT; cap at 6
        _order = {"HIGH": 0, "MODERATE": 1, "LOW": 2, "INSUFFICIENT": 3}
        white_spaces.sort(
            key=lambda x: _order.get(
                x.get("confidence", {}).get("level", "LOW"), 2
            )
        )
        return white_spaces[:6]

    # ------------------------------------------------------------------
    # Private analysis methods
    # ------------------------------------------------------------------

    def _feature_gaps(
        self, similarity_results: dict, landscape_df_size: int
    ) -> list[dict]:
        gaps: list[dict] = []
        unmatched = similarity_results.get("unmatched_features", [])
        all_matches = similarity_results.get("matches", [])

        # If >80% of features are unmatched, add a broad-terminology warning
        total_features = len(
            set(m["feature_label"] for m in all_matches) | {u["label"] for u in unmatched}
        )
        many_unmatched = (
            total_features > 0
            and len(unmatched) / total_features > 0.8
        )
        broad_note = (
            " Many features appear unmatched, which may indicate very broad "
            "or unconventional terminology rather than true novelty."
            if many_unmatched else ""
        )

        for feature in unmatched:
            nearest = [
                m for m in all_matches
                if m["feature_label"] == feature["label"]
            ][:3]

            max_sim = nearest[0]["similarity_score"] if nearest else 0.0

            confidence = self.scorer.score_finding(
                "feature_gap",
                {
                    "max_similarity": max_sim,
                    "total_patents": landscape_df_size,
                },
            )

            gaps.append(
                {
                    "type": "Feature Gap",
                    "title": f"Low prior art coverage: {feature['label']}",
                    "description": (
                        f"No existing patent claim element showed moderate or "
                        f"high similarity to your feature: "
                        f"'{feature.get('description', feature['label'])}'."
                        + broad_note
                    ),
                    "boundary_patents": [
                        {
                            "patent": m["patent_number"],
                            "score": m["similarity_score"],
                            "element": m["element_text"][:100],
                        }
                        for m in nearest
                    ],
                    "confidence": confidence,
                    "data_completeness": (
                        f"Based on {landscape_df_size} patents retrieved from "
                        f"Google BigQuery Patents Database. This search does "
                        f"not cover: pending unpublished applications, trade "
                        f"secrets, or non-patent literature."
                    ),
                    "disclaimer": (
                        "This is not a patentability opinion. Consult a "
                        "registered patent attorney or agent for professional "
                        "analysis."
                    ),
                }
            )

        return gaps

    def _classification_gaps(
        self,
        search_strategy: dict,
        landscape_df_size: int,
        detail_df: Optional[pd.DataFrame],
    ) -> list[dict]:
        gaps: list[dict] = []

        for cpc_entry in search_strategy.get("cpc_codes", []):
            code: str = cpc_entry.get("code", "")
            if not code:
                continue

            # Count retrieved patents that carry this CPC code
            cpc_count = 0
            if detail_df is not None and "cpc_code" in detail_df.columns:
                prefix = code[:4]  # compare at 4-char group level
                cpc_count = int(
                    detail_df["cpc_code"]
                    .dropna()
                    .apply(
                        lambda lst: (
                            isinstance(lst, list)
                            and any(
                                isinstance(c, str) and c.startswith(prefix)
                                for c in lst
                            )
                        )
                    )
                    .sum()
                )

            # Only flag if coverage is sparse (<= 10 patents)
            if cpc_count > 10:
                continue

            confidence = self.scorer.score_finding(
                "classification_gap",
                {
                    "total_patents": landscape_df_size,
                    "cpc_patent_count": cpc_count,
                },
            )

            rationale_txt = cpc_entry.get("rationale", "")
            gaps.append(
                {
                    "type": "Classification Gap",
                    "title": f"Sparse coverage in CPC subclass {code}",
                    "description": (
                        f"The predicted CPC subclass **{code}** "
                        f"({rationale_txt}) matched only {cpc_count} patent(s) "
                        f"in the retrieved landscape, suggesting low activity "
                        f"in this technology subclass."
                    ),
                    "boundary_patents": [],
                    "confidence": confidence,
                    "data_completeness": (
                        f"CPC density computed across {landscape_df_size} "
                        f"patents from Google BigQuery Patents Database."
                    ),
                    "disclaimer": (
                        "This is not a patentability opinion. Consult a "
                        "registered patent attorney or agent for professional "
                        "analysis."
                    ),
                }
            )

        return gaps

    def _combination_novelty(
        self,
        features: list[dict],
        similarity_results: dict,
        landscape_df_size: int,
        detail_df: Optional[pd.DataFrame],
    ) -> Optional[dict]:
        """Ask Gemini whether all features together appear in any single patent."""
        if not self.gemini:
            return None

        # Build feature text
        features_text = "\n".join(
            f"- {f['label']}: {f.get('description', '')}" for f in features
        )

        # Collect up to 10 representative abstracts from detail_df
        abstracts_text = "(No abstracts available)"
        if detail_df is not None and "abstract" in detail_df.columns:
            sample = detail_df.dropna(subset=["abstract"]).head(10)
            lines = []
            for _, row in sample.iterrows():
                pub = row.get("publication_number", "?")
                ab = str(row["abstract"])[:400]
                lines.append(f"[{pub}] {ab}")
            if lines:
                abstracts_text = "\n\n".join(lines)

        prompt = _COMBINATION_PROMPT.format(
            features_text=features_text,
            abstracts_text=abstracts_text,
        )

        try:
            response = self.gemini.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            raw = (response.text or "").strip()
            # Strip markdown fences if Gemini added them
            if raw.startswith("```"):
                raw = raw.split("```")[-2] if "```" in raw[3:] else raw
                raw = raw.lstrip("`json").lstrip("`").strip()
            parsed = json.loads(raw)
        except Exception as exc:
            logger.warning("Combination novelty Gemini call failed: %s", exc)
            return None

        appears = parsed.get("combination_appears_in_prior_art", True)
        explanation = parsed.get("explanation", "")
        gemini_conf = parsed.get("confidence", "LOW")
        closest = parsed.get("closest_patent", "")

        # Only surface this as a white-space finding if the combination
        # does NOT appear in prior art
        if appears:
            return None

        confidence = self.scorer.score_finding(
            "combination_novelty",
            {"gemini_confidence": gemini_conf, "total_patents": landscape_df_size},
        )

        boundary = (
            [{"patent": closest, "score": None, "element": "Closest prior-art patent"}]
            if closest
            else []
        )

        return {
            "type": "Combination Novelty",
            "title": "Specific feature combination not found in prior art",
            "description": explanation,
            "boundary_patents": boundary,
            "confidence": confidence,
            "data_completeness": (
                f"Assessed against {landscape_df_size} patents from Google "
                f"BigQuery Patents Database using Gemini AI analysis of "
                f"representative abstracts."
            ),
            "disclaimer": (
                "This is not a patentability opinion. Consult a registered "
                "patent attorney or agent for professional analysis."
            ),
        }
