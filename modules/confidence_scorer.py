"""
PatentScout — Confidence Scorer Module

Scores individual white-space finding types (feature gaps, classification
gaps, combination novelty) and returns a structured confidence dict with
level and human-readable rationale.
"""

from __future__ import annotations


class ConfidenceScorer:
    """
    Applies rule-based thresholds to score white-space findings.

    Each ``score_finding`` call returns::

        {"level": "HIGH" | "MODERATE" | "LOW" | "INSUFFICIENT",
         "rationale": "<explanation>"}

    Rules
    -----
    **feature_gap**
        =========== =========================================
        HIGH         ≥100 patents searched, max similarity < 0.30
        MODERATE     50–99 patents searched, max similarity < 0.40
        LOW          <50 patents OR max similarity 0.40–0.50
        INSUFFICIENT <10 patents searched
        =========== =========================================

    **classification_gap**
        =========== ==============================================
        HIGH         predicted CPC returned 0 patents AND ≥200 total
        MODERATE     predicted CPC returned 1–3 patents
        LOW          predicted CPC returned 4–10 patents
        INSUFFICIENT total search returned <20 patents
        =========== ==============================================

    **combination_novelty**
        Derived from Gemini's own confidence level + data volume.
    """

    def score_finding(
        self, finding_type: str, supporting_data: dict
    ) -> dict:
        """
        Score a single white-space finding.

        Parameters
        ----------
        finding_type:
            One of ``"feature_gap"``, ``"classification_gap"``,
            ``"combination_novelty"``.
        supporting_data:
            Dict of keys appropriate to the finding type — see class
            docstring for required keys per type.

        Returns
        -------
        dict
            ``{"level": str, "rationale": str}``
        """
        handler = {
            "feature_gap": self._feature_gap,
            "classification_gap": self._classification_gap,
            "combination_novelty": self._combination_novelty,
        }.get(finding_type)

        if handler is None:
            return {
                "level": "INSUFFICIENT",
                "rationale": f"Unknown finding type: {finding_type}",
            }
        return handler(supporting_data)

    # ------------------------------------------------------------------
    # Private per-type scorers
    # ------------------------------------------------------------------

    def _feature_gap(self, data: dict) -> dict:
        total: int = int(data.get("total_patents", 0))
        max_sim: float = float(data.get("max_similarity", 0.0))

        if total < 10:
            return {
                "level": "INSUFFICIENT",
                "rationale": (
                    f"Only {total} patents were retrieved — too few to draw "
                    "reliable conclusions about prior-art coverage."
                ),
            }
        if total >= 100 and max_sim < 0.30:
            return {
                "level": "HIGH",
                "rationale": (
                    f"Searched {total} patents; nearest embedding match scored "
                    f"{max_sim:.3f} (< 0.30 threshold). Strong evidence of low "
                    "prior-art coverage for this feature."
                ),
            }
        if total >= 50 and max_sim < 0.40:
            return {
                "level": "MODERATE",
                "rationale": (
                    f"Searched {total} patents; nearest embedding match scored "
                    f"{max_sim:.3f} (< 0.40 threshold). Moderate evidence of a "
                    "coverage gap, though the corpus is of limited size."
                ),
            }
        return {
            "level": "LOW",
            "rationale": (
                f"Searched {total} patents; nearest match scored {max_sim:.3f}. "
                "Insufficient evidence of a genuine gap — consider broadening "
                "the search or reviewing results manually."
            ),
        }

    def _classification_gap(self, data: dict) -> dict:
        total: int = int(data.get("total_patents", 0))
        cpc_count: int = int(data.get("cpc_patent_count", 0))

        if total < 20:
            return {
                "level": "INSUFFICIENT",
                "rationale": (
                    f"Total search returned only {total} patents — too few to "
                    "assess CPC classification density."
                ),
            }
        if cpc_count == 0 and total >= 200:
            return {
                "level": "HIGH",
                "rationale": (
                    f"The predicted CPC code returned 0 patents from a corpus "
                    f"of {total}. Strong signal of a classification-level gap."
                ),
            }
        if 1 <= cpc_count <= 3:
            return {
                "level": "MODERATE",
                "rationale": (
                    f"Only {cpc_count} retrieved patent(s) carry this CPC code. "
                    "Relatively sparse coverage in this technology subclass."
                ),
            }
        if 4 <= cpc_count <= 10:
            return {
                "level": "LOW",
                "rationale": (
                    f"{cpc_count} patents found with this CPC code. Some prior "
                    "art exists; gap may be narrower than initially assessed."
                ),
            }
        return {
            "level": "LOW",
            "rationale": (
                f"{cpc_count} patents carry this CPC code — coverage appears "
                "dense in this classification area."
            ),
        }

    def _combination_novelty(self, data: dict) -> dict:
        gemini_conf: str = str(data.get("gemini_confidence", "LOW")).upper()
        total: int = int(data.get("total_patents", 0))

        level_map = {"HIGH": "HIGH", "MODERATE": "MODERATE", "LOW": "LOW"}
        level = level_map.get(gemini_conf, "LOW")

        if total < 20:
            level = "INSUFFICIENT"
            rationale = (
                f"Corpus too small ({total} patents) for reliable combination "
                "novelty assessment."
            )
        else:
            rationale = (
                f"Gemini assessed combination novelty as {gemini_conf} based "
                f"on {total} retrieved patents and their abstracts."
            )

        return {"level": level, "rationale": rationale}

        raise NotImplementedError
