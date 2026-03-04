"""
PatentScout — White Space Finder Module

Synthesises the comparison matrix and landscape data to identify areas of
lower patent density, flagging them as potential IP white-space opportunities.

Three analysis types are run:
1. Feature Gap — features with no moderate/high prior-art match.
2. Classification Density — predicted CPC subclasses with few retrieved patents.
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

    # CPC areas known to have thousands of active patents — if our search
    # finds 0 in any of these the problem is likely our retrieval, not the
    # patent landscape.
    KNOWN_DENSE_CPC_AREAS: dict[str, str] = {
        "H02S": "Solar cells/panels (extremely dense)",
        "H02J": "Circuit arrangements for power supply",
        "H01L": "Semiconductor devices",
        "H04N": "Image communication",
        "G06V": "Image recognition",
        "H04W": "Wireless communication",
        "A61K": "Pharmaceutical preparations",
        "G06F": "Electric digital data processing",
    }

    def __init__(self, gemini_client=None) -> None:
        self.gemini = gemini_client
        self.scorer = ConfidenceScorer()

    # Public API

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

        # 2. Classification gap analysis (capped at 2 to avoid overwhelming report)
        cpc_gaps = self._classification_gaps(
                search_strategy, landscape_df_size, detail_df
            )
        white_spaces.extend(cpc_gaps[:2])

        # 3. Combination novelty (Gemini)
        combo = self._combination_novelty(
            features, similarity_results, landscape_df_size, detail_df
        )
        if combo:
            white_spaces.append(combo)

        # Rank: HIGH → MODERATE/MEDIUM → LOW → INSUFFICIENT; cap at 6
        _order = {"HIGH": 0, "MEDIUM": 0, "MODERATE": 1, "LOW": 2, "INSUFFICIENT": 3}
        white_spaces.sort(
            key=lambda x: _order.get(
                x.get("confidence", {}).get("level", "LOW"), 2
            )
        )

        # Filter out LOW confidence findings if we have MEDIUM+ findings
        has_medium_or_high = any(
            ws.get("confidence", {}).get("level", "LOW") in ("MEDIUM", "HIGH", "MODERATE")
            for ws in white_spaces
        )
        if has_medium_or_high:
            white_spaces = [
                ws for ws in white_spaces
                if ws.get("confidence", {}).get("level", "LOW") != "LOW"
            ]

        return white_spaces[:6]

    # Private analysis methods

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

        # NEW: low-coverage features — features matched by few patents
        # Use detail patent count as denominator, NOT landscape_df_size
        features_coverage_high: dict[str, set] = {}   # feature → patents with HIGH/MODERATE match
        features_coverage_any: dict[str, set] = {}    # feature → patents with any match
        all_detail_patents: set = set()

        for m in all_matches:
            fl = m.get("feature_label", "")
            pn = m.get("patent_number", "")
            score = m.get("similarity_score", 0)
            level = m.get("similarity_level", "LOW")
            all_detail_patents.add(pn)
            features_coverage_any.setdefault(fl, set()).add(pn)
            if level in ("HIGH", "MODERATE") and score >= 0.50:
                features_coverage_high.setdefault(fl, set()).add(pn)

        total_detail = max(len(all_detail_patents), 1)

        for fl, patent_set in features_coverage_any.items():
            high_set = features_coverage_high.get(fl, set())
            high_pct = (len(high_set) / total_detail) * 100
            any_pct = (len(patent_set) / total_detail) * 100

            if high_pct < 30:  # Less than 30% of detail patents have HIGH/MODERATE match
                if high_pct == 0:
                    conf_level = "HIGH"
                    desc = (
                        f"Feature '{fl}' has no high-confidence matches among the "
                        f"{total_detail} patents analysed in detail. While {len(patent_set)} "
                        f"patents show low-level textual similarity, none demonstrate "
                        f"substantive overlap with this specific feature."
                    )
                elif high_pct < 15:
                    conf_level = "MEDIUM"
                    desc = (
                        f"Feature '{fl}' shows strong overlap with only {len(high_set)} of "
                        f"{total_detail} detail-analysed patents ({high_pct:.0f}% coverage). "
                        f"This relatively sparse coverage suggests the invention's "
                        f"specific approach to this feature is less common in the prior art."
                    )
                else:
                    conf_level = "MEDIUM"
                    desc = (
                        f"Feature '{fl}' has moderate prior art coverage, with {len(high_set)} "
                        f"of {total_detail} patents ({high_pct:.0f}% coverage) showing meaningful overlap. "
                        f"The remaining patents address related technology but differ in implementation."
                    )

                gaps.append(
                    {
                        "type": "Low Coverage",
                        "title": f"Limited prior art coverage for: {fl}",
                        "description": desc,
                        "confidence": {
                            "level": conf_level,
                            "rationale": f"{high_pct:.0f}% high-confidence coverage among detail patents",
                        },
                        "data_completeness": (
                            f"Analysed {total_detail} detail patents; "
                            f"{len(high_set)} with HIGH/MODERATE match"
                        ),
                        "disclaimer": (
                            f"Coverage computed against {total_detail} detail-analysed "
                            f"patents, not the full {landscape_df_size} landscape."
                        ),
                        "boundary_patents": list(high_set)[:3],
                    }
                )

        return gaps

    @staticmethod
    def _normalise_cpc(code: str) -> str:
        """Normalise a CPC code for comparison: upper, no spaces."""
        return code.upper().replace(" ", "").strip()

    @staticmethod
    def _describe_cpc_density(count: int, total_searched: int) -> str | None:
        """Generate appropriate language for CPC density findings."""
        if count == 0:
            return (
                "No patents with this classification were found in our search "
                "results. This may indicate a gap in patent coverage or may "
                "reflect limitations of our keyword-based search methodology."
            )
        elif count <= 5:
            return (
                f"Only {count} patent(s) with this classification appeared in "
                f"the {total_searched} patents retrieved. This suggests "
                f"relatively limited patent activity in this specific subclass, "
                f"though a professional search may reveal additional patents "
                f"not captured by our keyword-based approach."
            )
        elif count <= 20:
            return (
                f"{count} patents with this classification were found among "
                f"{total_searched} retrieved patents. This represents moderate "
                f"activity. Further investigation recommended to determine if "
                f"specific feature combinations within this area remain uncovered."
            )
        else:
            # 20+ patents — this is NOT white space
            return None

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

            norm_code = self._normalise_cpc(code)

            # Count retrieved patents that carry this CPC code using
            # flexible multi-level prefix matching:
            #   1. Try full normalised code match
            #   2. Try subgroup prefix (up to '/'), e.g. "H02S40"
            #   3. Try subclass (4-char), e.g. "H02S"
            #   4. Try class (3-char), e.g. "H02"
            cpc_count = 0
            if detail_df is not None and "cpc_code" in detail_df.columns:
                # Build list of prefixes to try, longest first
                prefixes = [norm_code]
                slash_idx = norm_code.find("/")
                if slash_idx > 0:
                    prefixes.append(norm_code[:slash_idx])   # subgroup
                if len(norm_code) >= 4:
                    prefixes.append(norm_code[:4])           # subclass
                if len(norm_code) >= 3:
                    prefixes.append(norm_code[:3])           # class
                # De-dup while preserving order
                seen = set()
                unique_prefixes = []
                for p in prefixes:
                    if p not in seen:
                        seen.add(p)
                        unique_prefixes.append(p)

                def _matches_any_prefix(cpc_list):
                    if not isinstance(cpc_list, list):
                        return False
                    for c in cpc_list:
                        if not isinstance(c, str):
                            continue
                        nc = c.upper().replace(" ", "").strip()
                        for prefix in unique_prefixes:
                            if nc.startswith(prefix):
                                return True
                    return False

                cpc_count = int(
                    detail_df["cpc_code"]
                    .dropna()
                    .apply(_matches_any_prefix)
                    .sum()
                )

                logger.debug(
                    "CPC %s (norm: %s) — matched %d patents with prefixes %s",
                    code, norm_code, cpc_count, unique_prefixes,
                )

            # Generate density description — skip if not a gap
            density_desc = self._describe_cpc_density(cpc_count, landscape_df_size)
            if density_desc is None:
                continue

            rationale_txt = cpc_entry.get("rationale", "")

            # Set confidence based on count — all CPC density findings are LOW
            if cpc_count == 0:
                confidence = {
                    "level": "LOW",
                    "rationale": (
                        "Zero results may reflect search limitations rather "
                        "than a true gap in this CPC area."
                    ),
                }
            elif cpc_count <= 5:
                confidence = {
                    "level": "LOW",
                    "rationale": (
                        f"Low match count ({cpc_count}) with corpus size "
                        f"({landscape_df_size})."
                    ),
                }
            else:
                confidence = {
                    "level": "LOW",
                    "rationale": (
                        f"Moderate activity detected. Finding reflects "
                        f"relative density, not absence."
                    ),
                }

            # Sanity check: known dense CPC areas
            cpc_prefix_4 = norm_code[:4] if len(norm_code) >= 4 else norm_code
            known_desc = self.KNOWN_DENSE_CPC_AREAS.get(cpc_prefix_4)
            if known_desc and cpc_count == 0:
                confidence = {
                    "level": "LOW",
                    "rationale": (
                        f"CPC area {cpc_prefix_4} ({known_desc}) is known to "
                        f"be highly active. The absence of matches likely "
                        f"reflects search limitations rather than a true gap."
                    ),
                }

            # Quality gate: only report gaps if retrieval was topically relevant
            if detail_df is not None and "relevance_score" in detail_df.columns:
                top10_avg = detail_df.head(10)["relevance_score"].mean()
                if top10_avg < 0.3:
                    confidence = {
                        "level": "LOW",
                        "rationale": (
                            "Retrieval quality too low (avg relevance "
                            f"{top10_avg:.2f}) to make reliable CPC density claims."
                        ),
                    }

            gaps.append(
                {
                    "type": "Classification Density",
                    "title": f"Lower density in CPC {code}",
                    "description": density_desc,
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
