"""
PatentScout — Evaluation Harness

Reusable test framework for comparing retrieval strategies, similarity
quality, and overall pipeline metrics across multiple test cases.

Usage::

    from tests.evaluation_harness import (
        TEST_CASES, evaluate_retrieval_quality,
        evaluate_similarity_quality, print_experiment_comparison,
    )
"""

from __future__ import annotations

import pandas as pd

# TEST CASES

TEST_CASE_1 = {
    "name": "Solar Charger",
    "description": (
        "A portable solar panel that folds into a compact case and charges "
        "mobile phones via USB-C connection. It includes an integrated battery "
        "pack for storing energy when sunlight is not available."
    ),
    "expected_relevant_terms": [
        "solar", "photovoltaic", "solar cell", "solar panel",
        "solar charger", "solar charging", "portable power",
        "foldable", "collapsible", "USB", "battery",
        "energy storage", "power bank", "charging",
    ],
    "expected_cpc_areas": ["H02S", "H02J", "H01L31", "H01M"],
    "known_relevant_patents": [
        "solar charging", "portable photovoltaic",
        "foldable solar", "USB charging solar",
    ],
    "irrelevant_indicators": [
        "nuclear", "wind turbine", "hydroelectric",
        "gas turbine", "combustion engine", "diesel",
        "petroleum", "coal", "natural gas",
    ],
}

TEST_CASE_2 = {
    "name": "Smart Doorbell",
    "description": (
        "A smart doorbell with an integrated camera that uses facial "
        "recognition to identify known visitors and automatically unlocks "
        "the door for pre-approved people. It connects to home WiFi and "
        "sends alerts to the homeowner's phone."
    ),
    "expected_relevant_terms": [
        "doorbell", "door", "camera", "facial recognition",
        "face detection", "visitor", "entry", "lock",
        "unlock", "smart home", "surveillance", "alert",
        "notification", "WiFi", "wireless", "access control",
    ],
    "expected_cpc_areas": ["H04N", "G06V", "E05B", "G08B"],
    "irrelevant_indicators": [
        "automotive", "vehicle", "engine", "aircraft",
        "medical", "pharmaceutical", "chemical compound",
    ],
}

TEST_CASE_3 = {
    "name": "Bike Helmet",
    "description": (
        "A bicycle helmet with built-in bone conduction speakers for "
        "navigation audio, an air quality sensor that monitors pollution "
        "levels during the ride, and LED turn signals activated by hand "
        "gestures detected through an accelerometer in the rider's glove."
    ),
    "expected_relevant_terms": [
        "helmet", "bicycle", "bone conduction", "speaker",
        "audio", "navigation", "air quality", "sensor",
        "pollution", "LED", "turn signal", "gesture",
        "accelerometer", "cycling", "wearable",
    ],
    "expected_cpc_areas": ["A42B", "H04R", "G01N", "B62J"],
    "irrelevant_indicators": [
        "pharmaceutical", "chemical process", "oil drilling",
        "semiconductor fabrication", "gene therapy",
    ],
}

TEST_CASES = [TEST_CASE_1, TEST_CASE_2, TEST_CASE_3]


# EVALUATION METRICS

def evaluate_retrieval_quality(
    retrieved_patents_df: pd.DataFrame,
    test_case: dict,
) -> dict:
    """
    Score the retrieval quality of a set of patents against a test case.

    Returns a dict of metrics including topical_rate, contamination_rate,
    cpc_relevance_rate, top10_precision, and a weighted composite_score.
    """
    results: dict = {}
    total = len(retrieved_patents_df)

    if total == 0:
        return {
            "total_retrieved": 0,
            "topical_rate": 0.0,
            "topical_count": 0,
            "contamination_rate": 0.0,
            "contaminated_count": 0,
            "cpc_relevance_rate": 0.0,
            "top10_precision": 0.0,
            "composite_score": 0.0,
            "error": "no results",
        }

    # METRIC 1: Topical Relevance Rate
    relevant_count = 0
    for _, row in retrieved_patents_df.iterrows():
        text = (
            str(row.get("title", "")) + " " + str(row.get("abstract", ""))
        ).lower()
        has_relevant = any(
            term.lower() in text
            for term in test_case["expected_relevant_terms"]
        )
        relevant_count += int(has_relevant)

    results["topical_rate"] = relevant_count / total
    results["topical_count"] = relevant_count
    results["total_retrieved"] = total

    # METRIC 2: Irrelevant Contamination Rate
    contaminated = 0
    for _, row in retrieved_patents_df.iterrows():
        text = (
            str(row.get("title", "")) + " " + str(row.get("abstract", ""))
        ).lower()
        has_relevant = any(
            term.lower() in text
            for term in test_case["expected_relevant_terms"]
        )
        has_irrelevant = any(
            term.lower() in text
            for term in test_case["irrelevant_indicators"]
        )
        if has_irrelevant and not has_relevant:
            contaminated += 1

    results["contamination_rate"] = contaminated / total
    results["contaminated_count"] = contaminated

    # METRIC 3: CPC Relevance Rate
    cpc_relevant = 0
    for _, row in retrieved_patents_df.iterrows():
        cpc_list = row.get("cpc_code", [])
        if isinstance(cpc_list, list):
            has_expected_cpc = any(
                any(
                    str(code).startswith(prefix)
                    for prefix in test_case["expected_cpc_areas"]
                )
                for code in cpc_list
            )
            cpc_relevant += int(has_expected_cpc)

    results["cpc_relevance_rate"] = cpc_relevant / total

    # METRIC 4: Top-10 Precision
    top10 = retrieved_patents_df.head(10)
    top10_relevant = 0
    for _, row in top10.iterrows():
        text = (
            str(row.get("title", "")) + " " + str(row.get("abstract", ""))
        ).lower()
        has_relevant = any(
            term.lower() in text
            for term in test_case["expected_relevant_terms"]
        )
        top10_relevant += int(has_relevant)

    results["top10_precision"] = top10_relevant / min(10, total)

    # COMPOSITE SCORE (weighted)
    results["composite_score"] = (
        results["topical_rate"] * 0.35
        + results["top10_precision"] * 0.30
        + results["cpc_relevance_rate"] * 0.20
        + (1 - results["contamination_rate"]) * 0.15
    )

    return results


def evaluate_similarity_quality(
    similarity_results: dict,
    comparison_matrix: list,
    test_case: dict,
) -> dict:
    """
    Score the quality of the similarity analysis.
    """
    results: dict = {}

    matches = similarity_results.get("matches", [])

    # METRIC 1: Match volume
    results["total_matches"] = len(matches)
    results["high_matches"] = len(
        [m for m in matches if m.get("similarity_level") == "HIGH"]
    )
    results["moderate_matches"] = len(
        [m for m in matches if m.get("similarity_level") == "MODERATE"]
    )

    # METRIC 2: Feature coverage
    features_with_match: set = set()
    for m in matches:
        if m.get("similarity_level") in ("HIGH", "MODERATE"):
            features_with_match.add(m.get("feature_label", ""))
    total_features = (
        len(set(m.get("feature_label", "") for m in matches)) if matches else 1
    )
    results["feature_coverage"] = len(features_with_match) / total_features

    # METRIC 3: Match topical relevance
    relevant_matches = 0
    for m in matches:
        if m.get("similarity_level") in ("HIGH", "MODERATE"):
            element_lower = m.get("element_text", "").lower()
            if any(
                term.lower() in element_lower
                for term in test_case["expected_relevant_terms"]
            ):
                relevant_matches += 1

    high_mod_total = results["high_matches"] + results["moderate_matches"]
    results["match_topical_rate"] = (
        relevant_matches / high_mod_total if high_mod_total > 0 else 0
    )

    return results


def evaluate_whitespace_quality(
    white_spaces: list[dict],
    test_case: dict,
    retrieved_df: pd.DataFrame | None = None,
) -> dict:
    """
    Evaluate white space findings for false positives and meaningfulness.
    """
    results: dict = {
        "total_findings": len(white_spaces),
        "feature_gaps": 0,
        "classification_gaps": 0,
        "combination_novelty": 0,
        "false_positives": 0,
        "meaningful": 0,
    }

    for ws in white_spaces:
        ws_type = ws.get("type", "")
        if ws_type == "Feature Gap":
            results["feature_gaps"] += 1
        elif ws_type == "Classification Gap":
            results["classification_gaps"] += 1
            # Check for false positive: CPC gap when we actually have
            # many relevant patents
            if retrieved_df is not None and len(retrieved_df) > 50:
                # If we have 50+ patents, a CPC gap may be a format mismatch
                title_text = ws.get("title", "").lower()
                if "sparse" in title_text or "0 patent" in ws.get("description", "").lower():
                    results["false_positives"] += 1
                else:
                    results["meaningful"] += 1
            else:
                results["meaningful"] += 1
        elif ws_type == "Combination Novelty":
            results["combination_novelty"] += 1
            results["meaningful"] += 1

    # Feature gaps are generally meaningful
    results["meaningful"] += results["feature_gaps"]

    fp_rate = (
        results["false_positives"] / results["total_findings"]
        if results["total_findings"] > 0
        else 0.0
    )
    results["false_positive_rate"] = fp_rate

    return results


# COMPARISON DISPLAY

def print_experiment_comparison(experiments: list[dict]) -> None:
    """
    Print a formatted comparison table of all experiments for a given
    test case.

    Each experiment dict should have:
    - name: str
    - retrieval_metrics: dict (from evaluate_retrieval_quality)
    - similarity_metrics: dict (if applicable)
    - bytes_scanned: float (GB)
    - runtime_seconds: float
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPARISON")
    print("=" * 80)

    print(
        f"{'Experiment':<30} {'Topical%':>8} {'Top10%':>7} "
        f"{'CPC%':>5} {'Contam%':>8} {'Composite':>9} "
        f"{'GB':>6} {'Time':>6}"
    )
    print("-" * 80)

    for exp in sorted(
        experiments,
        key=lambda x: x.get("retrieval_metrics", {}).get("composite_score", 0),
        reverse=True,
    ):
        rm = exp.get("retrieval_metrics", {})
        print(
            f"{exp['name']:<30} "
            f"{rm.get('topical_rate', 0) * 100:>7.1f}% "
            f"{rm.get('top10_precision', 0) * 100:>6.1f}% "
            f"{rm.get('cpc_relevance_rate', 0) * 100:>4.1f}% "
            f"{rm.get('contamination_rate', 0) * 100:>7.1f}% "
            f"{rm.get('composite_score', 0):>9.3f} "
            f"{exp.get('bytes_scanned', 0):>5.1f} "
            f"{exp.get('runtime_seconds', 0):>5.0f}"
        )

    # Identify winner
    if experiments:
        winner = max(
            experiments,
            key=lambda x: x.get("retrieval_metrics", {}).get(
                "composite_score", 0
            ),
        )
        print(
            f"\n🏆 WINNER: {winner['name']} "
            f"(composite: {winner['retrieval_metrics']['composite_score']:.3f})"
        )


def print_full_summary(
    before_metrics: dict,
    after_metrics: dict,
    tests_passing: str = "22/22",
) -> None:
    """Print the final before/after summary."""
    print("\n" + "=" * 50)
    print("PATENTSCOUT POST-FIX SUMMARY")
    print("=" * 50)

    def _pct(v: float) -> str:
        return f"{v * 100:.1f}%"

    b = before_metrics
    a = after_metrics

    print(f"Retrieval Precision: {_pct(b.get('topical_rate', 0))} → {_pct(a.get('topical_rate', 0))} (topical rate)")
    print(f"Top-10 Precision:    {_pct(b.get('top10_precision', 0))} → {_pct(a.get('top10_precision', 0))}")
    print(f"Contamination Rate:  {_pct(b.get('contamination_rate', 0))} → {_pct(a.get('contamination_rate', 0))}")
    print(f"BigQuery Cost:       {b.get('gb_per_analysis', 0):.1f} GB → {a.get('gb_per_analysis', 0):.1f} GB per analysis")
    print(f"White Space FP Rate: {_pct(b.get('ws_fp_rate', 0))} → {_pct(a.get('ws_fp_rate', 0))}")
    print(f"Charts in PDF:       {b.get('charts_in_pdf', '0/3')} → {a.get('charts_in_pdf', '3/3')}")
    print(f"Runtime:             {b.get('runtime_s', 0):.0f}s → {a.get('runtime_s', 0):.0f}s")
    print(f"Tests Passing:       22/22 → {tests_passing}")
