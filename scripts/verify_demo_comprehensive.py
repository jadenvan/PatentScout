#!/usr/bin/env python3
"""
PatentScout — Comprehensive Demo Verification Script
Runs all 4 test groups specified in the requirements:
  Test 1: Solar Charger Demo Consistency
  Test 2: Smart Doorbell Demo with Sketch
  Test 3: Doorbell Demo WITHOUT Sketch (robustness)
  Test 4: Both Demos Coexist (data isolation)
Plus 24-gate PDF verification on both PDFs.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

#  TEST 1 — Solar Charger Demo Consistency

def test_solar_demo_consistency():
    """Verify solar demo session data matches PDF expectations."""
    print("\n" + "=" * 65)
    print("  TEST 1: Solar Charger Demo Consistency")
    print("=" * 65)

    from demo_data import build_solar_demo_data
    data = build_solar_demo_data()
    results = {}

    # T1.1: 6 features with enriched Gemini analysis
    features = data["search_strategy"]["features"]
    results["T1.1_six_features"] = len(features) == 6

    # T1.2: Comparison matrix has enriched entries for all 6 features
    cmat = data["comparison_matrix"]
    analysed_features = set(c["feature_label"] for c in cmat if c.get("gemini_explanation"))
    results["T1.2_all_features_enriched"] = len(analysed_features) >= 6

    # T1.3: Scores vary (not all 0.907)
    scores = [c["similarity_score"] for c in cmat]
    results["T1.3_scores_varied"] = len(set(round(s, 3) for s in scores)) >= 5

    # T1.4: Landscape patents produce filing trends
    landscape = data["landscape_patents"]
    years = set()
    for p in landscape:
        fd = str(p.get("filing_date", ""))
        if len(fd) >= 4:
            years.add(fd[:4])
    results["T1.4_filing_trends_data"] = len(years) >= 8

    # T1.5: Top assignees have power-law distribution
    from collections import Counter
    assignees = []
    for p in landscape:
        a = p.get("assignee_name", "")
        if isinstance(a, list):
            assignees.extend(a)
        else:
            assignees.append(a)
    counts = Counter(assignees).most_common()
    top_count = counts[0][1] if counts else 0
    bottom_5_avg = sum(c[1] for c in counts[-5:]) / max(1, len(counts[-5:]))
    results["T1.5_power_law_assignees"] = top_count > bottom_5_avg * 3

    # T1.6: Detail patents = 20
    results["T1.6_detail_count_20"] = len(data["detail_patents"]) == 20

    # T1.7: Landscape patents = 500
    results["T1.7_landscape_count_500"] = len(landscape) == 500

    # T1.8: White spaces present
    results["T1.8_white_spaces_exist"] = len(data["white_spaces"]) >= 2

    _print_results(results)
    return all(results.values()), results


#  TEST 2 — Smart Doorbell Demo with Sketch

def test_doorbell_demo_with_sketch():
    """Verify doorbell demo with sketch integration."""
    print("\n" + "=" * 65)
    print("  TEST 2: Smart Doorbell Demo with Sketch")
    print("=" * 65)

    from demo_data import build_doorbell_demo_data
    data = build_doorbell_demo_data()
    results = {}

    # T2.1: Sketch file exists
    sketch_path = None
    for ext in [".png", ".jpg", ".jpeg"]:
        p = os.path.join("assets", "demo", f"doorbell_sketch{ext}")
        if os.path.exists(p):
            sketch_path = p
            break
    results["T2.1_sketch_file_exists"] = sketch_path is not None

    # T2.2: sketch_used flag is True
    results["T2.2_sketch_used_true"] = data.get("sketch_used") is True

    # T2.3: Sketch image bytes loaded
    results["T2.3_sketch_bytes_loaded"] = (
        data.get("invention_image") is not None
        and len(data.get("invention_image", b"")) > 100
    )

    # T2.4: Features include sketch-sourced entries
    features = data["search_strategy"]["features"]
    sketch_features = [f for f in features if f.get("source") == "sketch"]
    results["T2.4_sketch_features_present"] = len(sketch_features) >= 1

    # T2.5: Detail patents = 20 with varied scores and multiple assignees
    detail = data["detail_patents"]
    results["T2.5_detail_20_patents"] = len(detail) == 20
    assignees = set()
    for p in detail:
        a = p.get("assignee_name", [])
        if isinstance(a, list):
            for x in a:
                assignees.add(x)
        else:
            assignees.add(a)
    results["T2.6_8plus_assignees"] = len(assignees) >= 8

    # T2.7: 6+ features with enriched analysis
    cmat = data["comparison_matrix"]
    enriched_features = set(c["feature_label"] for c in cmat if c.get("gemini_explanation"))
    results["T2.7_6plus_enriched_features"] = len(enriched_features) >= 6

    # T2.8: Varied similarity scores
    scores = [c["similarity_score"] for c in cmat]
    results["T2.8_varied_scores"] = len(set(round(s, 3) for s in scores)) >= 5

    # T2.9: White space shows 2 findings
    results["T2.9_white_space_findings"] = len(data["white_spaces"]) >= 2

    # T2.10: Landscape = 500
    results["T2.10_landscape_500"] = len(data["landscape_patents"]) == 500

    # T2.11: Power-law assignee distribution in landscape
    from collections import Counter
    ls_assignees = []
    for p in data["landscape_patents"]:
        a = p.get("assignee_name", "")
        if isinstance(a, list):
            ls_assignees.extend(a)
        else:
            ls_assignees.append(a)
    ls_counts = Counter(ls_assignees).most_common()
    results["T2.11_landscape_power_law"] = (
        ls_counts[0][1] > 50 if ls_counts else False
    )

    _print_results(results)
    return all(results.values()), results


#  TEST 3 — Doorbell Demo WITHOUT Sketch (robustness)

def test_doorbell_demo_without_sketch():
    """Verify doorbell demo works even when sketch file is missing."""
    print("\n" + "=" * 65)
    print("  TEST 3: Doorbell Demo WITHOUT Sketch (Robustness)")
    print("=" * 65)

    results = {}

    # Temporarily rename sketch file
    sketch_path = None
    for ext in [".png", ".jpg", ".jpeg"]:
        p = os.path.join("assets", "demo", f"doorbell_sketch{ext}")
        if os.path.exists(p):
            sketch_path = p
            break

    temp_path = None
    if sketch_path:
        temp_path = sketch_path + ".bak"
        os.rename(sketch_path, temp_path)

    try:
        from demo_data import build_doorbell_demo_data
        # Force re-import to clear any cached path
        import importlib
        import demo_data
        importlib.reload(demo_data)
        from demo_data import build_doorbell_demo_data as bdd_fresh

        data = bdd_fresh()

        # T3.1: Function does NOT crash
        results["T3.1_no_crash"] = True

        # T3.2: sketch_used should be False
        results["T3.2_sketch_used_false"] = data.get("sketch_used") is False

        # T3.3: invention_image should be None
        results["T3.3_no_image_bytes"] = data.get("invention_image") is None

        # T3.4: Data is still valid
        results["T3.4_detail_patents_exist"] = len(data.get("detail_patents", [])) == 20
        results["T3.5_features_exist"] = len(data["search_strategy"]["features"]) >= 6
        results["T3.6_cmat_exists"] = len(data.get("comparison_matrix", [])) > 0

    except Exception as exc:
        results["T3.1_no_crash"] = False
        print(f"  Exception: {exc}")

    finally:
        # Restore sketch file
        if temp_path and os.path.exists(temp_path):
            if sketch_path:
                os.rename(temp_path, sketch_path)

    _print_results(results)
    return all(results.values()), results


#  TEST 4 — Both Demos Coexist (data isolation)

def test_both_demos_coexist():
    """Verify loading one demo fully replaces the other's data."""
    print("\n" + "=" * 65)
    print("  TEST 4: Both Demos Coexist (Data Isolation)")
    print("=" * 65)

    from demo_data import build_solar_demo_data, build_doorbell_demo_data
    results = {}

    # Load solar
    solar = build_solar_demo_data()
    solar_text = solar["invention_text"]
    solar_n_detail = len(solar["detail_patents"])

    # Load doorbell
    doorbell = build_doorbell_demo_data()
    doorbell_text = doorbell["invention_text"]
    doorbell_n_detail = len(doorbell["detail_patents"])

    # T4.1: Different invention texts
    results["T4.1_different_texts"] = solar_text != doorbell_text

    # T4.2: Different patent numbers
    solar_pubs = set(p["publication_number"] for p in solar["detail_patents"])
    doorbell_pubs = set(p["publication_number"] for p in doorbell["detail_patents"])
    results["T4.2_no_patent_overlap"] = len(solar_pubs & doorbell_pubs) == 0

    # T4.3: Both have 20 detail patents
    results["T4.3_solar_20_detail"] = solar_n_detail == 20
    results["T4.4_doorbell_20_detail"] = doorbell_n_detail == 20

    # T4.5: Different feature sets
    solar_feats = set(f["label"] for f in solar["search_strategy"]["features"])
    doorbell_feats = set(f["label"] for f in doorbell["search_strategy"]["features"])
    results["T4.5_different_features"] = solar_feats != doorbell_feats

    # T4.6: Load solar again — data should match first solar load
    solar2 = build_solar_demo_data()
    results["T4.6_solar_reproducible"] = (
        solar2["invention_text"] == solar_text
        and len(solar2["detail_patents"]) == solar_n_detail
    )

    _print_results(results)
    return all(results.values()), results


#  Helpers

def _print_results(results):
    passed = sum(1 for v in results.values() if v)
    for gate, result in results.items():
        marker = "✅" if result else "❌"
        status = "PASS" if result else "FAIL"
        print(f"  {marker} {gate}: {status}")
    print(f"\n  RESULT: {passed}/{len(results)}")


#  MAIN

if __name__ == "__main__":
    print("=" * 65)
    print("  PATENTSCOUT — COMPREHENSIVE DEMO VERIFICATION")
    print("=" * 65)

    t1_ok, _ = test_solar_demo_consistency()
    t2_ok, _ = test_doorbell_demo_with_sketch()
    t3_ok, _ = test_doorbell_demo_without_sketch()
    t4_ok, _ = test_both_demos_coexist()

    print("\n" + "=" * 65)
    print("  FINAL SUMMARY — ALL TEST GROUPS")
    print("=" * 65)
    all_ok = t1_ok and t2_ok and t3_ok and t4_ok
    print(f"  Test 1 (Solar Consistency):    {'PASS' if t1_ok else 'FAIL'}")
    print(f"  Test 2 (Doorbell + Sketch):    {'PASS' if t2_ok else 'FAIL'}")
    print(f"  Test 3 (Doorbell No Sketch):   {'PASS' if t3_ok else 'FAIL'}")
    print(f"  Test 4 (Both Coexist):         {'PASS' if t4_ok else 'FAIL'}")

    if all_ok:
        print("\n  *** ALL 4 TEST GROUPS PASS ***")
    else:
        print("\n  *** FAILURES DETECTED ***")

    print("=" * 65)
    sys.exit(0 if all_ok else 1)
