#!/usr/bin/env python3
"""
PatentScout — 22-Gate Comprehensive Verification
Runs all 15 original gates + 7 new gates on both PDFs.
"""

import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import pdfplumber


def extract_pdf_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def verify_pdf_comprehensive(pdf_path, test_case_name):
    full_text = extract_pdf_text(pdf_path)
    results = {}

    # ORIGINAL 15 GATES (must still pass)

    # GATE 1: No wrong claim-feature matches
    g1_pass = True
    match_blocks = re.split(r'(?=Match \d+:)', full_text)
    for block in match_blocks:
        if not block.strip():
            continue
        header_area = block[:300].lower()
        has_enclosure = "enclosure" in header_area.split("patent")[0] if "patent" in header_area else "enclosure" in header_area[:150]
        has_storage = any(kw in header_area.split("patent")[0] if "patent" in header_area else kw in header_area[:150] for kw in ["storage", "battery"])
        if has_enclosure or has_storage:
            claim_area = ""
            for line in block.split("\n"):
                if "claim element" in line.lower():
                    claim_area = line.lower()
                    break
            if "folding mechanism" in claim_area:
                g1_pass = False
    results["G01_no_wrong_claim_matches"] = g1_pass

    # GATE 2: Gemini text coherence
    g2_pass = True
    for block in match_blocks:
        header = block[:200]
        is_folding = "Folding" in header.split(":")[0] if ":" in header else False
        if is_folding and "What This Claim Requires" in block:
            gemini_part = block.split("What This Claim Requires")[-1][:500].lower()
            sentences = gemini_part.split(".")[:2]
            first_two = ". ".join(sentences)
            if ("energy storage" in first_two or "rechargeable" in first_two) and "fold" not in first_two:
                g2_pass = False
            if ("usb-c" in first_two or "usb type" in first_two) and "fold" not in first_two:
                g2_pass = False
    results["G02_gemini_text_coherent"] = g2_pass

    # GATE 3: Key distinctions uniqueness (cross-feature)
    distinctions = re.findall(
        r'Key Technical Distinctions:?\s*\n(.*?)(?=\nLimitations|\n\*Rec|\nMatch \d|\n---|\n###|$)',
        full_text, re.DOTALL
    )
    unique_d = set(d.strip()[:200] for d in distinctions if d.strip())
    results["G03_unique_distinctions_cross_feature"] = len(unique_d) >= min(3, len(distinctions))

    # GATE 4: Gemini coverage
    gemini_count = full_text.count("What This Claim Requires")
    total_matches = len(re.findall(r'Match \d+:', full_text))
    results["G04_gemini_full_coverage"] = gemini_count >= total_matches * 0.90 if total_matches > 0 else True

    # GATE 5: Comparison Analysis count
    comparison_count = full_text.count("Comparison Analysis")
    results["G05_comparison_count"] = abs(comparison_count - gemini_count) <= 2

    # GATE 6: Assignee chart present
    results["G06_assignee_present"] = "Top Patent Holders" in full_text

    # GATE 7: Section 3 scores
    section3_area = ""
    if "Prior Art Summary" in full_text:
        parts = full_text.split("Prior Art Summary")
        s3 = parts[-1]
        if "Claim Element Comparison" in s3:
            section3_area = s3.split("Claim Element Comparison")[0]
    dash_count = section3_area.count("\u2014") + section3_area.count("—") + section3_area.count("--")
    total_rows_s3 = len(re.findall(r'US-\d+', section3_area))
    results["G07_section3_scores"] = dash_count < total_rows_s3 * 0.3 if total_rows_s3 > 0 else True

    # GATE 8: Section 3 snippets
    results["G08_section3_snippets"] = total_rows_s3 > 0 and len(section3_area) > total_rows_s3 * 50

    # GATE 9: Section 4 feature diversity
    section4_area = ""
    if "Claim Element Comparison" in full_text:
        parts = full_text.split("Claim Element Comparison")
        s4 = parts[-1]
        if "Match Details" in s4:
            section4_area = s4.split("Match Details")[0]
    feature_words = re.findall(
        r'(?:Panel|Enclosure|Storage|Battery|Controller|USB|Interface|Camera|Recognition|Lock|WiFi|Notification|Motion|Sensor|Actuator|Transceiver)',
        section4_area, re.IGNORECASE
    )
    unique_features_s4 = set(f.lower() for f in feature_words)
    results["G09_section4_diversity"] = len(unique_features_s4) >= 3

    # GATE 10: White space quality
    ws_area = ""
    if "White Space Analysis" in full_text:
        ws_split = full_text.split("White Space Analysis")
        ws_raw = ws_split[-1]
        if "Recommended Next Steps" in ws_raw:
            ws_area = ws_raw.split("Recommended Next Steps")[0]
        else:
            ws_area = ws_raw[:4000]
    has_feature_finding = any(kw in ws_area.lower() for kw in ["coverage", "feature", "differentiation", "overlap"])
    has_medium_high = "MEDIUM" in ws_area or "HIGH" in ws_area
    results["G10_whitespace_quality"] = has_feature_finding or has_medium_high

    # GATE 11: Patent count consistency
    count_mentions = re.findall(r'(\d+)\s*patents?\s*(?:retrieved|analysed|analyzed|reviewed|in detail)', full_text.lower())
    if len(count_mentions) >= 2:
        has_diff = "detail" in full_text.lower() or "landscape" in full_text.lower() or "broader" in full_text.lower()
        all_same = len(set(count_mentions)) == 1
        results["G11_count_consistency"] = all_same or has_diff
    else:
        results["G11_count_consistency"] = True

    # GATE 12: CPC no question marks
    results["G12_cpc_no_question_marks"] = "? H0" not in full_text and "? B6" not in full_text and "? E0" not in full_text and "? G0" not in full_text

    # GATE 13: Three charts present
    results["G13_three_charts"] = all(kw in full_text for kw in ["Filing Trends", "Top Patent Holders", "Classification Distribution"])

    # GATE 14: PDF in reports/ folder
    results["G14_correct_location"] = os.path.exists(pdf_path) and "reports" in pdf_path.replace("\\", "/")

    # GATE 15: No stray PDFs
    stray = []
    for root, dirs, files in os.walk(str(PROJECT_ROOT)):
        if "reports" in root or "venv" in root or "/." in root:
            continue
        for f in files:
            if f.endswith(".pdf") and "patentscout" in f.lower():
                stray.append(os.path.join(root, f))
    results["G15_no_stray_pdfs"] = len(stray) == 0

    # NEW GATES 16-22 (for this fix pass)

    # GATE 16: No red block — page 1 should have meaningful text
    page1_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        if pdf.pages:
            page1_text = pdf.pages[0].extract_text() or ""
    results["G16_no_red_block"] = "PATENTSCOUT" in page1_text and len(page1_text) > 100

    # GATE 17: Assignee chart — diverse companies
    if "Top Patent Holders" in full_text:
        holder_area = full_text.split("Top Patent Holders")[1][:2000]
        companies = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:LLC|Inc|Ltd|Corp)\.?)?', holder_area)
        unique_companies = set(c.strip() for c in companies if len(c) > 3)
        results["G17_assignee_diversity"] = len(unique_companies) >= 5
    else:
        results["G17_assignee_diversity"] = False

    # GATE 18: White space — differentiated percentages
    ws_percentages = re.findall(r'(\d+)%', ws_area)
    unique_pcts = set(ws_percentages)
    results["G18_whitespace_varied_pcts"] = len(unique_pcts) >= 2

    # GATE 19: White space — not all "4%"
    results["G19_whitespace_not_all_4pct"] = not all(p == "4" for p in ws_percentages) if ws_percentages else True

    # GATE 20: No `?` adjacent to patent numbers in references
    ref_area = ""
    if "References" in full_text:
        ref_area = full_text.split("References")[-1]
    results["G20_refs_no_question_marks"] = " ? " not in ref_area[:3000] if ref_area else True

    # GATE 21: Key distinctions vary WITHIN feature groups
    g21_pass = True
    feature_distinctions = defaultdict(list)
    for block in match_blocks:
        if not block.strip():
            continue
        header_match = re.match(r'Match \d+:\s*(.+?)(?:\n|$)', block)
        if header_match:
            feature_name = header_match.group(1).strip()
            dist_match = re.search(
                r'Key Technical Distinctions:?\s*\n(.*?)(?=\nLimitations|\n\*Rec|\nMatch|\n---|$)',
                block, re.DOTALL
            )
            if dist_match:
                dist_text = dist_match.group(1).strip()[:300]
                feature_distinctions[feature_name].append(dist_text)

    for feat, dists in feature_distinctions.items():
        if len(dists) > 1 and len(set(dists)) == 1:
            g21_pass = False
            break
    results["G21_distinctions_vary_within_feature"] = g21_pass

    # GATE 22: "What This Claim Requires" varies within feature groups
    g22_pass = True
    feature_explanations = defaultdict(list)
    for block in match_blocks:
        if not block.strip():
            continue
        header_match = re.match(r'Match \d+:\s*(.+?)(?:\n|$)', block)
        if header_match:
            feature_name = header_match.group(1).strip()
            if "What This Claim Requires" in block:
                exp_text = block.split("What This Claim Requires")[1][:400]
                exp_para = exp_text.split("\n\n")[0].strip() if "\n\n" in exp_text else exp_text.split("\n")[1].strip() if "\n" in exp_text else exp_text.strip()
                feature_explanations[feature_name].append(exp_para[:200])

    for feat, exps in feature_explanations.items():
        if len(exps) > 1 and len(set(exps)) == 1:
            g22_pass = False
            break
    results["G22_explanations_vary_within_feature"] = g22_pass

    # NEW GATES 23-24 (multi-modal / sketch)

    # GATE 23: Multi-modal input noted in PDF (doorbell only — solar should skip)
    multimodal_phrases = ["text + sketch", "visual input", "design sketch", "multi-modal", "multimodal"]
    has_multimodal = any(phrase in full_text.lower() for phrase in multimodal_phrases)
    # Only require for doorbell; solar passes automatically
    if "doorbell" in test_case_name.lower() or "smart" in test_case_name.lower():
        results["G23_multimodal_noted"] = has_multimodal
    else:
        results["G23_multimodal_noted"] = True  # N/A for solar

    # GATE 24: At least one sketch-sourced feature mentioned (doorbell only)
    sketch_feature_phrases = ["identified from sketch", "identified from visual", "visual input", "sketch input"]
    has_sketch_feature = any(phrase in full_text.lower() for phrase in sketch_feature_phrases)
    if "doorbell" in test_case_name.lower() or "smart" in test_case_name.lower():
        results["G24_sketch_feature_noted"] = has_sketch_feature
    else:
        results["G24_sketch_feature_noted"] = True  # N/A for solar

    # RESULTS OUTPUT
    print(f"\n{'='*65}")
    print(f"  VERIFICATION: {test_case_name}")
    print(f"{'='*65}")
    passed = 0
    for gate, result in results.items():
        status = "PASS" if result else "FAIL"
        marker = "✅" if result else "❌"
        print(f"  {marker} {gate}: {status}")
        if result:
            passed += 1
    print(f"\n  TOTAL: {passed}/{len(results)}")
    print(f"{'='*65}")

    # Print diagnostics for key sections
    print(f"\n{'~'*50}")
    print("DIAGNOSTIC: White Space Percentages Found")
    print(f"{'~'*50}")
    print(f"  Percentages: {ws_percentages}")
    print(f"  Unique: {unique_pcts}")

    print(f"\n{'~'*50}")
    print("DIAGNOSTIC: Feature Distinction Uniqueness")
    print(f"{'~'*50}")
    for feat, dists in feature_distinctions.items():
        unique_count = len(set(d[:100] for d in dists))
        print(f"  {feat}: {len(dists)} matches, {unique_count} unique distinctions")

    print(f"\n{'~'*50}")
    print("DIAGNOSTIC: Feature Explanation Uniqueness")
    print(f"{'~'*50}")
    for feat, exps in feature_explanations.items():
        unique_count = len(set(exps))
        print(f"  {feat}: {len(exps)} matches, {unique_count} unique explanations")

    # Print sample matches for manual review
    for match_num in [1, 4, 7, 13]:
        pattern = rf'Match {match_num}:.*?(?=Match {match_num+1}:|---\s*\n###|$)'
        m = re.search(pattern, full_text, re.DOTALL)
        if m:
            print(f"\n{'~'*50}")
            print(f"SAMPLE -- MATCH {match_num} (first 800 chars):")
            print(f"{'~'*50}")
            print(m.group()[:800])

    return passed == len(results), results


# RUN ON BOTH TEST CASES

solar_ok, solar_r = verify_pdf_comprehensive("reports/patentscout_solar_charger.pdf", "SOLAR CHARGER")
doorbell_ok, doorbell_r = verify_pdf_comprehensive("reports/patentscout_smart_doorbell.pdf", "SMART DOORBELL")

print(f"\n{'='*65}")
print("FINAL SUMMARY")
print(f"{'='*65}")
print(f"  Solar Charger:  {'ALL 24 PASS' if solar_ok else 'FAILURES DETECTED'}")
print(f"  Smart Doorbell: {'ALL 24 PASS' if doorbell_ok else 'FAILURES DETECTED'}")

if not solar_ok:
    failed = [g for g, r in solar_r.items() if not r]
    print(f"  Solar failures: {failed}")
if not doorbell_ok:
    failed = [g for g, r in doorbell_r.items() if not r]
    print(f"  Doorbell failures: {failed}")

if solar_ok and doorbell_ok:
    print("\n  *** PATENTSCOUT IS DEMO-READY ***")
