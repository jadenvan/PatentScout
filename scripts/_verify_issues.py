#!/usr/bin/env python3
"""Verify all 5 issues are fixed."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from modules.report_helpers import format_patent_date, format_patent_year

cache_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "experiments", "cache", "solar_similarity.json")
with open(cache_path, "r") as f:
    similarity = json.load(f)

print("=" * 60)
print("ISSUE 3 — Enriched Match Quality Check")
print("=" * 60)
if similarity and "matches" in similarity:
    matches = similarity["matches"]
    mod = [m for m in matches if m.get("confidence") in ("HIGH", "MODERATE")][:3]
    for m in mod:
        print(f"\nPatent: {m.get('publication_number', 'N/A')}")
        print(f"  Confidence: {m.get('confidence', 'N/A')}")
        ge = m.get("gemini_explanation", "")
        ga = m.get("gemini_assessment", "")
        kd = m.get("key_distinctions", "")
        rec = m.get("recommendation", "")
        print(f"  gemini_explanation length: {len(ge)}")
        print(f"  gemini_explanation snippet: {ge[:250]}...")
        print(f"  gemini_assessment snippet: {ga[:250]}...")
        print(f"  key_distinctions snippet: {kd[:250]}...")
        print(f"  recommendation snippet: {rec[:250]}...")
        # Check for generic text
        generic_phrases = [
            "conduct a thorough review",
            "consult with a patent attorney",
            "further investigation",
        ]
        all_text = (ge + ga + kd + rec).lower()
        is_generic = any(p in all_text for p in generic_phrases)
        has_patent_ref = m.get("publication_number", "") in (ge + ga + kd)
        print(f"  Contains generic boilerplate: {is_generic}")
        print(f"  References specific patent: {has_patent_ref}")

print("\n" + "=" * 60)
print("ISSUE 4 — Date Formatting Check")
print("=" * 60)
test_dates = [20180315, "20200701", 20210101, None, "", 0]
for d in test_dates:
    print(f"  format_patent_date({d!r}) -> {format_patent_date(d)!r}")
    print(f"  format_patent_year({d!r}) -> {format_patent_year(d)!r}")

# Check dates in matches
if similarity and "matches" in similarity:
    for m in similarity["matches"][:3]:
        pd = m.get("publication_date", "N/A")
        print(f"\n  Raw date: {pd} -> formatted: {format_patent_date(pd)} / year: {format_patent_year(pd)}")
