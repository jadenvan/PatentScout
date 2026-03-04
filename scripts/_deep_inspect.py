#!/usr/bin/env python3
"""Deep inspect of session data issues."""
import json

with open("examples/solar_charger_session.json") as f:
    s = json.load(f)

# Issue 2: Check for duplicates in comparison_matrix
cm = s.get("comparison_matrix", [])
print(f"Comparison matrix length: {len(cm)}")
from collections import Counter
keys = []
for m in cm:
    key = (m.get("feature_label", ""), m.get("patent_number", ""), m.get("element_id", ""))
    keys.append(key)
counter = Counter(keys)
dupes = {k: v for k, v in counter.items() if v > 1}
print(f"Exact duplicate entries: {len(dupes)}")
for key, count in dupes.items():
    print(f"  {count}x: {key}")

# Issue 4: Check Gemini fields in CM
print("\n--- CM field analysis ---")
if cm:
    for i, m in enumerate(cm[:3]):
        print(f"\nMatch {i+1}:")
        print(f"  Keys: {list(m.keys())}")
        print(f"  gemini_explanation: '{str(m.get('gemini_explanation', 'MISSING'))[:80]}'")
        print(f"  gemini_assessment: '{str(m.get('gemini_assessment', 'MISSING'))[:80]}'")
        print(f"  key_distinctions: {m.get('key_distinctions', 'MISSING')}")
        print(f"  gemini_distinction: '{str(m.get('gemini_distinction', 'MISSING'))[:80]}'")
        print(f"  cannot_determine: '{str(m.get('cannot_determine', 'MISSING'))[:80]}'")
        print(f"  overall_confidence: {m.get('overall_confidence', 'MISSING')}")

# Issue 4: Check similarity_results matches for Gemini fields
sr = s.get("similarity_results", {})
matches = sr.get("matches", [])
print(f"\n--- similarity_results matches ({len(matches)}) ---")
if matches:
    m = matches[0]
    print(f"First match keys: {list(m.keys())}")
    high_mod = [m for m in matches if m.get("similarity_level") in ("HIGH", "MODERATE")]
    print(f"HIGH+MODERATE matches: {len(high_mod)}")

# Issue 6: Check white space descriptions for corpus refs
ws = s.get("white_spaces", [])
print(f"\n--- White spaces ({len(ws)}) ---")
for w in ws:
    desc = w.get("description", "")
    data_comp = w.get("data_completeness", "")
    print(f"  Type: {w.get('type', '')}")
    print(f"  Title: {w.get('title', '')}")
    print(f"  Description: {desc[:150]}")
    print(f"  Data completeness: {data_comp[:150]}")
    conf = w.get("confidence", {})
    print(f"  Confidence: {conf}")
