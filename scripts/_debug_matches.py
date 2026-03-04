#!/usr/bin/env python3
"""Debug duplicate matches — inspect what's in each match."""
import json
import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def extract_pdf_text(pdf_path):
    import fitz
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# First, check the matches array from normalise_matches
import pandas as pd
from modules.report_generator import _normalise_matches

with open("examples/solar_charger_session.json") as f:
    session_data = json.load(f)
if isinstance(session_data.get("detail_patents"), list):
    session_data["detail_patents"] = pd.DataFrame(session_data["detail_patents"])

matches = _normalise_matches(session_data)
print(f"Total normalised matches: {len(matches)}")

# Check for true duplicates (same feature + same patent)
from collections import Counter
keys1 = [(m.get("feature_label"), m.get("patent_number")) for m in matches]
c1 = Counter(keys1)
dupes1 = {k: v for k, v in c1.items() if v > 1}
print(f"\nDuplicates by (feature, patent): {len(dupes1)}")
for k, v in list(dupes1.items())[:5]:
    print(f"  {v}x: {k}")

# Check with element_id too
keys2 = [(m.get("feature_label"), m.get("patent_number"), m.get("element_id")) for m in matches]
c2 = Counter(keys2)
dupes2 = {k: v for k, v in c2.items() if v > 1}
print(f"\nDuplicates by (feature, patent, element_id): {len(dupes2)}")
for k, v in list(dupes2.items())[:5]:
    print(f"  {v}x: {k}")

# Show first 25 matches summary
print("\nFirst 25 matches:")
for i, m in enumerate(matches[:25]):
    print(f"  {i+1}. [{m.get('similarity_level')}] {m.get('feature_label')[:30]} | {m.get('patent_number')} | elem={m.get('element_id')} | score={m.get('similarity_score')}")

# Now check which ones make it to sorted_matches
def _sort_key(m):
    order = {'HIGH': 0, 'MODERATE': 1, 'LOW': 2}
    return (order.get(m.get('overall_confidence', m.get('similarity_level', 'LOW')), 2),
            -float(m.get('similarity_score', 0)))
sorted_matches = sorted(matches, key=_sort_key)[:20]
print(f"\nTop 20 sorted matches:")
for i, m in enumerate(sorted_matches):
    print(f"  {i+1}. [{m.get('overall_confidence', m.get('similarity_level', ''))}] {m.get('feature_label')[:30]} | {m.get('patent_number')} | score={m.get('similarity_score')}")

# Count feature_label occurrences in top 20  
feat_counts = Counter(m.get("feature_label") for m in sorted_matches)
print(f"\nFeature label counts in top 20:")
for k, v in feat_counts.most_common():
    print(f"  {v}x: {k}")
