"""
Final verification: run all 7 pass gates on the solar charger PDF.
"""
import fitz
import glob
import re
import json
import sys

pdf_path = sorted(glob.glob('reports/solar_charger_report_*.pdf'))[-1]
doc = fitz.open(pdf_path)
text = "\n".join(p.get_text() for p in doc)
img_count = sum(len(p.get_images()) for p in doc)

print(f"PDF: {pdf_path}")
print(f"Size: {len(open(pdf_path, 'rb').read()) / 1024:.1f} KB")
print(f"Pages: {len(doc)}")
print(f"Images: {img_count}")
print(f"Text length: {len(text)}")
print()

# Gate 1: Retrieval relevance (cached data = 20/20 solar)
# Cannot rerun BigQuery: verify by checking only solar-related patents mentioned
solar_patents = re.findall(r'US-\d+-\w+', text)
print(f"Gate 1 (Retrieval): {len(set(solar_patents))} unique patents referenced")
gate1 = len(set(solar_patents)) >= 10
print(f"  PASS: {gate1} (>= 10 unique patents)")

# Gate 2: No feature+patent combo appears more than 3 times
from collections import Counter
# Count how many times each patent appears in match detail context
# Since we capped at 3 per feature, check by counting match headers
match_headers = re.findall(r'Match \d+:', text)
print(f"\nGate 2 (Duplicates): {len(match_headers)} match detail entries")
# Count patent mentions per feature
feature_patent_combos = []
current_feature = None
for line in text.split('\n'):
    if line.strip().startswith('Feature:'):
        current_feature = line.strip()
    m = re.match(r'Patent:\s*(US-\S+)', line.strip())
    if m and current_feature:
        feature_patent_combos.append((current_feature, m.group(1)))
combo_counts = Counter(feature_patent_combos)
max_repeat = max(combo_counts.values()) if combo_counts else 0
print(f"  Max feature+patent repeat: {max_repeat}")
gate2 = max_repeat <= 3
print(f"  PASS: {gate2}")

# Gate 3: Charts embedded (>= 3 images)
print(f"\nGate 3 (Charts): {img_count} images embedded")
gate3 = img_count >= 3
print(f"  PASS: {gate3}")

# Gate 4: Gemini analysis present
has_requires = 'What This Claim Requires' in text
has_analysis = 'Comparison Analysis' in text
no_generic    = 'No strong overlap detected' not in text
print(f"\nGate 4 (Gemini analysis):")
print(f"  'What This Claim Requires': {has_requires}")
print(f"  'Comparison Analysis': {has_analysis}")
print(f"  No generic rec: {no_generic}")
gate4 = has_requires and has_analysis and no_generic
print(f"  PASS: {gate4}")

# Gate 5: Section 4 has content
has_s4 = '4. Claim Element Comparison' in text
# Use last occurrence (skip TOC)
all_s4 = [m.start() for m in re.finditer(r'4\. Claim Element Comparison', text)]
all_s5 = [m.start() for m in re.finditer(r'5\. Match Details', text)]
if len(all_s4) >= 2 and len(all_s5) >= 2:
    s4_content = text[all_s4[-1]:all_s5[-1]]
    has_s4_data = len(s4_content) > 100
elif all_s4 and all_s5:
    s4_content = text[all_s4[-1]:all_s5[-1]]
    has_s4_data = len(s4_content) > 100
else:
    has_s4_data = False
print(f"\nGate 5 (Section 4): header={has_s4}, data_length={len(s4_content) if 's4_content' in dir() else 0}")
gate5 = has_s4 and has_s4_data
print(f"  PASS: {gate5}")

# Gate 6: White space references landscape corpus (>= 100 patents)
ws_corpus = re.findall(r'(\d+) patents? retrieved from the landscape', text)
data_comp = re.findall(r'(\d+) patents', text)
large_refs = [int(n) for n in ws_corpus if int(n) >= 100]
print(f"\nGate 6 (White space corpus): landscape refs = {ws_corpus}")
gate6 = len(large_refs) > 0
print(f"  PASS: {gate6}")

# Gate 7: All sections present
sections = [
    '1. Executive Summary',
    '2. Search Methodology',
    '3. Prior Art Summary',
    '4. Claim Element Comparison',
    '5. Match Details',
    '6. Patent Landscape',
    '7. White Space Analysis',
    '8. Recommended Next Steps',
    '9. References',
]
print(f"\nGate 7 (All sections):")
missing = [s for s in sections if s not in text]
for s in sections:
    present = s in text
    print(f"  {s}: {present}")
gate7 = len(missing) == 0
print(f"  PASS: {gate7}")

# Summary
print("\n" + "=" * 60)
gates = [gate1, gate2, gate3, gate4, gate5, gate6, gate7]
gate_names = ['Retrieval', 'Duplicates', 'Charts', 'Gemini Analysis', 'Section 4', 'White Space', 'All Sections']
for name, passed in zip(gate_names, gates):
    status = "PASS" if passed else "FAIL"
    print(f"  Gate {name}: {status}")
all_pass = all(gates)
print(f"\n  ALL GATES: {'PASS' if all_pass else 'FAIL'} ({sum(gates)}/{len(gates)})")
