"""Verify Issue 4: Gemini contextual analysis appears in PDF."""
import fitz, glob, re

pdf = sorted(glob.glob('reports/solar_charger_report_*.pdf'))[-1]
doc = fitz.open(pdf)
text = "\n".join(p.get_text() for p in doc)

has_requires = 'What This Claim Requires' in text
has_analysis = 'Comparison Analysis' in text
has_distinctions = 'Key Technical Distinctions' in text
has_generic_rec = 'No strong overlap detected' in text

print(f'PDF: {pdf}')
print(f'Has "What This Claim Requires": {has_requires}')
print(f'Has "Comparison Analysis": {has_analysis}')
print(f'Has "Key Technical Distinctions": {has_distinctions}')
print(f'Has generic rec text: {has_generic_rec}')
print()

n_requires = len(re.findall(r'What This Claim Requires', text))
n_analysis = len(re.findall(r'Comparison Analysis', text))
print(f'  "What This Claim Requires" count: {n_requires}')
print(f'  "Comparison Analysis" count: {n_analysis}')
print()

# Also check Section 4 table
has_section4 = '4. Claim Element Comparison' in text
has_cmat_data = 'AI Conf.' in text
print(f'Has Section 4 header: {has_section4}')
print(f'Has AI Conf. column: {has_cmat_data}')

# Check overall confidence levels in text
for level in ['HIGH', 'MODERATE', 'LOW']:
    count = text.count(level)
    print(f'  "{level}" appears: {count} times')

print()
passed = has_requires and has_analysis and not has_generic_rec
print(f'GEMINI ANALYSIS PASS: {passed}')
