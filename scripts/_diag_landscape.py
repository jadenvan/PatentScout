"""Quick diagnostic: check how many patents match domain-specific patterns."""
import json, re, pandas as pd

with open('tests/_strategy_cache.json') as f:
    s = json.load(f)

patterns = []
for st in s.get('search_terms', [])[:3]:
    rx = st.get('bigquery_regex', '')
    if rx:
        patterns.append(re.compile(rx, re.IGNORECASE))
        print(f'Pattern: {rx}')

# Also add patterns 4 and 5
for st in s.get('search_terms', [])[3:]:
    rx = st.get('bigquery_regex', '')
    if rx:
        print(f'Skipped pattern: {rx}')

with open('examples/solar_charger_session.json') as f:
    session = json.load(f)
detail = session.get('detail_patents', [])
df = pd.DataFrame(detail) if detail else pd.DataFrame()
print(f'Total patents: {len(df)}')

if not df.empty and patterns:
    def match(row):
        text = str(row.get('title', '')) + ' ' + str(row.get('abstract', ''))
        return any(p.search(text) for p in patterns)
    mask = df.apply(match, axis=1)
    print(f'Matching first 3 patterns: {mask.sum()}')
    for _, row in df[mask].head(15).iterrows():
        print(f"  {row.get('title', '??')[:70]}")
    
    # Also check with ALL 5 patterns
    all_patterns = []
    for st in s.get('search_terms', []):
        rx = st.get('bigquery_regex', '')
        if rx:
            all_patterns.append(re.compile(rx, re.IGNORECASE))
    mask_all = df.apply(lambda r: any(p.search(str(r.get('title',''))+' '+str(r.get('abstract',''))) for p in all_patterns), axis=1)
    print(f'\nMatching ALL 5 patterns: {mask_all.sum()}')
    
    # Check which pattern each match hits
    for _, row in df[mask_all & ~mask].head(10).iterrows():
        text = str(row.get('title', '')) + ' ' + str(row.get('abstract', ''))
        for i, p in enumerate(all_patterns):
            if p.search(text):
                print(f"  Pattern {i}: {row.get('title', '??')[:60]}")
