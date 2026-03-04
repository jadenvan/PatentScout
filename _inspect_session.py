#!/usr/bin/env python3
"""Quick inspection of solar_charger_session.json"""
import json

with open('examples/solar_charger_session.json', 'r') as f:
    d = json.load(f)

print('Top keys:', list(d.keys()))

dp = d.get('detail_patents', [])
print(f'detail_patents count: {len(dp) if isinstance(dp, list) else "not a list"}')
if isinstance(dp, list) and len(dp) > 0:
    p0 = dp[0]
    print('Patent 0 keys:', list(p0.keys()))
    print('Patent 0 publication_number:', p0.get('publication_number', ''))
    print('Patent 0 title:', str(p0.get('title', ''))[:100])
    print('Patent 0 assignee_name:', repr(p0.get('assignee_name', '')))
    print('Patent 0 claims_text[:200]:', str(p0.get('claims_text', ''))[:200])
    print('Patent 0 relevance_score:', p0.get('relevance_score', ''))
    print('Patent 0 cpc_code:', str(p0.get('cpc_code', ''))[:100])
    print('Patent 0 filing_date:', p0.get('filing_date', ''))
    # Check a few more patents for diversity
    for i in [0, 5, 10, 15]:
        if i < len(dp):
            pi = dp[i]
            print(f'  Patent {i}: assignee={repr(pi.get("assignee_name",""))}, title={str(pi.get("title",""))[:60]}, rel_score={pi.get("relevance_score","")}')

lp = d.get('landscape_patents', [])
print(f'landscape_patents count: {len(lp) if isinstance(lp, list) else "not a list"}')

sr = d.get('similarity_results', {})
matches = sr.get('matches', [])
print(f'similarity matches count: {len(matches)}')
if matches:
    print('Match 0 keys:', list(matches[0].keys()))
    for i in [0, 5, 10, 15]:
        if i < len(matches):
            mi = matches[i]
            print(f'  Match {i}: feature={mi.get("feature_label","")}, patent={mi.get("patent_number","")}, score={mi.get("similarity_score","")}, level={mi.get("similarity_level","")}, element_text[:80]={str(mi.get("element_text",""))[:80]}')

cm = d.get('comparison_matrix', [])
print(f'comparison_matrix count: {len(cm)}')
if cm:
    print('CM 0 keys:', list(cm[0].keys()))

ws = d.get('white_spaces', [])
print(f'white_spaces count: {len(ws)}')

ss = d.get('search_strategy', {})
print(f'search_strategy features count: {len(ss.get("features", []))}')
for f in ss.get('features', []):
    print(f'  Feature: {f.get("label", "")}')

print(f'invention_text[:200]: {str(d.get("invention_text", ""))[:200]}')
print(f'chart_images keys: {list(d.get("chart_images", {}).keys()) if isinstance(d.get("chart_images"), dict) else "N/A"}')
