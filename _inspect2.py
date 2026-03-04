#!/usr/bin/env python3
import json

with open('/Users/jack/Documents/PatentScout/examples/solar_charger_session.json', 'r') as f:
    d = json.load(f)

cm = d.get('comparison_matrix', [])
if cm:
    c0 = cm[0]
    print('CM[0] full:')
    for k, v in c0.items():
        print(f'  {k}: {str(v)[:120]}')
    print()
    if len(cm) > 1:
        c1 = cm[1]
        print('CM[1]:')
        print(f'  feature_label: {c1.get("feature_label","")}')
        print(f'  gemini_explanation: {str(c1.get("gemini_explanation",""))[:150]}')
        print(f'  gemini_distinction: {str(c1.get("gemini_distinction",""))[:150]}')

ws = d.get('white_spaces', [])
for w in ws:
    print(f'WS: type={w.get("type","")}, title={w.get("title","")[:60]}, conf={w.get("confidence",{})}')

lps = d['landscape_patents']
assignees = set()
for lp in lps:
    a = lp.get('assignee_name', '')
    assignees.add(str(a)[:40])
print(f'Unique landscape assignees: {len(assignees)}')
for a in list(assignees)[:10]:
    print(f'  {a}')
