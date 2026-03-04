#!/usr/bin/env python3
"""Verify Issue 2: Duplicate matches"""
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

def main():
    import pandas as pd
    from modules.report_generator import ReportGenerator

    with open("examples/solar_charger_session.json") as f:
        session_data = json.load(f)

    if isinstance(session_data.get("detail_patents"), list):
        session_data["detail_patents"] = pd.DataFrame(session_data["detail_patents"])
    if isinstance(session_data.get("landscape_patents"), list):
        session_data["landscape_patents"] = pd.DataFrame(session_data["landscape_patents"])

    rg = ReportGenerator()
    pdf_bytes = rg.generate(session_data)

    pdf_path = "examples/test_issue2.pdf"
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)
    print(f"PDF written: {pdf_path} ({len(pdf_bytes)/1024:.1f} KB)")

    pdf_text = extract_pdf_text(pdf_path)

    # Check match headers
    match_headers = re.findall(r'Match \d+: (.+?)(?:\n|$)', pdf_text)
    from collections import Counter
    mc = Counter(match_headers)
    max_dupes = max(mc.values()) if mc else 0
    print(f"Total match headers: {len(match_headers)}")
    print(f"Max repeat count: {max_dupes}")
    
    for k, v in sorted(mc.items(), key=lambda x: -x[1]):
        print(f"  {v}x: {k[:60]}")

    if max_dupes <= 3:
        print("\nISSUE 2: PASS - No excessive duplicates")
    else:
        print(f"\nISSUE 2: FAIL - Max repeat = {max_dupes}")

if __name__ == "__main__":
    main()
