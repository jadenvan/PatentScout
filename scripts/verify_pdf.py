#!/usr/bin/env python3
"""Generate a test PDF report from cached session and verify it."""

import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from datetime import datetime
from modules.report_generator import ReportGenerator

with open("examples/solar_charger_session.json") as f:
    session = json.load(f)

rg = ReportGenerator()
pdf = rg.generate(session)

ts = datetime.now().strftime("%Y%m%dT%H%M%SZ")
pdf_path = f"examples/{ts}_patentscout_report.pdf"
with open(pdf_path, "wb") as f:
    f.write(pdf)
print(f"PDF saved: {pdf_path} ({len(pdf)} bytes)")

import pdfplumber
with pdfplumber.open(pdf_path) as p:
    text = "\n".join((pg.extract_text() or "") for pg in p.pages)
    n_pages = len(p.pages)

patents = re.findall(r"US[\-]?\d{5,}", text)
checks = {
    "TOC": "Table of Contents" in text,
    "Feature": "Feature" in text,
    "Claim Element": "Claim Element" in text,
    "Recommendation": "Recommendation" in text,
    "Embedding Similarity": "Embedding Similarity" in text,
    ">=5 patents": len(patents) >= 5,
}

print(f"Pages: {n_pages}")
print(f"Patent numbers found: {len(patents)}")
for k, v in checks.items():
    print(f"  {k}: {'PASS' if v else 'FAIL'}")

if all(checks.values()):
    print("ALL CHECKS PASSED")
else:
    print("SOME CHECKS FAILED")
    sys.exit(1)
