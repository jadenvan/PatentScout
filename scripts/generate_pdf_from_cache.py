#!/usr/bin/env python3
"""Generate a PDF report from a cached session file (no BigQuery charges)."""

import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.report_generator import ReportGenerator

SESSION_FILE = "examples/solar_charger_session.json"
OUTPUT_DIR   = "reports"

def main():
    print(f"Loading session from {SESSION_FILE}...")
    with open(SESSION_FILE) as f:
        session_data = json.load(f)

    # Convert detail_patents list-of-dicts to DataFrame for report generator
    import pandas as pd
    if "detail_patents" in session_data and isinstance(session_data["detail_patents"], list):
        session_data["detail_patents"] = pd.DataFrame(session_data["detail_patents"])
    if "landscape_patents" in session_data and isinstance(session_data["landscape_patents"], list):
        session_data["landscape_patents"] = pd.DataFrame(session_data["landscape_patents"])

    print(f"  detail_patents: {len(session_data['detail_patents'])} patents")
    print(f"  search_strategy features: {len(session_data['search_strategy'].get('features', []))}")
    print(f"  similarity_results matches: {len(session_data['similarity_results'].get('matches', []))}")
    print(f"  comparison_matrix entries: {len(session_data.get('comparison_matrix', []))}")
    print(f"  white_spaces: {len(session_data.get('white_spaces', []))}")

    print("\nGenerating PDF...")
    rg = ReportGenerator()
    pdf_bytes = rg.generate(session_data)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(OUTPUT_DIR, f"solar_charger_report_{ts}.pdf")

    with open(out_path, "wb") as f:
        f.write(pdf_bytes)

    size_kb = len(pdf_bytes) / 1024
    print(f"\nPDF written: {out_path} ({size_kb:.1f} KB)")

if __name__ == "__main__":
    main()
