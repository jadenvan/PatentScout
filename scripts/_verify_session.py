#!/usr/bin/env python3
"""Inspect cached session data and generate/verify PDF."""
import json
import os
import sys
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SESSION_FILE = "examples/solar_charger_session.json"


def extract_pdf_text(pdf_path):
    """Extract all text from a PDF file."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except ImportError:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text


def count_pdf_images(pdf_path):
    """Count images in a PDF file."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        count = 0
        for page in doc:
            count += len(page.get_images(full=True))
        doc.close()
        return count
    except ImportError:
        return -1


def inspect_session():
    print(f"Loading {SESSION_FILE}...")
    with open(SESSION_FILE) as f:
        s = json.load(f)

    print(f"Keys: {list(s.keys())}")
    print(f"detail_patents count: {len(s.get('detail_patents', []))}")
    print(f"landscape_patents count: {len(s.get('landscape_patents', []))}")
    print(f"comparison_matrix count: {len(s.get('comparison_matrix', []))}")
    print(f"white_spaces count: {len(s.get('white_spaces', []))}")
    sr = s.get("similarity_results", {})
    print(f"similarity_results matches: {len(sr.get('matches', []))}")
    print(f"stats: {sr.get('stats', {})}")

    # Check first few patent titles
    print("\nFirst 10 patent titles:")
    for i, p in enumerate(s["detail_patents"][:10]):
        t = p.get("title", "N/A")[:70]
        print(f"  {i+1}. {t}")

    # Check comparison_matrix first entry
    cm = s.get("comparison_matrix", [])
    if cm:
        m = cm[0]
        print(f"\nFirst CM entry keys: {list(m.keys())}")
        print(f"  gemini_explanation: {repr(m.get('gemini_explanation', 'MISSING')[:100])}")
        print(f"  gemini_assessment: {repr(m.get('gemini_assessment', 'MISSING')[:100])}")
        print(f"  key_distinctions: {m.get('key_distinctions', 'MISSING')}")
    else:
        print("\nNo comparison_matrix entries!")

    # Check white_spaces
    ws = s.get("white_spaces", [])
    print(f"\nWhite spaces ({len(ws)}):")
    for w in ws[:3]:
        desc = w.get("description", "")[:120]
        print(f"  {w.get('type', '')} - {w.get('title', '')}")
        print(f"    {desc}")

    # Check retrieval topicality
    solar_terms = ["solar", "photovoltaic", "charger", "charging", 
                   "battery pack", "energy storage", "power bank", "solar cell", "solar panel"]
    solar_count = 0
    total = min(20, len(s["detail_patents"]))
    for p in s["detail_patents"][:20]:
        text = (str(p.get("title", "")).lower() + " " + str(p.get("abstract", "")).lower())
        is_solar = any(t in text for t in solar_terms)
        solar_count += int(is_solar)
        marker = "Y" if is_solar else "N"
        print(f"  [{marker}] {p.get('title', 'N/A')[:60]}")
    print(f"\nSolar-related in top {total}: {solar_count}/{total}")


def generate_and_verify_pdf():
    """Generate PDF from cache and verify content."""
    import pandas as pd
    from modules.report_generator import ReportGenerator

    print(f"\nLoading {SESSION_FILE}...")
    with open(SESSION_FILE) as f:
        session_data = json.load(f)

    if isinstance(session_data.get("detail_patents"), list):
        session_data["detail_patents"] = pd.DataFrame(session_data["detail_patents"])
    if isinstance(session_data.get("landscape_patents"), list):
        session_data["landscape_patents"] = pd.DataFrame(session_data["landscape_patents"])

    print("Generating PDF...")
    rg = ReportGenerator()
    pdf_bytes = rg.generate(session_data)

    pdf_path = "examples/test_solar_current.pdf"
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)
    print(f"PDF written: {pdf_path} ({len(pdf_bytes)/1024:.1f} KB)")

    # Verify
    pdf_text = extract_pdf_text(pdf_path)
    image_count = count_pdf_images(pdf_path)

    print(f"\n=== PDF VERIFICATION ===")
    print(f"PDF text length: {len(pdf_text)} chars")
    print(f"Images in PDF: {image_count}")

    # Check charts
    has_png_fallback = "PNG export was not available" in pdf_text
    has_viz_fallback = "available in the interactive dashboard" in pdf_text
    print(f"Has 'PNG not available' message: {has_png_fallback}")
    print(f"Has 'interactive dashboard' fallback: {has_viz_fallback}")

    # Check Gemini analysis
    has_claim_requires = "What This Claim Requires" in pdf_text
    has_comparison = "Comparison Analysis" in pdf_text
    has_distinctions = "Key Technical Distinctions" in pdf_text
    has_generic = "No strong overlap detected" in pdf_text
    print(f"'What This Claim Requires': {has_claim_requires}")
    print(f"'Comparison Analysis': {has_comparison}")
    print(f"'Key Technical Distinctions': {has_distinctions}")
    print(f"Generic recommendation present: {has_generic}")

    # Check Section 4
    has_empty_sec4 = "Comparison matrix data not available" in pdf_text
    print(f"Section 4 empty message: {has_empty_sec4}")

    # Check white space corpus refs
    corpus_refs = re.findall(r"(\d+) patents? retrieved", pdf_text)
    print(f"Corpus sizes mentioned in whitespace: {corpus_refs}")

    # Check match duplicates
    match_headers = re.findall(r"Match \d+: (.+?)(?:\n|$)", pdf_text)
    from collections import Counter
    mc = Counter(match_headers)
    max_dupes = max(mc.values()) if mc else 0
    print(f"Match headers found: {len(match_headers)}, max repeat: {max_dupes}")
    if max_dupes > 3:
        dupes = {k: v for k, v in mc.items() if v > 3}
        print(f"Excess duplicates: {dupes}")


if __name__ == "__main__":
    if "--pdf" in sys.argv:
        generate_and_verify_pdf()
    else:
        inspect_session()
        generate_and_verify_pdf()
