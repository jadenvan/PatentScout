#!/usr/bin/env python3
"""Debug: inspect white space section content from PDFs."""
import pdfplumber, re, sys, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for pdf_path in ["reports/patentscout_solar_charger.pdf", "reports/patentscout_smart_doorbell.pdf"]:
    print(f"\n{'='*60}")
    print(f"PDF: {pdf_path}")
    print(f"{'='*60}")
    with pdfplumber.open(pdf_path) as pdf:
        full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    if "White Space Analysis" in full_text:
        ws = full_text.split("White Space Analysis")[1]
        if "Recommended Next Steps" in ws:
            ws = ws.split("Recommended Next Steps")[0]
        else:
            ws = ws[:4000]
        print("WHITE SPACE SECTION (first 2500 chars):")
        print(ws[:2500])
        print("---")
        print(f"Has % char: {'%' in ws}")
        print(f"Has 'percent': {'percent' in ws.lower()}")
        pcts = re.findall(r'(\d+)%', ws)
        print(f"Percentages found: {pcts}")
    else:
        print("NO 'White Space Analysis' heading found!")
        # Check for variations
        for kw in ["White Space", "Innovation Opportunities", "Gap Analysis",
                    "Opportunity", "Coverage", "Under-represented"]:
            if kw.lower() in full_text.lower():
                idx = full_text.lower().index(kw.lower())
                print(f"  Found '{kw}' near pos {idx}")
                print(f"  Context: ...{full_text[max(0,idx-50):idx+200]}...")
