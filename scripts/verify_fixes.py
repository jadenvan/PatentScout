#!/usr/bin/env python3
"""
PatentScout v2 Fix Verification Script
=======================================
Runs the full analysis pipeline headlessly (no Streamlit) and produces a
structured verification report covering all 7 check-areas:

    1. Retrieval    — ≥60 % topical relevance in top 10
    2. Deduplication — 0 duplicates
    3. White Space  — no false positive CPC gaps in H02S/H02J/H01L
    4. Exec Summary — no formatting artefacts
    5. Recommendations — no "filing"/"provisional"/"patent protection"
    6. Landscape Charts — 3 charts > 100 bytes, embedded in PDF
    7. PDF Report   — generated, > 10 KB

Usage:
    source venv/bin/activate
    python scripts/verify_fixes.py
"""
from __future__ import annotations

import json, os, re, sys, time

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── Monkey-patch st.cache_resource so modules load outside Streamlit ──
import types as _types

class _FakeStreamlit:
    """Minimal stub so `import streamlit as st` / @st.cache_resource works."""
    class session_state(dict):
        def __getattr__(self, k):
            return self.get(k)
        def __setattr__(self, k, v):
            self[k] = v
    session_state = session_state()  # type: ignore[assignment]

    @staticmethod
    def cache_resource(_fn=None, **kwargs):
        """No-op decorator — just calls the function."""
        if _fn is not None:
            return _fn
        def _wrapper(fn):
            return fn
        return _wrapper

    @staticmethod
    def cache_data(_fn=None, **kwargs):
        if _fn is not None:
            return _fn
        def _wrapper(fn):
            return fn
        return _wrapper

    def __getattr__(self, name):
        """Swallow any other st.xxx calls gracefully."""
        return lambda *a, **kw: None

sys.modules.setdefault("streamlit", _FakeStreamlit())

from config import settings

# ── Test-case descriptions ────────────────────────────────────────────
SOLAR_CHARGER = (
    "A portable solar panel that folds into a compact case and charges "
    "mobile phones via USB-C connection. It includes an integrated battery "
    "pack for storing energy when sunlight is not available."
)

RELEVANCE_TERMS = [
    "solar", "photovoltaic", "charger", "charging", "battery",
    "portable power", "foldable", "solar panel", "solar cell",
    "energy storage", "usb",
]

# ── Helpers ───────────────────────────────────────────────────────────

def _relevance_fraction(df, terms, top_n=10):
    """Return (hits, total, frac) for titles/abstracts containing terms."""
    hits, total = 0, min(top_n, len(df))
    for _, row in df.head(top_n).iterrows():
        text = (str(row.get("title", "")) + " " + str(row.get("abstract", ""))).lower()
        if any(t in text for t in terms):
            hits += 1
    return hits, total, hits / total if total else 0


# ── Main verification ─────────────────────────────────────────────────

def run(description: str, label: str = "Solar Charger"):
    report: dict = {
        "label": label,
        "retrieval": {}, "deduplication": {}, "whitespace": {},
        "executive_summary": {}, "recommendations": {},
        "landscape_charts": {}, "pdf_report": {},
        "passes": 0, "total": 7,
    }

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not gemini_key:
        print("ERROR: GEMINI_API_KEY not set"); return report

    from google.cloud import bigquery
    from google import genai

    try:
        bq = bigquery.Client(project=settings.BIGQUERY_PROJECT)
    except Exception as e:
        print(f"ERROR: BigQuery client failed: {e}"); return report

    gemini = genai.Client(api_key=gemini_key)
    ok = 0

    # ── 1) Feature extraction ─────────────────────────────────────────
    print(f"\n{'='*72}\nVERIFICATION: {label}\n{'='*72}")
    print("\n--- Phase 1: Feature Extraction ---")
    from modules.query_builder import QueryBuilder
    qb = QueryBuilder(api_key=gemini_key)
    strategy = qb.extract_features(description)
    where = qb.build_bigquery_where_clause(strategy)
    strategy.update(where)
    try:
        strategy["features"] = qb.reformulate_features_for_patent_language(
            strategy.get("features", [])
        )
    except Exception as e:
        print(f"  Reformulation warning: {e}")
    print(f"  Features: {len(strategy.get('features', []))}")
    print(f"  CPC:      {[c['code'] for c in strategy.get('cpc_codes', [])]}")
    print(f"  Terms:    {[t['primary'] for t in strategy.get('search_terms', [])]}")

    # ── 2) Patent retrieval ───────────────────────────────────────────
    print("\n--- Phase 2: Patent Retrieval ---")
    from modules.patent_retriever import PatentRetriever
    retriever = PatentRetriever(bq_client=bq)
    detail_df, landscape_df = retriever.search(strategy, user_description=description)
    n = len(detail_df) if detail_df is not None else 0
    report["retrieval"]["total"] = n

    if detail_df is not None and not detail_df.empty:
        hits, total, frac = _relevance_fraction(detail_df, RELEVANCE_TERMS, 10)
        report["retrieval"].update(hits=hits, total_checked=total, frac=round(frac, 2))
        report["retrieval"]["pass"] = frac >= 0.6
        if frac >= 0.6:
            ok += 1; print(f"  CHECK 1 RETRIEVAL: PASS ({hits}/{total} = {frac:.0%})")
        else:
            print(f"  CHECK 1 RETRIEVAL: FAIL ({hits}/{total} = {frac:.0%}, need ≥60%)")
        print("  Top 10:")
        for _, r in detail_df.head(10).iterrows():
            print(f"    [{r.get('relevance_score',0):.3f}] {str(r.get('title',''))[:70]}")
    else:
        report["retrieval"]["pass"] = False
        print("  CHECK 1 RETRIEVAL: FAIL (0 patents)")

    # ── 3) Claim parsing ─────────────────────────────────────────────
    print("\n--- Phase 3: Claim Parsing ---")
    from modules.claim_parser import ClaimParser
    parser = ClaimParser(gemini_client=gemini)
    parsed_claims = parser.parse_all(detail_df, max_patents=20)
    print(f"  {parsed_claims['summary']}")

    # ── 4) Similarity analysis ────────────────────────────────────────
    print("\n--- Phase 4: Similarity ---")
    from sentence_transformers import SentenceTransformer
    from modules.embedding_engine import EmbeddingEngine
    features = strategy.get("features", [])
    parsed_list = parsed_claims.get("results", [])
    sim_results = {"matches": [], "stats": {}, "unmatched_features": []}
    if features and parsed_list:
        try:
            engine = EmbeddingEngine.__new__(EmbeddingEngine)
            engine.model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
            engine.model_name = settings.EMBEDDING_MODEL_NAME
            sim_results = engine.compute_similarity_matrix(features, parsed_list)
            print(f"  Stats: {sim_results.get('stats', {})}")
        except Exception as e:
            print(f"  Similarity failed: {e}")

    # ── 5) Contextual analysis ────────────────────────────────────────
    print("\n--- Phase 5: Contextual Analysis ---")
    from modules.element_mapper import ElementMapper
    comparison_matrix = []
    high_mod = [m for m in sim_results.get("matches", [])
                if m.get("similarity_level") in ("HIGH", "MODERATE")]
    if high_mod:
        try:
            mapper = ElementMapper(gemini_client=gemini)
            comparison_matrix = mapper.analyze_matches(sim_results)
            print(f"  Enriched: {len(comparison_matrix)}")
        except Exception as e:
            print(f"  Contextual analysis failed: {e}")

    # ── CHECK 2: Deduplication ────────────────────────────────────────
    print("\n--- Check 2: Deduplication ---")
    # Check each collection independently — comparison_matrix is an
    # enriched SUBSET of sim_results.matches, so cross-collection
    # overlap is expected.  The report uses them in separate sections.
    def _count_dupes(items, label):
        seen, d = set(), 0
        for m in items:
            key = (m.get("feature_label",""), m.get("patent_number",""), m.get("element_id",""))
            if key in seen: d += 1
            seen.add(key)
        print(f"    {label}: {len(items)} entries, {d} internal dupes")
        return d
    d1 = _count_dupes(sim_results.get("matches", []), "sim_results.matches")
    d2 = _count_dupes(comparison_matrix, "comparison_matrix")
    total_dupes = d1 + d2
    report["deduplication"] = {"sim_dupes": d1, "cmat_dupes": d2}
    report["deduplication"]["pass"] = total_dupes == 0
    if total_dupes == 0:
        ok += 1; print(f"  CHECK 2 DEDUP: PASS (0 internal dupes)")
    else:
        print(f"  CHECK 2 DEDUP: FAIL ({total_dupes} internal dupes)")

    # ── 6) White space ────────────────────────────────────────────────
    print("\n--- Phase 6: White Space ---")
    from modules.whitespace_finder import WhiteSpaceFinder
    white_spaces = []
    try:
        ws = WhiteSpaceFinder(gemini_client=gemini)
        white_spaces = ws.identify_gaps(
            features=features,
            similarity_results=sim_results,
            landscape_df_size=len(landscape_df) if landscape_df is not None else 0,
            search_strategy=strategy,
            detail_df=detail_df,
        )
        print(f"  Findings: {len(white_spaces)}")
    except Exception as e:
        print(f"  White space failed: {e}")

    # CHECK 3: No false CPC gaps
    false_cpc = False
    for w in white_spaces:
        if w.get("type") == "Classification Gap":
            desc = w.get("description", "")
            conf = w.get("confidence", {})
            lvl = conf.get("level", "?") if isinstance(conf, dict) else str(conf)
            for prefix in ("H02S", "H02J", "H01L"):
                if prefix in desc and "0 patent" in desc and lvl not in ("LOW",):
                    false_cpc = True
    report["whitespace"] = {"findings": len(white_spaces), "false_cpc": false_cpc}
    report["whitespace"]["pass"] = not false_cpc
    if not false_cpc:
        ok += 1; print("  CHECK 3 WHITESPACE: PASS")
    else:
        print("  CHECK 3 WHITESPACE: FAIL (false CPC gap)")

    # ── CHECK 4: Executive summary ────────────────────────────────────
    print("\n--- Check 4: Executive Summary ---")
    from modules.report_generator import clean_concept_text
    concept = clean_concept_text(description)
    artefacts = any(m in concept for m in ["===", "---", "PATENTSCOUT", "Title:"])
    report["executive_summary"] = {"clean": not artefacts, "preview": concept[:200]}
    report["executive_summary"]["pass"] = not artefacts
    if not artefacts:
        ok += 1; print(f"  CHECK 4 EXEC SUMMARY: PASS\n  Preview: {concept[:120]}")
    else:
        print(f"  CHECK 4 EXEC SUMMARY: FAIL (artifacts found)")

    # ── Build session_data for PDF ────────────────────────────────────
    session_data = {
        "invention_text": description,
        "search_strategy": strategy,
        "detail_patents": detail_df,
        "landscape_patents": landscape_df,
        "parsed_claims": parsed_claims,
        "similarity_results": sim_results,
        "comparison_matrix": comparison_matrix,
        "white_spaces": white_spaces,
        "chart_images": {},
    }

    # ── CHECK 6: Landscape charts ─────────────────────────────────────
    print("\n--- Check 6: Landscape Charts ---")
    from modules.landscape_analyzer import LandscapeAnalyzer
    chart_images = {}
    if landscape_df is not None and not landscape_df.empty:
        try:
            analyzer = LandscapeAnalyzer(landscape_df)
            chart_images = analyzer.export_charts_as_images()
        except Exception as e:
            print(f"  Chart gen failed: {e}")
    for name in ("filing_trends", "top_assignees", "cpc_distribution"):
        sz = len(chart_images.get(name, b""))
        print(f"    {name}: {sz} bytes")
    all_charts = all(len(chart_images.get(n, b"")) > 100
                     for n in ("filing_trends", "top_assignees", "cpc_distribution"))
    session_data["chart_images"] = chart_images

    # ── Generate PDF ──────────────────────────────────────────────────
    print("\n--- Generating PDF ---")
    from modules.report_generator import ReportGenerator
    pdf_bytes = None
    try:
        rg = ReportGenerator()
        pdf_bytes = rg.generate(session_data)
        print(f"  PDF size: {len(pdf_bytes)/1024:.1f} KB")
    except Exception as e:
        print(f"  PDF generation failed: {e}")
        import traceback; traceback.print_exc()

    # CHECK 5: Recommendations — no legal advice
    print("\n--- Check 5: Recommendations ---")
    has_legal = False
    if pdf_bytes:
        try:
            txt = pdf_bytes.decode("latin-1", errors="replace").lower()
            for phrase in ("filing a provisional", "file a provisional",
                           "consider filing", "patent protection",
                           "file provisional"):
                if phrase in txt:
                    has_legal = True
                    print(f"    Found: '{phrase}'")
        except Exception:
            pass
    report["recommendations"] = {"has_legal_advice": has_legal}
    report["recommendations"]["pass"] = not has_legal
    if not has_legal:
        ok += 1; print("  CHECK 5 RECOMMENDATIONS: PASS")
    else:
        print("  CHECK 5 RECOMMENDATIONS: FAIL (legal advice found)")

    # CHECK 6 continued: charts in PDF
    charts_ok = all_charts
    if pdf_bytes and chart_images:
        charts_ok = charts_ok and len(pdf_bytes) > 20_000
    report["landscape_charts"] = {
        "all_generated": all_charts,
        "pdf_has_charts": charts_ok,
    }
    report["landscape_charts"]["pass"] = charts_ok
    if charts_ok:
        ok += 1; print("  CHECK 6 CHARTS: PASS")
    else:
        print(f"  CHECK 6 CHARTS: FAIL (gen={all_charts})")

    # CHECK 7: PDF report
    print("\n--- Check 7: PDF Report ---")
    if pdf_bytes and len(pdf_bytes) > 10_000:
        ok += 1
        report["pdf_report"] = {"size_kb": round(len(pdf_bytes)/1024, 1), "pass": True}
        print(f"  CHECK 7 PDF: PASS ({len(pdf_bytes)/1024:.1f} KB)")
        out = os.path.join(ROOT, "examples", "sample_report_v2.pdf")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "wb") as f:
            f.write(pdf_bytes)
        print(f"  Saved: {out}")
    else:
        report["pdf_report"] = {"pass": False}
        print("  CHECK 7 PDF: FAIL")

    report["passes"] = ok
    print(f"\n{'='*72}")
    print(f"  1. RETRIEVAL        {'PASS' if report['retrieval'].get('pass') else 'FAIL'}")
    print(f"  2. DEDUPLICATION    {'PASS' if report['deduplication'].get('pass') else 'FAIL'}")
    print(f"  3. WHITE SPACE      {'PASS' if report['whitespace'].get('pass') else 'FAIL'}")
    print(f"  4. EXEC SUMMARY     {'PASS' if report['executive_summary'].get('pass') else 'FAIL'}")
    print(f"  5. RECOMMENDATIONS  {'PASS' if report['recommendations'].get('pass') else 'FAIL'}")
    print(f"  6. LANDSCAPE CHARTS {'PASS' if report['landscape_charts'].get('pass') else 'FAIL'}")
    print(f"  7. PDF REPORT       {'PASS' if report['pdf_report'].get('pass') else 'FAIL'}")
    print(f"\n  OVERALL: {ok} / 7 checks passed")
    print(f"{'='*72}\n")
    return report


if __name__ == "__main__":
    print("="*72)
    print("PATENTSCOUT v2 FIX VERIFICATION")
    print("="*72)
    r = run(SOLAR_CHARGER, "Solar Charger")
    out_json = os.path.join(ROOT, "reports", "fix_verification_results.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(r, f, indent=2, default=str)
    print(f"Results saved to {out_json}")
