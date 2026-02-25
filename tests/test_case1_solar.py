"""
Phase 10 Integration Test — Test Case 1: Solar Charger (Full Pipeline)
Run with: venv/bin/python tests/test_case1_solar.py

Pass --cached to reuse the last saved strategy and skip Gemini extraction.

Logs per phase:
  - Feature extraction: model used, features/CPC/terms
  - BigQuery retrieval: rows returned, bytes processed per query
  - Claim parsing: patents attempted/successful, total claims parsed
  - Embedding similarity: HIGH/MODERATE/LOW match counts
  - Contextual analysis: enriched match count
  - White space: findings count and types
  - PDF generation: success / fail + file size
  - Total pipeline runtime
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("test_case1")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault(
    "GOOGLE_APPLICATION_CREDENTIALS",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "credentials", "service-account.json"),
)

# ---------------------------------------------------------------------------
# Test description (updated per task spec)
# ---------------------------------------------------------------------------
TEST_DESCRIPTION = (
    "A portable solar panel that folds into a compact case and charges mobile "
    "phones via USB-C connection. It includes an integrated battery pack for "
    "storing energy when sunlight is not available."
)
_CACHE_FILE = os.path.join(os.path.dirname(__file__), "_strategy_cache.json")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIVIDER = "=" * 70

def _phase(name: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {name}")
    print(DIVIDER)


def _elapsed(t0: float) -> str:
    return f"{time.time() - t0:.2f}s"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901
    pipeline_start = time.time()
    use_cache = "--cached" in sys.argv or "-c" in sys.argv

    # ── Env / config ─────────────────────────────────────────────────────
    from config import settings
    from google import genai
    from google.cloud import bigquery

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key and not use_cache:
        sys.exit("ERROR: GEMINI_API_KEY not set")

    project = settings.BIGQUERY_PROJECT
    if not project:
        sys.exit("ERROR: GOOGLE_CLOUD_PROJECT not set in .env")

    print(f"\nProject  : {project}")
    print(f"Model    : gemini-2.5-flash (primary)")
    print(f"Desc     : {TEST_DESCRIPTION[:100]}...")

    # Initialise single shared Gemini client
    gemini_client = genai.Client(api_key=api_key) if api_key else None

    # ── Phase 1: Feature extraction ───────────────────────────────────────
    _phase("PHASE 1 — Gemini Feature Extraction")
    t0 = time.time()

    from modules.query_builder import QueryBuilder
    strategy: dict = {}

    if use_cache and os.path.exists(_CACHE_FILE):
        print("  [CACHE] Loading strategy from cache (--cached)")
        with open(_CACHE_FILE) as f:
            strategy = json.load(f)
    else:
        qb = QueryBuilder(api_key=api_key)
        strategy = qb.extract_features(TEST_DESCRIPTION)
        clause = qb.build_bigquery_where_clause(strategy)
        strategy.update(clause)
        with open(_CACHE_FILE, "w") as f:
            json.dump(strategy, f, indent=2)
        print(f"  Strategy cached → {_CACHE_FILE}")

    if "text_filter" not in strategy:
        qb = QueryBuilder(api_key=api_key)
        clause = qb.build_bigquery_where_clause(strategy)
        strategy.update(clause)

    phase1_time = _elapsed(t0)
    print(f"  Features     : {len(strategy.get('features', []))}")
    print(f"  CPC codes    : {[c['code'] for c in strategy.get('cpc_codes', [])]}")
    print(f"  Search terms : {[t['primary'] for t in strategy.get('search_terms', [])]}")
    print(f"  Runtime      : {phase1_time}")
    print(f"\n  Final text_filter:\n{strategy['text_filter']}")

    # ── Phase 2: BigQuery Patent Retrieval ────────────────────────────────
    _phase("PHASE 2 — BigQuery Patent Retrieval")
    t0 = time.time()

    from modules.patent_retriever import PatentRetriever

    print(f"  Active BQ project : {project}")
    bq_client = bigquery.Client(project=project)
    retriever = PatentRetriever(bq_client=bq_client)

    detail_df, landscape_df = retriever.search(strategy)

    phase2_time = _elapsed(t0)
    print(f"\n  Detail rows    : {len(detail_df)}")
    print(f"  Landscape rows : {len(landscape_df)}")
    print(f"  Runtime        : {phase2_time}")

    if detail_df.empty:
        print("  [ERROR] detail_df is empty — cannot continue pipeline")
        print(f"  TOTAL: {time.time() - pipeline_start:.2f}s")
        sys.exit("FAIL: detail_df is empty after all BigQuery fallbacks")

    if not detail_df.empty:
        print(f"  Sample pub numbers : {detail_df['publication_number'].head(5).tolist()}")
        if "bytes_processed" in detail_df.columns:
            print(f"  Bytes processed (detail)    : {detail_df['bytes_processed'].iloc[0]:,}")
        if "bytes_processed" in landscape_df.columns and not landscape_df.empty:
            print(f"  Bytes processed (landscape) : {landscape_df['bytes_processed'].iloc[0]:,}")

    # ── Phase 3: Claim Parsing ────────────────────────────────────────────
    _phase("PHASE 3 — Claim Parsing")
    t0 = time.time()

    from modules.claim_parser import ClaimParser

    parser = ClaimParser(gemini_client=gemini_client)
    parse_output = parser.parse_all(detail_df, max_patents=20)
    parsed_claims = parse_output["results"]
    parse_summary = parse_output["summary"]

    phase3_time = _elapsed(t0)
    total_elements = sum(
        len(c.get("elements", []))
        for p in parsed_claims
        for c in p.get("independent_claims", [])
    )
    total_ind_claims = sum(len(p.get("independent_claims", [])) for p in parsed_claims)

    print(f"  Patents attempted    : {parse_summary['attempted']}")
    print(f"  Patents successful   : {parse_summary['successful']}")
    print(f"  Patents skipped      : {parse_summary['skipped']}")
    print(f"  Patents failed       : {parse_summary['failed']}")
    print(f"  Independent claims  : {total_ind_claims}")
    print(f"  Total elements      : {total_elements}")
    print(f"  Runtime             : {phase3_time}")

    # ── Phase 4: Embedding Similarity ────────────────────────────────────
    _phase("PHASE 4 — Embedding Similarity")
    t0 = time.time()

    sim_results: dict = {"matches": [], "stats": {}}
    phase4_time = "N/A"

    if parsed_claims:
        try:
            from modules.embedding_engine import EmbeddingEngine
            engine = EmbeddingEngine()
            features = strategy.get("features", [])
            sim_results = engine.compute_similarity_matrix(features, parsed_claims)
            phase4_time = _elapsed(t0)
            stats = sim_results.get("stats", {})
            print(f"  Total comparisons : {stats.get('total_comparisons', 0)}")
            print(f"  HIGH matches      : {stats.get('high_matches', 0)}")
            print(f"  MODERATE matches  : {stats.get('moderate_matches', 0)}")
            print(f"  LOW matches       : {stats.get('low_matches', 0)}")
            unmatched = sim_results.get("unmatched_features", [])
            print(f"  Unmatched features: {len(unmatched)}")
            print(f"  Runtime           : {phase4_time}")
        except Exception as exc:
            phase4_time = _elapsed(t0)
            print(f"  [ERROR] Embedding failed: {exc}")
            logger.exception("Embedding phase error")
    else:
        print("  [SKIP] No parsed claims — skipping embedding")

    # ── Phase 5: Contextual Analysis (ElementMapper) ──────────────────────
    _phase("PHASE 5 — Contextual Analysis (ElementMapper)")
    t0 = time.time()

    enriched_matches: list = []
    phase5_time = "N/A"

    if sim_results.get("matches") and gemini_client:
        try:
            from modules.element_mapper import ElementMapper
            mapper = ElementMapper(gemini_client=gemini_client)
            enriched_matches = mapper.analyze_matches(sim_results)
            phase5_time = _elapsed(t0)
            high_conf = sum(1 for m in enriched_matches if m.get("overall_confidence") == "HIGH")
            mod_conf = sum(1 for m in enriched_matches if m.get("overall_confidence") == "MODERATE")
            print(f"  Enriched matches  : {len(enriched_matches)}")
            print(f"  HIGH confidence   : {high_conf}")
            print(f"  MODERATE confidence: {mod_conf}")
            print(f"  Runtime           : {phase5_time}")
        except Exception as exc:
            phase5_time = _elapsed(t0)
            print(f"  [ERROR] Element mapping failed: {exc}")
            logger.exception("ElementMapper phase error")
    else:
        print("  [SKIP] No HIGH/MODERATE matches or no Gemini client")

    # ── Phase 6: White Space Analysis ────────────────────────────────────
    _phase("PHASE 6 — White Space Analysis")
    t0 = time.time()

    white_spaces: list = []
    phase6_time = "N/A"

    try:
        from modules.whitespace_finder import WhiteSpaceFinder
        finder = WhiteSpaceFinder(gemini_client=gemini_client)
        white_spaces = finder.identify_gaps(
            features=strategy.get("features", []),
            similarity_results=sim_results,
            landscape_df_size=len(landscape_df),
            search_strategy=strategy,
            detail_df=detail_df if not detail_df.empty else None,
        )
        phase6_time = _elapsed(t0)
        print(f"  White space findings : {len(white_spaces)}")
        for ws in white_spaces:
            conf = ws.get("confidence", {})
            conf_level = conf.get("level", "?") if isinstance(conf, dict) else str(conf)
            print(f"    [{conf_level}] {ws.get('type', '?')}: {ws.get('title', '')[:60]}")
        print(f"  Runtime              : {phase6_time}")
    except Exception as exc:
        phase6_time = _elapsed(t0)
        print(f"  [ERROR] White space failed: {exc}")
        logger.exception("WhiteSpace phase error")

    # ── Phase 7: PDF Generation ───────────────────────────────────────────
    _phase("PHASE 7 — PDF Report Generation")
    t0 = time.time()

    pdf_success = False
    pdf_size_bytes = 0
    phase7_time = "N/A"

    try:
        from modules.report_generator import ReportGenerator
        session_data = {
            "invention_text": TEST_DESCRIPTION,
            "search_strategy": strategy,
            "detail_patents": detail_df,
            "similarity_results": sim_results,
            "comparison_matrix": enriched_matches,
            "white_spaces": white_spaces,
            "chart_images": {},
        }
        rg = ReportGenerator()
        pdf_bytes = rg.generate(session_data)
        phase7_time = _elapsed(t0)

        if pdf_bytes:
            pdf_success = True
            pdf_size_bytes = len(pdf_bytes)
            out_path = os.path.join(os.path.dirname(__file__), "test_case1_output.pdf")
            with open(out_path, "wb") as f:
                f.write(pdf_bytes)
            print(f"  PDF generated  : YES")
            print(f"  PDF size       : {pdf_size_bytes:,} bytes")
            print(f"  PDF saved to   : {out_path}")
        else:
            print(f"  PDF generated  : NO (empty output)")
        print(f"  Runtime        : {phase7_time}")
    except Exception as exc:
        phase7_time = _elapsed(t0)
        print(f"  [ERROR] PDF generation failed: {exc}")
        logger.exception("PDF generation error")

    # ── Final Summary ─────────────────────────────────────────────────────
    total_time = f"{time.time() - pipeline_start:.2f}s"
    _phase("PIPELINE SUMMARY")
    print(f"  {'Phase':<30} {'Runtime':>10}")
    print(f"  {'-'*42}")
    print(f"  {'Feature Extraction':<30} {phase1_time:>10}")
    print(f"  {'BigQuery Retrieval':<30} {phase2_time:>10}")
    print(f"  {'Claim Parsing':<30} {phase3_time:>10}")
    print(f"  {'Embedding Similarity':<30} {phase4_time:>10}")
    print(f"  {'Contextual Analysis':<30} {phase5_time:>10}")
    print(f"  {'White Space':<30} {phase6_time:>10}")
    print(f"  {'PDF Generation':<30} {phase7_time:>10}")
    print(f"  {'-'*42}")
    print(f"  {'TOTAL':<30} {total_time:>10}")

    print(f"\n  Detail patents      : {len(detail_df)}")
    print(f"  Landscape patents   : {len(landscape_df)}")
    print(f"  Claims parsed       : {total_ind_claims} independent / {total_elements} elements")
    sim_stats = sim_results.get("stats", {})
    print(f"  Similarity HIGH     : {sim_stats.get('high_matches', 0)}")
    print(f"  Similarity MODERATE : {sim_stats.get('moderate_matches', 0)}")
    print(f"  Similarity LOW      : {sim_stats.get('low_matches', 0)}")
    print(f"  White space findings: {len(white_spaces)}")
    print(f"  PDF generated       : {'YES' if pdf_success else 'NO'} ({pdf_size_bytes:,} bytes)")

    # ── Assertions ────────────────────────────────────────────────────────
    errors: list[str] = []
    if detail_df.empty:
        errors.append("FAIL: detail_df is empty")
    if landscape_df.empty:
        errors.append("FAIL: landscape_df is empty")
    if not parsed_claims:
        errors.append("WARN: no claims were parsed (possibly missing claims_text column)")

    if errors:
        print(f"\n  RESULT: FAILED")
        for e in errors:
            print(f"    - {e}")
        sys.exit(1)
    else:
        print(f"\n  RESULT: PASSED ✓")


if __name__ == "__main__":
    main()
