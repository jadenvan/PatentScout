#!/usr/bin/env python3
"""
Post-fix diagnostic: Run the full two-phase retrieval and verify results.
"""
import json, os, sys, re, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import settings
from google.cloud import bigquery
from modules.query_builder import QueryBuilder
from modules.patent_retriever import PatentRetriever

DIAG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".tmp", "diagnostics")
os.makedirs(DIAG_DIR, exist_ok=True)

CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "_strategy_cache.json")
SOLAR_KEYWORDS = ["solar", "photovoltaic", "charger", "battery", "usb", "power bank", "portable", "foldable"]

# Load cached strategy
with open(CACHE_PATH) as f:
    strategy = json.load(f)

# Rebuild WHERE clause with updated builder
qb = QueryBuilder(api_key=os.environ.get("GEMINI_API_KEY", "unused"))
where_clause = qb.build_bigquery_where_clause(strategy)
strategy.update(where_clause)

print(f"[DIAG] text_filter:\n{strategy['text_filter']}")
print(f"\n[DIAG] cpc_filter:\n{strategy['cpc_filter']}")
print(f"\n[DIAG] cpc_prefixes: {strategy.get('cpc_prefixes', [])}")

# Clear old cost log for this session
cost_log_path = settings.QUERY_COST_LOG_PATH
if os.path.exists(cost_log_path):
    os.remove(cost_log_path)

# Run the full retrieval
bq_client = bigquery.Client(project=settings.BIGQUERY_PROJECT)
retriever = PatentRetriever(bq_client=bq_client)

print("\n" + "=" * 72)
print("RUNNING FULL TWO-PHASE RETRIEVAL")
print("=" * 72)

t0 = time.time()
detail_df, landscape_df = retriever.search(strategy)
elapsed = time.time() - t0
print(f"\n[DIAG] Total elapsed: {elapsed:.1f}s")
print(f"[DIAG] detail_df:    {len(detail_df)} rows")
print(f"[DIAG] landscape_df: {len(landscape_df)} rows")

# Keyword analysis on detail results
solar_frac = 0
if not detail_df.empty:
    print(f"\n  Top 20 detail titles:")
    for i, (_, row) in enumerate(detail_df.head(20).iterrows()):
        title = str(row.get("title", ""))[:100]
        print(f"    {i+1:2d}. {title}")

    def has_kw(text, kws):
        text = str(text).lower()
        return any(kw in text for kw in kws)

    solar_hits = sum(
        1 for _, row in detail_df.head(20).iterrows()
        if has_kw(row.get("title", ""), SOLAR_KEYWORDS)
    )
    top20 = min(20, len(detail_df))
    solar_frac = solar_hits / top20 if top20 > 0 else 0
    print(f"\n  Solar keyword fraction (top 20 titles): {solar_hits}/{top20} = {solar_frac:.0%}")
    print(f"  Acceptance criterion: >=40%  ->  {'PASS' if solar_frac >= 0.40 else 'FAIL'}")

    detail_df.head(50).to_csv(os.path.join(DIAG_DIR, "detail_preview.csv"), index=False)
    print(f"  detail_preview.csv saved")

# Load cost log for this session
total_gb = 0.0
if os.path.exists(cost_log_path):
    with open(cost_log_path) as f:
        costs = json.load(f)
    for c in costs:
        total_gb += c.get("gb", 0)
    print(f"\n  Session GB scanned: {total_gb:.2f} GB")
    print(f"  Per-query breakdown:")
    for c in costs:
        print(f"    {c['query_name']:30s} {c['gb']:.2f} GB  {c['elapsed_s']:.1f}s")

# Save results summary
summary = {
    "detail_rows": len(detail_df),
    "landscape_rows": len(landscape_df),
    "elapsed_s": round(elapsed, 1),
    "total_gb_scanned": round(total_gb, 2),
    "solar_keyword_fraction_top20": round(solar_frac, 3) if not detail_df.empty else 0,
    "top_20_titles": [str(r.get("title", ""))[:120] for _, r in detail_df.head(20).iterrows()] if not detail_df.empty else [],
    "pass_40pct": solar_frac >= 0.40 if not detail_df.empty else False,
}
with open(os.path.join(DIAG_DIR, "retrieval_results.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== DIAGNOSTIC COMPLETE ===")
