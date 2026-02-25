#!/usr/bin/env python3
"""
Generate examples/solar_charger_session.json by running the full pipeline
on the cached search strategy from tests/_strategy_cache.json.

Usage:
    python scripts/generate_demo_session.py
"""

import json
import os
import sys
import time

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd

_CACHE_PATH  = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "tests", "_strategy_cache.json")
_OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "examples", "solar_charger_session.json")
_SAMPLE_DESC_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                 "examples", "sample_description.txt")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serial(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, (bytes, bytearray)):
        return None
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


# ---------------------------------------------------------------------------
# Bootstrap: env + clients
# ---------------------------------------------------------------------------

from config import settings
from google import genai
from google.cloud import bigquery

api_key       = os.getenv("GEMINI_API_KEY", "")
project       = settings.BIGQUERY_PROJECT
gemini_client = genai.Client(api_key=api_key) if api_key else None

print(f"Project : {project}")
print(f"Gemini  : {'OK' if gemini_client else 'MISSING'}")

# ---------------------------------------------------------------------------
# Load strategy cache
# ---------------------------------------------------------------------------

print("\nLoading strategy cache …")
with open(_CACHE_PATH) as fh:
    strategy = json.load(fh)

# strategy has: features, cpc_codes, search_terms
# Build text_filter if missing
from modules.query_builder import QueryBuilder
if "text_filter" not in strategy:
    qb = QueryBuilder(api_key=api_key)
    clause = qb.build_bigquery_where_clause(strategy)
    strategy.update(clause)

features = strategy.get("features", [])
print(f"  features     : {len(features)}")
print(f"  search_terms : {[t['primary'] for t in strategy.get('search_terms', [])]}")

# Load invention_text
try:
    with open(_SAMPLE_DESC_PATH) as fh:
        invention_text = fh.read()
except FileNotFoundError:
    invention_text = "Solar-powered portable USB charger with MPPT charge controller."

# ---------------------------------------------------------------------------
# Reformulation
# ---------------------------------------------------------------------------

if features and api_key:
    print("\nReformulating features …")
    t0 = time.time()
    qb = QueryBuilder(api_key=api_key)
    qb.reformulate_features_for_patent_language(features)
    has_reform = any("patent_language" in f for f in features)
    print(f"  applied={has_reform}  ({time.time()-t0:.1f}s)")

# ---------------------------------------------------------------------------
# BigQuery retrieval (BQ result cache → fast)
# ---------------------------------------------------------------------------

print("\nBigQuery retrieval …")
t0 = time.time()
bq_client = bigquery.Client(project=project)
from modules.patent_retriever import PatentRetriever
retriever = PatentRetriever(bq_client=bq_client)
detail_df, landscape_df = retriever.search(strategy)
print(f"  detail_df    : {len(detail_df)} rows  ({time.time()-t0:.1f}s)")
print(f"  landscape_df : {len(landscape_df)} rows")

if detail_df.empty:
    sys.exit("ERROR: detail_df empty — cannot build demo session")

# ---------------------------------------------------------------------------
# Claim parsing
# ---------------------------------------------------------------------------

print("\nParsing claims …")
t0 = time.time()
from modules.claim_parser import ClaimParser
parser = ClaimParser(gemini_client=gemini_client)
parse_output  = parser.parse_all(detail_df, max_patents=20)
parsed_claims = parse_output["results"]
n_elements = sum(
    len(cl.get("elements", []))
    for p in parsed_claims
    for cl in p.get("independent_claims", [])
)
print(f"  {len(parsed_claims)} patents, {n_elements} elements  ({time.time()-t0:.1f}s)")

# ---------------------------------------------------------------------------
# Embedding similarity
# ---------------------------------------------------------------------------

print("\nComputing similarity …")
t0 = time.time()
from modules.embedding_engine import EmbeddingEngine
engine = EmbeddingEngine()
sim_results = engine.compute_similarity_matrix(features, parsed_claims)
stats = sim_results.get("stats", {})
print(f"  HIGH={stats.get('high_matches',0)}  "
      f"MODERATE={stats.get('moderate_matches',0)}  "
      f"LOW={stats.get('low_matches',0)}  "
      f"model={sim_results.get('embedding_model','?')}  "
      f"reformulation={sim_results.get('uses_reformulation','?')}  "
      f"({time.time()-t0:.1f}s)")

# ---------------------------------------------------------------------------
# Element mapper (comparison_matrix)
# ---------------------------------------------------------------------------

enriched_matches: list = []
if sim_results.get("matches") and gemini_client:
    print("\nBuilding comparison matrix …")
    t0 = time.time()
    from modules.element_mapper import ElementMapper
    mapper = ElementMapper(gemini_client=gemini_client)
    enriched_matches = mapper.analyze_matches(sim_results)
    print(f"  {len(enriched_matches)} enriched matches  ({time.time()-t0:.1f}s)")
else:
    print("\n[SKIP] Element mapper (no matches or no Gemini)")

# ---------------------------------------------------------------------------
# White space analysis
# ---------------------------------------------------------------------------

white_spaces: list = []
print("\nWhite space analysis …")
t0 = time.time()
try:
    from modules.whitespace_finder import WhiteSpaceFinder
    finder = WhiteSpaceFinder(gemini_client=gemini_client)
    white_spaces = finder.identify_gaps(
        features=features,
        similarity_results=sim_results,
        landscape_df_size=len(landscape_df),
        search_strategy=strategy,
        detail_df=detail_df if not detail_df.empty else None,
    )
    print(f"  {len(white_spaces)} findings  ({time.time()-t0:.1f}s)")
except Exception as exc:
    print(f"  [ERROR] {exc}")

# ---------------------------------------------------------------------------
# Save demo session
# ---------------------------------------------------------------------------

def _strip_numpy(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "matrix"}

snapshot = {
    "invention_text":    invention_text,
    "search_strategy":   strategy,
    "detail_patents":    _serial(detail_df),
    "landscape_patents": _serial(landscape_df),
    "parsed_claims":     parsed_claims,
    "similarity_results": _strip_numpy(sim_results),
    "comparison_matrix": enriched_matches,
    "white_spaces":      white_spaces,
    "query_costs":       [],
    "total_gb_scanned":  0.0,
}

os.makedirs(os.path.dirname(_OUTPUT_PATH), exist_ok=True)
print(f"\nSaving → {_OUTPUT_PATH}")
with open(_OUTPUT_PATH, "w") as fh:
    json.dump(snapshot, fh, indent=2, default=_serial)

size_kb = os.path.getsize(_OUTPUT_PATH) / 1024
print(f"  {size_kb:.0f} KB saved")
print("\n✓ Done")
