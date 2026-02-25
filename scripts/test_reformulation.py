"""
Quick reformulation + embedding test for Solar Charger (uses cached strategy).
Run from repo root: venv/bin/python scripts/test_reformulation.py
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from dotenv import load_dotenv; load_dotenv()

os.environ.setdefault(
    "GOOGLE_APPLICATION_CREDENTIALS",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "credentials", "service-account.json"),
)

from config import settings
from modules.query_builder import QueryBuilder
from modules.embedding_engine import EmbeddingEngine
from modules.claim_parser import ClaimParser
from modules.patent_retriever import PatentRetriever
from google.cloud import bigquery

# Load cached strategy
_CACHE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "_strategy_cache.json")
with open(_CACHE) as f:
    strategy = json.load(f)

api_key = os.getenv("GEMINI_API_KEY", "")
qb = QueryBuilder(api_key=api_key) if api_key else None

if "text_filter" not in strategy and qb:
    clause = qb.build_bigquery_where_clause(strategy)
    strategy.update(clause)

print("\n=== Features BEFORE reformulation ===")
for f in strategy.get("features", []):
    print(f"  - {f['label']}: {f['description'][:70]}")

# Reformulate
if qb:
    t0 = time.time()
    strategy["features"] = qb.reformulate_features_for_patent_language(strategy.get("features", []))
    print(f"\n=== Features AFTER reformulation ({time.time()-t0:.1f}s) ===")
    for f in strategy.get("features", []):
        pl = f.get("patent_language", "")
        changed = "(*)" if pl and pl != f.get("description","") else ""
        print(f"  {changed} {f['label']}: {pl[:80]}")
else:
    print("No GEMINI_API_KEY — skipping reformulation")

# BQ retrieval
print("\n=== BigQuery retrieval ===")
bq_client = bigquery.Client(project=settings.BIGQUERY_PROJECT)
retriever = PatentRetriever(bq_client=bq_client)
detail_df, _ = retriever.search(strategy)
print(f"  detail_df: {len(detail_df)} rows")

# Claim parsing (no Gemini — regex fallback)
print("\n=== Claim parsing ===")
parser = ClaimParser(gemini_client=None)
parse_out = parser.parse_all(detail_df, max_patents=20)
parsed_claims = parse_out["results"]
total_elements = sum(
    len(c.get("elements", []))
    for p in parsed_claims
    for c in p.get("independent_claims", [])
)
print(f"  Parsed: {len(parsed_claims)} patents, {total_elements} elements")

# Embedding with max(original, patent_language)
print("\n=== Similarity (max strategy) ===")
engine = EmbeddingEngine()
sim = engine.compute_similarity_matrix(strategy.get("features", []), parsed_claims)
stats = sim.get("stats", {})
print(f"  Model              : {sim.get('embedding_model','?')}")
print(f"  Uses reformulation : {sim.get('uses_reformulation', False)}")
print(f"  HIGH               : {stats.get('high_matches', 0)}")
print(f"  MODERATE           : {stats.get('moderate_matches', 0)}")
print(f"  LOW                : {stats.get('low_matches', 0)}")

top_high = [m for m in sim.get("matches", []) if m["similarity_level"] == "HIGH"][:5]
if top_high:
    print("\n  Top HIGH matches:")
    for m in top_high:
        print(f"    {m['similarity_score']:.3f} | {m['feature_label']} → {m['element_text'][:70]}")
else:
    print("\n  No HIGH matches found.")
    print("  Top scores overall:")
    all_m = sorted(sim.get("matches", []), key=lambda x: x["similarity_score"], reverse=True)[:5]
    for m in all_m:
        print(f"    {m['similarity_score']:.3f} [{m['similarity_level']}] | {m['feature_label']}")
