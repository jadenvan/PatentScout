#!/usr/bin/env python3
"""Quick evaluation of cached solar data - baseline metrics."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from tests.evaluation_harness import TEST_CASE_1, evaluate_retrieval_quality, print_experiment_comparison

# Load cached solar data
detail_df = pd.read_json("experiments/cache/solar_detail.json")
landscape_df = pd.read_json("experiments/cache/solar_landscape.json")

print("=== Solar Charger Cached Data Stats ===")
print(f"Detail patents: {len(detail_df)}")
print(f"Landscape patents: {len(landscape_df)}")
print(f"Has abstracts: {detail_df['abstract'].notna().sum()} / {len(detail_df)}")
print(f"Has CPC: {detail_df['cpc_code'].apply(lambda x: isinstance(x, list) and len(x) > 0).sum()} / {len(detail_df)}")

# Evaluate baseline
metrics_a = evaluate_retrieval_quality(detail_df, TEST_CASE_1)
print(f"\n=== Baseline Metrics (Strategy A) ===")
for k, v in metrics_a.items():
    print(f"  {k}: {v}")

# Re-rank with embeddings (Strategy G)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")
df_g = detail_df[detail_df["abstract"].notna() & (detail_df["abstract"] != "")].copy()
print(f"\nPatents with abstracts for re-ranking: {len(df_g)}")

if not df_g.empty:
    user_emb = model.encode([TEST_CASE_1["description"]])
    abs_embs = model.encode(df_g["abstract"].tolist())
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    scores = cos_sim(user_emb, abs_embs)[0]
    df_g["semantic_relevance"] = scores
    df_g = df_g.sort_values("semantic_relevance", ascending=False).head(100)

    metrics_g = evaluate_retrieval_quality(df_g, TEST_CASE_1)
    print(f"\n=== Re-Ranked Metrics (Strategy G) ===")
    for k, v in metrics_g.items():
        print(f"  {k}: {v}")

    print("\nTop 5 after re-ranking:")
    for _, row in df_g.head(5).iterrows():
        title = str(row.get("title", ""))[:80]
        print(f"  {row['publication_number']}: {title} (sim={row['semantic_relevance']:.3f})")

    print_experiment_comparison([
        {"name": "A: Baseline", "retrieval_metrics": metrics_a, "bytes_scanned": 0, "runtime_seconds": 0},
        {"name": "G: Re-Rank", "retrieval_metrics": metrics_g, "bytes_scanned": 0, "runtime_seconds": 0},
    ])
else:
    print("No abstracts available for re-ranking")
