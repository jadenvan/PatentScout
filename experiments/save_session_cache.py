"""
save_session_cache.py
---------------------
Run ONCE to export the canonical demo session (examples/solar_charger_session.json)
into individual, purpose-named cache files under experiments/cache/.

These files are then used by CachedBigQueryClient and the experiment runner
so that Strategy A / Strategy G baselines never need a live BigQuery hit.

Usage:
    python experiments/save_session_cache.py
"""

import json
import os
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSION_FILE = os.path.join(ROOT, "examples", "solar_charger_session.json")
CACHE_DIR = os.path.join(ROOT, "experiments", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def main() -> None:
    # ------------------------------------------------------------------
    # Load source snapshot
    # ------------------------------------------------------------------
    if not os.path.exists(SESSION_FILE):
        print(
            f"ERROR: session snapshot not found at {SESSION_FILE}\n"
            "Run the full pipeline via the Streamlit app first, then "
            "re-run this script."
        )
        sys.exit(1)

    with open(SESSION_FILE) as fh:
        snap = json.load(fh)

    print(f"Loaded session snapshot from {SESSION_FILE}")

    # ------------------------------------------------------------------
    # 1. Detail patents (the ~100 with claims)
    # ------------------------------------------------------------------
    raw_detail = snap.get("detail_patents")
    if raw_detail:
        detail_df = pd.DataFrame(raw_detail)
        out = os.path.join(CACHE_DIR, "solar_detail.json")
        detail_df.to_json(out, orient="records", indent=2)
        print(f"  Saved detail_patents  → {out}  ({len(detail_df)} rows)")
    else:
        print("  WARN: detail_patents not found in snapshot — skipping")

    # ------------------------------------------------------------------
    # 2. Landscape patents (the ~500 for charts)
    # ------------------------------------------------------------------
    raw_landscape = snap.get("landscape_patents")
    if raw_landscape:
        landscape_df = pd.DataFrame(raw_landscape)
        out = os.path.join(CACHE_DIR, "solar_landscape.json")
        landscape_df.to_json(out, orient="records", indent=2)
        print(
            f"  Saved landscape_patents → {out}  ({len(landscape_df)} rows)"
        )
    else:
        print("  WARN: landscape_patents not found in snapshot — skipping")

    # ------------------------------------------------------------------
    # 3. Search strategy (features, CPC codes, search terms)
    # ------------------------------------------------------------------
    strategy = snap.get("search_strategy")
    if strategy:
        out = os.path.join(CACHE_DIR, "solar_strategy.json")
        with open(out, "w") as fh:
            json.dump(strategy, fh, indent=2)
        print(f"  Saved search_strategy  → {out}")
    else:
        print("  WARN: search_strategy not found in snapshot — skipping")

    # ------------------------------------------------------------------
    # 4. Parsed claims
    # ------------------------------------------------------------------
    claims = snap.get("parsed_claims")
    if claims:
        out = os.path.join(CACHE_DIR, "solar_claims.json")
        with open(out, "w") as fh:
            json.dump(claims, fh, indent=2)
        print(f"  Saved parsed_claims    → {out}")
    else:
        print("  WARN: parsed_claims not found in snapshot — skipping")

    # ------------------------------------------------------------------
    # 5. Similarity results
    # ------------------------------------------------------------------
    similarity = snap.get("similarity_results")
    if similarity:
        out = os.path.join(CACHE_DIR, "solar_similarity.json")
        with open(out, "w") as fh:
            json.dump(similarity, fh, indent=2, default=str)
        print(f"  Saved similarity_results → {out}")
    else:
        print("  WARN: similarity_results not found in snapshot — skipping")

    # ------------------------------------------------------------------
    # 6. Comparison matrix (element-level scores)
    # ------------------------------------------------------------------
    matrix = snap.get("comparison_matrix")
    if matrix:
        out = os.path.join(CACHE_DIR, "solar_comparison_matrix.json")
        with open(out, "w") as fh:
            json.dump(matrix, fh, indent=2, default=str)
        print(f"  Saved comparison_matrix → {out}")
    else:
        print("  WARN: comparison_matrix not found in snapshot — skipping")

    # ------------------------------------------------------------------
    # 7. White-space opportunities
    # ------------------------------------------------------------------
    white_spaces = snap.get("white_spaces")
    if white_spaces:
        out = os.path.join(CACHE_DIR, "solar_white_spaces.json")
        with open(out, "w") as fh:
            json.dump(white_spaces, fh, indent=2, default=str)
        print(f"  Saved white_spaces     → {out}")
    else:
        print("  WARN: white_spaces not found in snapshot — skipping")

    print(
        "\nAll available data cached under experiments/cache/.\n"
        "No need to re-query BigQuery for Strategy A baseline or "
        "Strategy G re-ranking."
    )


if __name__ == "__main__":
    main()
