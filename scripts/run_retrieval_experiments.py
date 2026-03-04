#!/usr/bin/env python3
"""
PatentScout — Retrieval Experiment Runner

Implements and evaluates 8 retrieval strategies (A–H) across 3 test cases
to identify the best approach for retrieval precision.

Cost-optimized: uses CachedBigQueryClient, runs cheapest strategies first,
and only runs top-3 strategies on test cases 2 and 3.

Usage::

    python scripts/run_retrieval_experiments.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import pandas as pd

# Ensure project root is on path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from experiments.query_cache import CachedBigQueryClient
from tests.evaluation_harness import (
    TEST_CASES, TEST_CASE_1, TEST_CASE_2, TEST_CASE_3,
    evaluate_retrieval_quality, print_experiment_comparison,
)

# STRATEGY IMPLEMENTATIONS

def run_strategy_a(test_case: dict, bq_client: CachedBigQueryClient) -> tuple[pd.DataFrame, float, float]:
    """Strategy A: Current Baseline — load from cache for Solar, else run current retriever."""
    t0 = time.time()
    if test_case["name"] == "Solar Charger":
        cache_path = os.path.join(_ROOT, "experiments", "cache", "solar_detail.json")
        if os.path.exists(cache_path):
            df = pd.read_json(cache_path)
            return df, 0.0, time.time() - t0

    # For non-cached test cases, run current retriever
    df, gb = _run_current_retriever(test_case, bq_client)
    return df, gb, time.time() - t0


def run_strategy_b(test_case: dict, bq_client: CachedBigQueryClient) -> tuple[pd.DataFrame, float, float]:
    """Strategy B: Title-First Search — scans ~1-3 GB."""
    t0 = time.time()

    title_regexes = {
        "Solar Charger": (
            r'(?i)(solar.*charg|solar.*panel.*portab|'
            r'photovoltaic.*charg|solar.*fold|'
            r'solar.*power.*portab)'
        ),
        "Smart Doorbell": (
            r'(?i)(smart.*door.*bell|door.*bell.*camera|'
            r'video.*door|door.*entry.*recogni|'
            r'door.*facial|doorbell.*alert)'
        ),
        "Bike Helmet": (
            r'(?i)(helmet.*speaker|helmet.*audio|'
            r'bicycle.*helmet.*signal|helmet.*sensor|'
            r'cycling.*helmet.*led|bone.*conduction.*helmet)'
        ),
    }

    title_regex = title_regexes.get(test_case["name"], "")
    if not title_regex:
        return pd.DataFrame(), 0.0, time.time() - t0

    sql = f"""
    SELECT DISTINCT
        publication_number,
        title.text AS title,
        abstract.text AS abstract,
        filing_date, grant_date, publication_date
    FROM
        `patents-public-data.patents.publications`,
        UNNEST(title_localized) AS title,
        UNNEST(abstract_localized) AS abstract
    WHERE
        country_code = 'US'
        AND filing_date > 20050101
        AND title.language = 'en'
        AND abstract.language = 'en'
        AND grant_date > 0
        AND REGEXP_CONTAINS(title.text, r'{title_regex}')
    LIMIT 100
    """

    df = bq_client.query(sql, max_gb=5.0, description=f"Strategy B: {test_case['name']}")
    gb = bq_client.total_bytes_used  # approximate
    return df, gb, time.time() - t0


def run_strategy_c(test_case: dict, bq_client: CachedBigQueryClient) -> tuple[pd.DataFrame, float, float]:
    """Strategy C: Abstract Keyword Search — scans ~5-10 GB."""
    t0 = time.time()

    abstract_regexes = {
        "Solar Charger": (
            r'(?i)(solar\s*(panel|cell|charg|power)|'
            r'photovoltaic\s*(charg|panel|portab)|'
            r'fold.{0,20}solar|USB.{0,10}solar|'
            r'portable\s*solar)'
        ),
        "Smart Doorbell": (
            r'(?i)(doorbell.{0,20}camera|'
            r'facial\s*recogni.{0,20}door|'
            r'video\s*door|smart\s*door.{0,10}bell|'
            r'door.{0,20}unlock.{0,20}face)'
        ),
        "Bike Helmet": (
            r'(?i)(helmet.{0,20}(speaker|audio|bone\s*conduct)|'
            r'bicycle.*helmet.{0,20}(LED|signal|sensor)|'
            r'cycling.*helmet.{0,20}(gesture|acceleromet)|'
            r'air\s*quality.{0,20}helmet)'
        ),
    }

    abstract_regex = abstract_regexes.get(test_case["name"], "")
    if not abstract_regex:
        return pd.DataFrame(), 0.0, time.time() - t0

    sql = f"""
    SELECT DISTINCT
        publication_number,
        title.text AS title,
        abstract.text AS abstract,
        filing_date, grant_date, publication_date
    FROM
        `patents-public-data.patents.publications`,
        UNNEST(title_localized) AS title,
        UNNEST(abstract_localized) AS abstract
    WHERE
        country_code = 'US'
        AND filing_date > 20050101
        AND title.language = 'en'
        AND abstract.language = 'en'
        AND grant_date > 0
        AND REGEXP_CONTAINS(abstract.text, r'{abstract_regex}')
    LIMIT 200
    """

    df = bq_client.query(sql, max_gb=12.0, description=f"Strategy C: {test_case['name']}")
    return df, bq_client.total_bytes_used, time.time() - t0


def run_strategy_d(test_case: dict, bq_client: CachedBigQueryClient) -> tuple[pd.DataFrame, float, float]:
    """Strategy D: Strict CPC-First — scans ~2-4 GB for stage 1."""
    t0 = time.time()

    cpc_filters = {
        "Solar Charger": (
            "cpc.code LIKE 'H02S%' "
            "OR cpc.code LIKE 'H02J7%' "
            "OR cpc.code LIKE 'H01L31%'"
        ),
        "Smart Doorbell": (
            "cpc.code LIKE 'H04N7%' "
            "OR cpc.code LIKE 'G06V40%' "
            "OR cpc.code LIKE 'E05B47%' "
            "OR cpc.code LIKE 'G08B13%'"
        ),
        "Bike Helmet": (
            "cpc.code LIKE 'A42B3%' "
            "OR cpc.code LIKE 'H04R1%' "
            "OR cpc.code LIKE 'B62J6%' "
            "OR cpc.code LIKE 'G01N33%'"
        ),
    }

    cpc_filter = cpc_filters.get(test_case["name"], "")
    if not cpc_filter:
        return pd.DataFrame(), 0.0, time.time() - t0

    # Stage 1: Get patent numbers by CPC
    sql_stage1 = f"""
    SELECT DISTINCT pub.publication_number
    FROM
        `patents-public-data.patents.publications` AS pub,
        UNNEST(pub.cpc) AS cpc
    WHERE
        pub.country_code = 'US'
        AND pub.filing_date > 20050101
        AND pub.grant_date > 0
        AND ({cpc_filter})
    LIMIT 500
    """

    numbers_df = bq_client.query(
        sql_stage1, max_gb=5.0,
        description=f"Strategy D stage1: {test_case['name']}",
    )

    if numbers_df.empty:
        return pd.DataFrame(), bq_client.total_bytes_used, time.time() - t0

    # Stage 2: Get text for those specific patents
    patent_list = ", ".join(
        [f"'{p}'" for p in numbers_df["publication_number"].tolist()[:200]]
    )

    sql_stage2 = f"""
    SELECT DISTINCT
        publication_number,
        title.text AS title,
        abstract.text AS abstract,
        filing_date, grant_date, publication_date
    FROM
        `patents-public-data.patents.publications`,
        UNNEST(title_localized) AS title,
        UNNEST(abstract_localized) AS abstract
    WHERE
        publication_number IN ({patent_list})
        AND title.language = 'en'
        AND abstract.language = 'en'
    LIMIT 200
    """

    df = bq_client.query(
        sql_stage2, max_gb=10.0,
        description=f"Strategy D stage2: {test_case['name']}",
    )

    return df, bq_client.total_bytes_used, time.time() - t0


def run_strategy_e(test_case: dict, bq_client: CachedBigQueryClient) -> tuple[pd.DataFrame, float, float]:
    """Strategy E: CPC + Title combined — uses both filters."""
    t0 = time.time()

    params = {
        "Solar Charger": {
            "cpc": "cpc.code LIKE 'H02S%' OR cpc.code LIKE 'H02J7%' OR cpc.code LIKE 'H01L31%'",
            "title_regex": r"(?i)(solar|photovoltaic|charger|battery.*portab)",
        },
        "Smart Doorbell": {
            "cpc": "cpc.code LIKE 'H04N7%' OR cpc.code LIKE 'G06V40%' OR cpc.code LIKE 'E05B47%'",
            "title_regex": r"(?i)(doorbell|door.*camera|facial.*door|video.*door|smart.*door)",
        },
        "Bike Helmet": {
            "cpc": "cpc.code LIKE 'A42B3%' OR cpc.code LIKE 'H04R1%' OR cpc.code LIKE 'B62J6%'",
            "title_regex": r"(?i)(helmet|bicycle.*audio|cycling.*sensor|bone.*conduct)",
        },
    }

    p = params.get(test_case["name"])
    if not p:
        return pd.DataFrame(), 0.0, time.time() - t0

    sql = f"""
    SELECT DISTINCT
        pub.publication_number,
        title.text AS title,
        abstract.text AS abstract,
        pub.filing_date, pub.grant_date, pub.publication_date
    FROM
        `patents-public-data.patents.publications` AS pub,
        UNNEST(pub.cpc) AS cpc,
        UNNEST(pub.title_localized) AS title,
        UNNEST(pub.abstract_localized) AS abstract
    WHERE
        pub.country_code = 'US'
        AND pub.filing_date > 20050101
        AND pub.grant_date > 0
        AND title.language = 'en'
        AND abstract.language = 'en'
        AND ({p['cpc']})
        AND REGEXP_CONTAINS(title.text, r'{p["title_regex"]}')
    LIMIT 200
    """

    df = bq_client.query(sql, max_gb=10.0, description=f"Strategy E: {test_case['name']}")
    return df, bq_client.total_bytes_used, time.time() - t0


def run_strategy_f(test_case: dict, bq_client: CachedBigQueryClient) -> tuple[pd.DataFrame, float, float]:
    """Strategy F: CPC + Abstract keyword combined."""
    t0 = time.time()

    params = {
        "Solar Charger": {
            "cpc": "cpc.code LIKE 'H02S%' OR cpc.code LIKE 'H02J7%'",
            "abstract_regex": r"(?i)(solar.*charg|photovoltaic.*portab|fold.*solar|solar.*panel.*USB)",
        },
        "Smart Doorbell": {
            "cpc": "cpc.code LIKE 'H04N7%' OR cpc.code LIKE 'E05B47%'",
            "abstract_regex": r"(?i)(doorbell.*camera|facial.*recogni.*door|video.*door.*bell|smart.*door.*lock)",
        },
        "Bike Helmet": {
            "cpc": "cpc.code LIKE 'A42B3%' OR cpc.code LIKE 'B62J6%'",
            "abstract_regex": r"(?i)(helmet.*speaker|helmet.*audio|bicycle.*helmet.*signal|helmet.*air.*quality)",
        },
    }

    p = params.get(test_case["name"])
    if not p:
        return pd.DataFrame(), 0.0, time.time() - t0

    sql = f"""
    SELECT DISTINCT
        pub.publication_number,
        title.text AS title,
        abstract.text AS abstract,
        pub.filing_date, pub.grant_date, pub.publication_date
    FROM
        `patents-public-data.patents.publications` AS pub,
        UNNEST(pub.cpc) AS cpc,
        UNNEST(pub.title_localized) AS title,
        UNNEST(pub.abstract_localized) AS abstract
    WHERE
        pub.country_code = 'US'
        AND pub.filing_date > 20050101
        AND pub.grant_date > 0
        AND title.language = 'en'
        AND abstract.language = 'en'
        AND ({p['cpc']})
        AND REGEXP_CONTAINS(abstract.text, r'{p["abstract_regex"]}')
    LIMIT 200
    """

    df = bq_client.query(sql, max_gb=12.0, description=f"Strategy F: {test_case['name']}")
    return df, bq_client.total_bytes_used, time.time() - t0


def run_strategy_g(test_case: dict, baseline_df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """Strategy G: Embedding Re-Rank — 0 GB, re-ranks baseline data."""
    t0 = time.time()

    if baseline_df is None or baseline_df.empty:
        return pd.DataFrame(), 0.0, time.time() - t0

    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Filter to patents with valid abstracts
    df = baseline_df[
        baseline_df["abstract"].notna() & (baseline_df["abstract"] != "")
    ].copy()

    if df.empty:
        # Fall back to title-based re-ranking
        df = baseline_df[
            baseline_df["title"].notna() & (baseline_df["title"] != "")
        ].copy()
        if df.empty:
            return baseline_df, 0.0, time.time() - t0
        text_col = "title"
    else:
        text_col = "abstract"

    # Encode user description
    user_emb = model.encode([test_case["description"]])

    # Encode all texts
    texts = df[text_col].tolist()
    text_embs = model.encode(texts, show_progress_bar=True)

    # Compute similarity
    scores = cosine_similarity(user_emb, text_embs)[0]

    df["semantic_relevance"] = scores
    df = df.sort_values("semantic_relevance", ascending=False)

    return df.head(100), 0.0, time.time() - t0


def run_strategy_h(
    test_case: dict,
    strategy_results: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, float, float]:
    """Strategy H: Combined — merge top 2 strategies' results and deduplicate."""
    t0 = time.time()

    # Combine all dataframes
    frames = [df for df in strategy_results.values() if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame(), 0.0, time.time() - t0

    combined = pd.concat(frames, ignore_index=True)

    # Deduplicate by publication_number
    if "publication_number" in combined.columns:
        combined = combined.drop_duplicates(subset=["publication_number"])

    # If we have semantic_relevance, sort by it
    if "semantic_relevance" in combined.columns:
        combined = combined.sort_values("semantic_relevance", ascending=False)

    return combined.head(200), 0.0, time.time() - t0


def _run_current_retriever(test_case: dict, bq_client: CachedBigQueryClient) -> tuple[pd.DataFrame, float]:
    """Run current PatentRetriever for non-cached test cases."""
    from modules.query_builder import QueryBuilder
    from modules.patent_retriever import PatentRetriever

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("  WARNING: GEMINI_API_KEY not set, cannot run current retriever")
        return pd.DataFrame(), 0.0

    qb = QueryBuilder(api_key=api_key)
    strategy = qb.extract_features(test_case["description"])
    where = qb.build_bigquery_where_clause(strategy)
    strategy.update(where)

    retriever = PatentRetriever(bq_client=bq_client.client)
    detail_df, landscape_df = retriever.search(strategy)

    combined = pd.concat([detail_df, landscape_df], ignore_index=True)
    return combined, bq_client.total_bytes_used


def _add_cpc_data(df: pd.DataFrame, bq_client: CachedBigQueryClient) -> pd.DataFrame:
    """Fetch CPC codes for a DataFrame that may not have them."""
    if df.empty or "cpc_code" in df.columns:
        # Check if CPC data is populated
        if "cpc_code" in df.columns:
            has_cpc = df["cpc_code"].apply(
                lambda x: isinstance(x, list) and len(x) > 0
            ).sum()
            if has_cpc > 0:
                return df

    if "publication_number" not in df.columns:
        return df

    # Fetch CPC codes for these patents
    pub_nums = df["publication_number"].tolist()[:200]
    if not pub_nums:
        return df

    patent_list = ", ".join([f"'{p}'" for p in pub_nums])
    sql = f"""
    SELECT
        publication_number,
        (SELECT STRING_AGG(c.code, ' | ') FROM UNNEST(cpc) c) AS cpc_codes_agg
    FROM `patents-public-data.patents.publications`
    WHERE publication_number IN ({patent_list})
    """

    try:
        cpc_df = bq_client.query(
            sql, max_gb=5.0,
            description=f"CPC fetch for {len(pub_nums)} patents",
        )
        if not cpc_df.empty:
            cpc_df["cpc_code"] = cpc_df["cpc_codes_agg"].apply(
                lambda v: [x.strip() for x in v.split(" | ")] if isinstance(v, str) and v else []
            )
            cpc_df = cpc_df.drop(columns=["cpc_codes_agg"])
            df = df.merge(cpc_df, on="publication_number", how="left", suffixes=("_old", ""))
            if "cpc_code_old" in df.columns:
                df.drop(columns=["cpc_code_old"], inplace=True)
            if "cpc_code" not in df.columns:
                df["cpc_code"] = [[] for _ in range(len(df))]
    except Exception as exc:
        print(f"  CPC fetch failed: {exc}")
        if "cpc_code" not in df.columns:
            df["cpc_code"] = [[] for _ in range(len(df))]

    return df


# MAIN EXPERIMENT RUNNER

def run_all_experiments() -> dict:
    """Run all retrieval experiments and return results."""
    bq_client = CachedBigQueryClient()
    all_results: dict = {}  # test_case_name -> list of experiment dicts

    print("\n" + "=" * 72)
    print("PHASE 1: Zero-cost experiments on Solar Charger")
    print("=" * 72)

    tc1 = TEST_CASE_1
    tc1_experiments: list[dict] = []
    tc1_dfs: dict[str, pd.DataFrame] = {}

    # Strategy A: Baseline (from cache)
    print("\n--- Strategy A: Current Baseline ---")
    bytes_before = bq_client.total_bytes_used
    df_a, gb_a, time_a = run_strategy_a(tc1, bq_client)
    gb_a = bq_client.total_bytes_used - bytes_before
    print(f"  Retrieved {len(df_a)} patents in {time_a:.1f}s ({gb_a:.1f} GB)")
    tc1_dfs["A"] = df_a

    metrics_a = evaluate_retrieval_quality(df_a, tc1)
    tc1_experiments.append({
        "name": "A: Baseline",
        "retrieval_metrics": metrics_a,
        "bytes_scanned": gb_a,
        "runtime_seconds": time_a,
    })

    # Strategy G: Embedding Re-Rank (0 GB)
    print("\n--- Strategy G: Embedding Re-Rank ---")
    df_g, gb_g, time_g = run_strategy_g(tc1, df_a)
    print(f"  Re-ranked to {len(df_g)} patents in {time_g:.1f}s ({gb_g:.1f} GB)")
    tc1_dfs["G"] = df_g

    metrics_g = evaluate_retrieval_quality(df_g, tc1)
    tc1_experiments.append({
        "name": "G: Embedding Re-Rank",
        "retrieval_metrics": metrics_g,
        "bytes_scanned": gb_g,
        "runtime_seconds": time_g,
    })

    print("\n" + "=" * 72)
    print("PHASE 2: Cheap experiments on Solar Charger")
    print("=" * 72)

    # Strategy B: Title-First
    print("\n--- Strategy B: Title-First Search ---")
    bytes_before = bq_client.total_bytes_used
    df_b, _, time_b = run_strategy_b(tc1, bq_client)
    gb_b = bq_client.total_bytes_used - bytes_before
    df_b = _add_cpc_data(df_b, bq_client)
    print(f"  Retrieved {len(df_b)} patents in {time_b:.1f}s ({gb_b:.1f} GB)")
    tc1_dfs["B"] = df_b

    metrics_b = evaluate_retrieval_quality(df_b, tc1)
    tc1_experiments.append({
        "name": "B: Title-First",
        "retrieval_metrics": metrics_b,
        "bytes_scanned": gb_b,
        "runtime_seconds": time_b,
    })

    # Strategy D: Strict CPC-First
    print("\n--- Strategy D: Strict CPC-First ---")
    bytes_before = bq_client.total_bytes_used
    df_d, _, time_d = run_strategy_d(tc1, bq_client)
    gb_d = bq_client.total_bytes_used - bytes_before
    print(f"  Retrieved {len(df_d)} patents in {time_d:.1f}s ({gb_d:.1f} GB)")
    tc1_dfs["D"] = df_d

    metrics_d = evaluate_retrieval_quality(df_d, tc1)
    tc1_experiments.append({
        "name": "D: CPC-First",
        "retrieval_metrics": metrics_d,
        "bytes_scanned": gb_d,
        "runtime_seconds": time_d,
    })

    print("\n" + "=" * 72)
    print("PHASE 3: Moderate experiments on Solar Charger")
    print("=" * 72)

    # Check budget
    if bq_client.total_bytes_used > 150:
        print("  ⚠️ BUDGET EXCEEDED 150 GB — stopping new BQ queries")
    else:
        # Strategy C: Abstract Keyword
        print("\n--- Strategy C: Abstract Keyword ---")
        bytes_before = bq_client.total_bytes_used
        df_c, _, time_c = run_strategy_c(tc1, bq_client)
        gb_c = bq_client.total_bytes_used - bytes_before
        print(f"  Retrieved {len(df_c)} patents in {time_c:.1f}s ({gb_c:.1f} GB)")
        tc1_dfs["C"] = df_c

        metrics_c = evaluate_retrieval_quality(df_c, tc1)
        tc1_experiments.append({
            "name": "C: Abstract Keyword",
            "retrieval_metrics": metrics_c,
            "bytes_scanned": gb_c,
            "runtime_seconds": time_c,
        })

        # Strategy E: CPC + Title
        print("\n--- Strategy E: CPC + Title ---")
        bytes_before = bq_client.total_bytes_used
        df_e, _, time_e = run_strategy_e(tc1, bq_client)
        gb_e = bq_client.total_bytes_used - bytes_before
        print(f"  Retrieved {len(df_e)} patents in {time_e:.1f}s ({gb_e:.1f} GB)")
        tc1_dfs["E"] = df_e

        metrics_e = evaluate_retrieval_quality(df_e, tc1)
        tc1_experiments.append({
            "name": "E: CPC + Title",
            "retrieval_metrics": metrics_e,
            "bytes_scanned": gb_e,
            "runtime_seconds": time_e,
        })

        # Strategy F: CPC + Abstract
        print("\n--- Strategy F: CPC + Abstract ---")
        bytes_before = bq_client.total_bytes_used
        df_f, _, time_f = run_strategy_f(tc1, bq_client)
        gb_f = bq_client.total_bytes_used - bytes_before
        print(f"  Retrieved {len(df_f)} patents in {time_f:.1f}s ({gb_f:.1f} GB)")
        tc1_dfs["F"] = df_f

        metrics_f = evaluate_retrieval_quality(df_f, tc1)
        tc1_experiments.append({
            "name": "F: CPC + Abstract",
            "retrieval_metrics": metrics_f,
            "bytes_scanned": gb_f,
            "runtime_seconds": time_f,
        })

    print("\n" + "=" * 72)
    print("PHASE 4: Test Case 1 (Solar Charger) Results")
    print("=" * 72)

    print_experiment_comparison(tc1_experiments)
    all_results["Solar Charger"] = tc1_experiments

    # Pick top 3 strategies
    sorted_exps = sorted(
        tc1_experiments,
        key=lambda x: x["retrieval_metrics"]["composite_score"],
        reverse=True,
    )
    top3_names = [e["name"] for e in sorted_exps[:3]]
    print(f"\nTop 3 strategies: {top3_names}")

    # Map strategy names back to runner functions
    strategy_runners = {
        "A: Baseline": run_strategy_a,
        "B: Title-First": run_strategy_b,
        "C: Abstract Keyword": run_strategy_c,
        "D: CPC-First": run_strategy_d,
        "E: CPC + Title": run_strategy_e,
        "F: CPC + Abstract": run_strategy_f,
    }

    for tc in [TEST_CASE_2, TEST_CASE_3]:
        print(f"\n{'=' * 72}")
        print(f"PHASE 5: Top 3 strategies on {tc['name']}")
        print("=" * 72)

        tc_exps: list[dict] = []
        tc_dfs: dict[str, pd.DataFrame] = {}

        for strategy_name in top3_names:
            # Skip G (re-rank) and H (combined) — handled separately
            if "Re-Rank" in strategy_name or "Combined" in strategy_name:
                continue

            runner = strategy_runners.get(strategy_name)
            if runner is None:
                continue

            print(f"\n--- {strategy_name} ---")
            bytes_before = bq_client.total_bytes_used

            if bq_client.total_bytes_used > 150:
                print("  ⚠️ BUDGET WARNING: > 150 GB used")

            try:
                df, _, t = runner(tc, bq_client)
                gb = bq_client.total_bytes_used - bytes_before
                df = _add_cpc_data(df, bq_client)
                print(f"  Retrieved {len(df)} patents in {t:.1f}s ({gb:.1f} GB)")
                tc_dfs[strategy_name] = df

                metrics = evaluate_retrieval_quality(df, tc)
                tc_exps.append({
                    "name": strategy_name,
                    "retrieval_metrics": metrics,
                    "bytes_scanned": gb,
                    "runtime_seconds": t,
                })
            except Exception as exc:
                print(f"  FAILED: {exc}")

        # Run Strategy G (re-rank) on the best baseline for this TC
        if tc_dfs:
            best_baseline_name = max(tc_dfs.keys(), key=lambda k: len(tc_dfs[k]))
            print(f"\n--- G: Embedding Re-Rank (on {best_baseline_name} data) ---")
            df_g_tc, gb_g_tc, time_g_tc = run_strategy_g(tc, tc_dfs[best_baseline_name])
            print(f"  Re-ranked to {len(df_g_tc)} patents in {time_g_tc:.1f}s")
            tc_dfs["G"] = df_g_tc

            metrics_g_tc = evaluate_retrieval_quality(df_g_tc, tc)
            tc_exps.append({
                "name": "G: Re-Rank",
                "retrieval_metrics": metrics_g_tc,
                "bytes_scanned": 0.0,
                "runtime_seconds": time_g_tc,
            })

        # Strategy H: Combined
        if len(tc_dfs) >= 2:
            print(f"\n--- H: Combined (top 2) ---")
            # Pick top 2 by composite score
            top2_exps = sorted(tc_exps, key=lambda x: x["retrieval_metrics"]["composite_score"], reverse=True)[:2]
            top2_dfs = {}
            for e in top2_exps:
                name = e["name"]
                if name in tc_dfs:
                    top2_dfs[name] = tc_dfs[name]
            df_h, gb_h, time_h = run_strategy_h(tc, top2_dfs)
            df_h = _add_cpc_data(df_h, bq_client)
            print(f"  Combined {len(df_h)} patents in {time_h:.1f}s")

            metrics_h = evaluate_retrieval_quality(df_h, tc)
            tc_exps.append({
                "name": "H: Combined",
                "retrieval_metrics": metrics_h,
                "bytes_scanned": 0.0,
                "runtime_seconds": time_h,
            })

        print_experiment_comparison(tc_exps)
        all_results[tc["name"]] = tc_exps

    print(f"\n{'=' * 72}")
    print("PHASE 6: Strategy H (Combined) on Solar Charger")
    print("=" * 72)

    if len(tc1_dfs) >= 2:
        top2_tc1 = sorted(tc1_experiments, key=lambda x: x["retrieval_metrics"]["composite_score"], reverse=True)[:2]
        top2_tc1_dfs = {}
        for e in top2_tc1:
            name = e["name"].split(":")[0].strip()
            if name in tc1_dfs:
                top2_tc1_dfs[name] = tc1_dfs[name]
        if top2_tc1_dfs:
            df_h1, gb_h1, time_h1 = run_strategy_h(tc1, top2_tc1_dfs)
            metrics_h1 = evaluate_retrieval_quality(df_h1, tc1)
            tc1_experiments.append({
                "name": "H: Combined",
                "retrieval_metrics": metrics_h1,
                "bytes_scanned": 0.0,
                "runtime_seconds": time_h1,
            })
            print_experiment_comparison(tc1_experiments)
            all_results["Solar Charger"] = tc1_experiments

    print(f"\n{'=' * 72}")
    print("TOTAL BIGQUERY COST FOR ALL EXPERIMENTS:")
    print(f"  {bq_client.total_bytes_used:.1f} GB scanned")
    print(f"  {400 - bq_client.total_bytes_used:.1f} GB remaining in free tier")
    print("=" * 72)

    # Determine overall winner across all test cases
    all_winners: dict[str, int] = {}
    for tc_name, exps in all_results.items():
        if exps:
            winner = max(exps, key=lambda x: x["retrieval_metrics"]["composite_score"])
            name = winner["name"]
            all_winners[name] = all_winners.get(name, 0) + 1

    if all_winners:
        overall_winner = max(all_winners, key=all_winners.get)
        print(f"\n🏆 OVERALL WINNER: {overall_winner} (won {all_winners[overall_winner]}/{len(all_results)} test cases)")

    # Save results
    results_path = os.path.join(_ROOT, "experiments", "retrieval_experiment_results.json")
    serializable = {}
    for tc_name, exps in all_results.items():
        serializable[tc_name] = exps

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return all_results


if __name__ == "__main__":
    results = run_all_experiments()
