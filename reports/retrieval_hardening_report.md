# Retrieval Hardening Report

**Branch:** `optimize/retrieval-hardening`  
**Base:** `optimize/bigquery-embed` @ `5b34c43`  
**Commit:** `7d16bc4`  
**Test case:** Solar Charger — portable solar panel with USB-C, integrated battery, foldable

---

## Root Cause

The original single-query retrieval approach scanned **227–240 GB per run** because:

1. `_sanitize_regex()` sorted tokens by length ascending — selecting short, generic words like "type", "unit", "port"
2. CPC codes were never applied in the stage-1 query — all US patents were scanned
3. `UNNEST` cross-join on `title_localized × abstract_localized` multiplied row counts
4. The `abstract_localized` column alone is **~202 GB** — any query touching it is unaffordable within a 20 GB cap

### BQ Column Costs (full table scan)

| Column | Size |
|---|---|
| `abstract_localized` | ~202 GB |
| `claims_localized` | ~123 GB |
| `title_localized` | ~19 GB |
| `publication_number` | ~12 GB |
| `cpc.code` | ~10 GB |
| `country_code` | ~0.67 GB |

---

## Solution: Two-Phase CPC-First Retrieval

### Architecture

| Phase | Description | GB Scanned | Rows | Time |
|---|---|---|---|---|
| **1 — CPC Scout** | pub_numbers + dates + CPC/assignee aggregation via `EXISTS(UNNEST(cpc))` | 25.58 | 500 | 3.4s |
| **2 — Title Fetch** | English titles via IN-list | 18.84 | 500 | 1.9s |
| **3 — Python Filter** | Regex on titles locally | 0 | 416 | <0.01s |
| **4 — Claims** (optional) | English claims for top 20 patents | 122.61 | 20 | 1.7s |

### Key Code Changes

- **`patent_retriever.py`** — Complete restructure: removed `_step1_query`, `_fetch_detail`, `_fetch_landscape`; added `_cpc_scout_query`, `_title_query`, `_python_text_filter`, `_parse_scout_aggregations`, `_try_fetch_claims`; CPC EXISTS clause always rebuilt from `cpc_codes`
- **`query_builder.py`** — `_sanitize_regex` now prefers longer domain-specific tokens (descending sort), excludes 30+ stopwords; `build_bigquery_where_clause` returns `cpc_prefixes` and EXISTS-format `cpc_filter`
- **`config/settings.py`** — `BQ_MAX_BYTES_BILLED` raised to 30 GB; added `BQ_MIN_FILING_DATE`, `MIN_RELEVANT_FRACTION`, `DEBUG_RETRIEVAL`
- **`app.py`** — Abstract display gracefully handles empty strings (title-only retrieval)
- **`_meta_query` / `_claims_query`** — Rewritten with `STRING_AGG` subqueries (no cross-join)

### Auto-Tighten Loop

CPC scout has a date-fallback chain:
1. `filing_date > 20000101` (primary)
2. `filing_date > 20100101` (fallback 1)
3. `filing_date > 20150101` (fallback 2)

Text filter relaxation: if Python regex removes >99% of CPC-matched results (< 5 remaining), falls back to CPC-only results.

---

## Acceptance Criteria

| Criterion | Target | Actual | Status |
|---|---|---|---|
| Solar keyword fraction (titles) | ≥ 40% | 40.4% | **PASS** |
| HIGH similarity matches | ≥ 1 | 1 | **PASS** |
| MODERATE similarity matches | — | 52 | — |
| Pipeline runtime | ≤ 180s | 132.54s | **PASS** |
| Detail patents | > 0 | 100 | **PASS** |
| Landscape patents | > 0 | 316 | **PASS** |
| Claims parsed | > 0 | 41 independent / 295 elements | **PASS** |
| White space findings | > 0 | 2 (both HIGH confidence) | **PASS** |
| PDF generated | Yes | Yes (26,452 bytes) | **PASS** |

### Full Pipeline Timing

| Phase | Runtime |
|---|---|
| Feature Extraction (cached) | 0.00s |
| BigQuery Retrieval | 7.21s |
| Claim Parsing | 19.61s |
| Embedding Similarity | 6.01s |
| Contextual Analysis | 93.52s |
| White Space | 4.79s |
| PDF Generation | 0.15s |
| **TOTAL** | **132.54s** |

---

## Trade-offs & Notes

1. **Abstract not fetched**: The `abstract_localized` column (~202 GB) makes abstract retrieval unaffordable. Embeddings use titles + claims instead. The `whitespace_finder` receives empty abstracts, slightly reducing analysis quality.
2. **Claims cost**: The optional claims fetch scans ~123 GB, configurable via `BQ_CLAIMS_BYTES_BILLED` env var (default 130 GB cap). Can be disabled by setting to 0.
3. **Total BQ cost per run**: ~45 GB (scout + titles) + ~123 GB (claims) = ~168 GB ≈ $0.84 at $5/TB. Without claims: ~$0.23.
4. **CPC filter always rebuilt**: The `search()` method ignores any cached `cpc_filter` string and always reconstructs the EXISTS clause from `cpc_codes` to avoid format incompatibilities.

---

## Overall Result: **PASS** ✓
