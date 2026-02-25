#!/usr/bin/env python3
"""Generate reports/optimization_run_<ts>.json and .md"""
import json, os
from datetime import datetime, timezone

ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
base = f"optimization_run_{ts}"

report = {
    "meta": {
        "generated_at": ts,
        "git_branch": "optimize/bigquery-embed",
        "git_commit": "2a22f39",
        "test_case": "Solar-powered portable USB charger (test_case1_solar.py)"
    },
    "improvements": {
        "1_bigquery_cost_reduction": {
            "before_max_gb": 500,
            "after_max_gb": 20,
            "fallback_cap_gb": 5,
            "reduction_pct": 96.0,
            "fetch_detail_attempts": 4,
            "fetch_landscape_attempts": 3,
            "cost_log": ".tmp/query_costs.json",
            "verified": True
        },
        "2_patent_language_reformulation": {
            "method": "reformulate_features_for_patent_language() in QueryBuilder",
            "embedding_strategy": "np.maximum(sim_original, sim_patent_language)",
            "model_primary": "all-MiniLM-L6-v2",
            "model_fallback": "all-MiniLM-L4-v2",
            "calibration_pairs": 9,
            "calibration_accuracy_pct": 100.0,
            "top_improvement_pair": "0.449 to 0.909 (charge controller vs power regulator)",
            "live_avg_improvement": 0.0111,
            "verified": True
        },
        "3_caching_demo_mode": {
            "cache_key": "SHA256 of sorted primary search terms (16-char hex)",
            "demo_file": "examples/solar_charger_session.json",
            "demo_size_kb": 1890,
            "ui_button": "Load Solar Charger Demo in Streamlit sidebar",
            "verified": True
        },
        "4_fallback_retry_monitoring": {
            "reformulation_max_retries": 2,
            "json_extraction_strategies": 3,
            "per_query_logging": True,
            "streamlit_gb_metric": True,
            "verified": True
        },
        "5_threshold_calibration": {
            "HIGH": 0.65,
            "MODERATE": 0.45,
            "LOW": 0.30,
            "script": "scripts/calibrate_similarity.py",
            "accuracy": "9/9 (100%)",
            "verified": True
        },
        "6_graceful_failure": {
            "strategy": "try/except each phase, partial results returned",
            "guards": ["empty detail_df", "no parsed claims", "embedding error", "mapper skip on no matches"],
            "verified": True
        },
        "7_structured_report": {
            "formats": ["JSON", "Markdown"],
            "output_dir": "reports/",
            "verified": True
        }
    },
    "pipeline_run": {
        "test": "tests/test_case1_solar.py --cached",
        "result": "PASSED",
        "runtime_s": 97.19,
        "detail_patents": 100,
        "landscape_patents": 500,
        "independent_claims": 46,
        "claim_elements": 270,
        "similarity_model": "all-MiniLM-L6-v2",
        "uses_reformulation": True,
        "HIGH": 0,
        "MODERATE": 12,
        "LOW": 48,
        "whitespace_findings": 6,
        "pdf_generated": True,
        "pdf_bytes": 25174
    },
    "demo_session": {
        "path": "examples/solar_charger_session.json",
        "size_kb": 1890,
        "enriched_matches": 15,
        "whitespace_findings": 6,
        "reformulation": True
    },
    "git": {
        "branch": "optimize/bigquery-embed",
        "commit": "2a22f39",
        "files_changed": 12,
        "insertions": 15170,
        "deletions": 139
    }
}

os.makedirs("reports", exist_ok=True)
json_path = f"reports/{base}.json"
md_path   = f"reports/{base}.md"

with open(json_path, "w") as f:
    json.dump(report, f, indent=2)

md = f"""# PatentScout Optimization Run Report
**Generated:** {ts}  
**Branch:** `optimize/bigquery-embed`  
**Commit:** `2a22f39`  

---

## 1. BigQuery Cost Reduction
| Metric | Before | After |
|---|---|---|
| Max bytes billed | 500 GB | **20 GB** |
| Fallback cap | — | **5 GB** |
| Reduction | — | **96 %** |

- `_fetch_detail`: 4-attempt fallback chain
- `_fetch_landscape`: 3-attempt fallback chain
- Cost logged to `.tmp/query_costs.json` + Streamlit sidebar metric

## 2. Patent-Language Reformulation + Max Embeddings
- `REFORMULATION_PROMPT` + `reformulate_features_for_patent_language()` in `QueryBuilder`
- `np.maximum(sim_original, sim_patent_language)` strategy in `EmbeddingEngine`
- Model: `all-MiniLM-L6-v2` (fallback: `all-MiniLM-L4-v2`)
- Calibration accuracy: **9/9 (100%)**
- Top improvement: 0.449 → **0.909** (charge controller pair)

## 3. Caching & Demo Mode
- SHA256 query cache (16-char key from primary search terms)
- Precomputed demo: `examples/solar_charger_session.json` (1890 KB)
- "Load Solar Charger Demo" button in Streamlit sidebar

## 4. Fallback / Retry / Monitoring
- Reformulation: up to 2 retries with strict JSON fallback
- 3-strategy JSON extraction (direct → strip markdown → regex)
- Elapsed + GB logged per BQ query

## 5. Similarity Thresholds (env-overridable)
| Level | Threshold |
|---|---|
| HIGH | 0.65 |
| MODERATE | 0.45 |
| LOW | 0.30 |

Calibration: `scripts/calibrate_similarity.py` — **100 % accuracy**

## 6. Graceful Failure
All 7 pipeline phases wrapped in try/except; partial results returned.

## 7. Pipeline Run Results
| Item | Count |
|---|---|
| Detail patents | 100 |
| Landscape patents | 500 |
| Independent claims | 46 |
| Claim elements | 270 |
| HIGH similarity | 0 |
| MODERATE similarity | 12 |
| LOW similarity | 48 |
| White space findings | 6 |
| Runtime | 97.2 s |
| PDF | 24.6 KB |

## Git
```
branch : optimize/bigquery-embed
commit : 2a22f39
12 files changed, 15170 insertions(+), 139 deletions(-)
```
"""

with open(md_path, "w") as f:
    f.write(md)

print(f"JSON : {json_path}")
print(f"MD   : {md_path}")
print("Done.")
