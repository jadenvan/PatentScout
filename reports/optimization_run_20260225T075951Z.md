# PatentScout Optimization Run Report
**Generated:** 20260225T075951Z  
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
