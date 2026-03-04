"""
PatentScout Configuration Settings

Central location for all application constants and environment-driven
configuration values.  Every numeric threshold can be overridden at runtime
via the corresponding environment variable (see names below).
"""

import os
from dotenv import load_dotenv

load_dotenv()


# Google Cloud / BigQuery
BIGQUERY_DATASET = "patents-public-data.patents.publications"
BIGQUERY_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")

# ---------------------------------------------------------------------------
# Similarity thresholds (cosine similarity, 0–1)
# Override via env:  SIMILARITY_THRESHOLD_HIGH / MODERATE / LOW
# Calibrated defaults (2026-02): lower thresholds improve recall for
# patent-language descriptions since claim text differs from natural language.
# ---------------------------------------------------------------------------
SIMILARITY_THRESHOLD_HIGH     = float(os.getenv("SIMILARITY_THRESHOLD_HIGH",     "0.65"))
SIMILARITY_THRESHOLD_MODERATE = float(os.getenv("SIMILARITY_THRESHOLD_MODERATE", "0.45"))
SIMILARITY_THRESHOLD_LOW      = float(os.getenv("SIMILARITY_THRESHOLD_LOW",      "0.30"))


# Query limits
MAX_PATENTS_DETAIL   = 20    # Patents shown in detailed comparison view
MAX_PATENTS_LANDSCAPE = 500  # Patents loaded for landscape visualisation

BQ_QUERY_LIMIT_DETAIL    = 100   # LIMIT clause for detail queries
BQ_QUERY_LIMIT_LANDSCAPE = 500   # LIMIT clause for landscape queries


# BigQuery billing caps
# Phase 1 (CPC scout): ~26 GB.  Phase 2 (title fetch): ~19 GB.
# Cap per query at 30 GB to accommodate both phases with margin.
# Override via env:  BQ_MAX_BYTES_BILLED (applies to every individual job).
BQ_MAX_BYTES_BILLED: int = int(
    os.getenv("BQ_MAX_BYTES_BILLED", str(30_000_000_000))   # 30 GB default
)
# Fallback cap used when the primary cap is hit (title-only search).
BQ_FALLBACK_BYTES_BILLED: int = int(
    os.getenv("BQ_FALLBACK_BYTES_BILLED", str(5_000_000_000))  # 5 GB
)
# Maximum two-stage attempts before giving up.
BQ_MAX_FALLBACK_ATTEMPTS: int = int(os.getenv("BQ_MAX_FALLBACK_ATTEMPTS", "2"))

# Minimum filing date (YYYYMMDD) for stage-1 retrieval.  Narrows scan range.
BQ_MIN_FILING_DATE: int = int(os.getenv("BQ_MIN_FILING_DATE", "20000101"))

# Minimum fraction of stage-1 results that must contain a target keyword
# for the result set to be considered "relevant enough".
MIN_RELEVANT_FRACTION: float = float(os.getenv("MIN_RELEVANT_FRACTION", "0.10"))

# Debug flag for extra retrieval logging (set via env, default off)
DEBUG_RETRIEVAL: bool = os.getenv("DEBUG_RETRIEVAL", "0") in ("1", "true", "True")


# Query cost logging
# JSON file where per-query bytes / elapsed are appended.
QUERY_COST_LOG_PATH: str = os.getenv("QUERY_COST_LOG_PATH", ".tmp/query_costs.json")

# ---------------------------------------------------------------------------
# Embedding model
# Primary and fallback model names for sentence-transformers.
# Override via env:  EMBEDDING_MODEL_NAME
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME          = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_MODEL_FALLBACK_NAME = os.getenv("EMBEDDING_MODEL_FALLBACK_NAME", "all-MiniLM-L4-v2")


# Gemini reformulation settings
REFORMULATION_MAX_RETRIES: int = int(os.getenv("REFORMULATION_MAX_RETRIES", "2"))
CONTEXTUAL_ANALYSIS_MAX_PAIRS: int = int(os.getenv("CONTEXTUAL_ANALYSIS_MAX_PAIRS", "10"))
CONTEXTUAL_ANALYSIS_BATCH_SIZE: int = int(os.getenv("CONTEXTUAL_ANALYSIS_BATCH_SIZE", "5"))
