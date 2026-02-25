"""
PatentScout Configuration Settings

Central location for all application constants and environment-driven
configuration values.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Google Cloud / BigQuery
# ---------------------------------------------------------------------------
BIGQUERY_DATASET = "patents-public-data.patents.publications"
BIGQUERY_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")

# ---------------------------------------------------------------------------
# Similarity thresholds (cosine similarity, 0–1)
# ---------------------------------------------------------------------------
SIMILARITY_THRESHOLD_HIGH = 0.75
SIMILARITY_THRESHOLD_MODERATE = 0.50
SIMILARITY_THRESHOLD_LOW = 0.30

# ---------------------------------------------------------------------------
# Query limits
# ---------------------------------------------------------------------------
MAX_PATENTS_DETAIL = 20       # Patents shown in detailed comparison view
MAX_PATENTS_LANDSCAPE = 500   # Patents loaded for landscape visualisation

BQ_QUERY_LIMIT_DETAIL = 100   # LIMIT clause for detail queries
BQ_QUERY_LIMIT_LANDSCAPE = 500  # LIMIT clause for landscape queries

# Cap each BigQuery job at 500 GB billed.
# The patents-public-data.patents.publications table requires ~200-350 GB
# for a full-text REGEXP_CONTAINS scan across all US grants.  BigQuery's
# on-demand free tier provides 1 TB of free query processing per month, so
# most development / test runs fall within the free allocation.
BQ_MAX_BYTES_BILLED = 500_000_000_000  # 500 GB

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
