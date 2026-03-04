# experiments/query_cache.py
"""
CachedBigQueryClient
---------------------
Drop-in wrapper around google.cloud.bigquery.Client that caches query
results to disk so repeated identical queries never re-scan BigQuery.

Cache files live in experiments/cache/<md5_of_sql>.json and persist
across Python sessions.  The cache is intentionally dumb: same SQL
bytes → same result.  Parameterise your SQL carefully.

Usage::

    from experiments.query_cache import CachedBigQueryClient

    bq = CachedBigQueryClient()
    df = bq.query(sql, max_gb=2.0, description="detail patents – solar")
    print(bq.get_total_cost())
"""

import hashlib
import os

import pandas as pd
from google.cloud import bigquery

from config import settings

# ---------------------------------------------------------------------------
# Cache location
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(_ROOT, "experiments", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


class CachedBigQueryClient:
    """BigQuery client with transparent on-disk result caching."""

    def __init__(self) -> None:
        self.client = bigquery.Client(project=settings.BIGQUERY_PROJECT)
        self.total_bytes_used: float = 0.0  # GB scanned this session

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        sql: str,
        max_gb: float = 5.0,
        description: str = "",
    ) -> pd.DataFrame:
        """
        Run a BigQuery query with caching.

        If this exact SQL has been run before the cached DataFrame is
        returned immediately (0 bytes scanned).  Otherwise the query is
        executed, capped at *max_gb* GB, and the result is saved for
        future calls.

        Parameters
        ----------
        sql:         The SQL to execute.
        max_gb:      Hard billing cap in gigabytes (raises if exceeded).
        description: Human-readable label shown in log output.

        Returns
        -------
        pd.DataFrame – query results (empty DataFrame on cap violation).
        """
        cache_key = hashlib.md5(sql.encode()).hexdigest()
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")

        # ── Cache hit ──────────────────────────────────────────────────
        if os.path.exists(cache_file):
            print(f"  CACHE HIT:  {description}")
            print(f"  (0 GB scanned)")
            return pd.read_json(cache_file, orient="records")

        # ── Cache miss – run the real query ────────────────────────────
        print(f"  CACHE MISS: {description}")
        print(f"  Running BigQuery (cap: {max_gb} GB)…")

        job_config = bigquery.QueryJobConfig(
            maximum_bytes_billed=int(max_gb * 1024**3)
        )

        try:
            job = self.client.query(sql, job_config=job_config)
            df: pd.DataFrame = job.to_dataframe()

            gb_scanned = (job.total_bytes_processed or 0) / (1024**3)
            self.total_bytes_used += gb_scanned
            print(f"  Scanned:      {gb_scanned:.2f} GB")
            print(f"  Session total: {self.total_bytes_used:.2f} GB")

            # Persist to cache
            df.to_json(cache_file, orient="records", indent=2)
            print(f"  Cached → {cache_file}")

            return df

        except Exception as exc:
            if "bytesBilledLimit" in str(exc):
                print(f"  QUERY EXCEEDED {max_gb} GB CAP — SKIPPED")
                return pd.DataFrame()
            raise

    def invalidate(self, sql: str) -> bool:
        """
        Remove the cached result for *sql*.

        Returns True if a cache file was deleted, False if nothing was
        cached for that query.
        """
        cache_key = hashlib.md5(sql.encode()).hexdigest()
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"  Cache invalidated: {cache_file}")
            return True
        return False

    def get_total_cost(self) -> str:
        """Human-readable summary of GB scanned this session."""
        return f"{self.total_bytes_used:.1f} GB used this session"
