"""
PatentScout — Patent Retriever Module

Executes BigQuery queries against the patents-public-data public dataset
and returns normalised DataFrames for the Prior Art and Landscape tabs.

Design notes
------------
* Two-stage approach to minimise bytes scanned:
    Stage 1 — title + abstract text search → list of publication_numbers
    Stage 2 — targeted IN-list queries for CPC, assignees, and claims
* Every BigQuery job is capped at BQ_MAX_BYTES_BILLED (default 20 GB).
* If a job exceeds the cap an automatic fallback chain runs:
    Fallback 1: restrict filing_date window (>2010-01-01) — same cap
    Fallback 2: title-only search at BQ_FALLBACK_BYTES_BILLED (5 GB)
* Per-query bytes processed and elapsed time are appended to
  .tmp/query_costs.json and to st.session_state['query_costs'] when
  Streamlit is active (non-blocking — failures are silently suppressed).
"""

from __future__ import annotations

import json
import os
import re
import time
import logging
from typing import Optional

import pandas as pd
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPICallError

from config import settings

logger = logging.getLogger(__name__)

# Alternative cutoff used in fallback #1 (filing_date > 2010)
_DATE_CUTOFF      = 20000101   # primary
_DATE_CUTOFF_FB1  = 20100101   # fallback #1 — narrower date window


# ---------------------------------------------------------------------------
# Cost logging helpers
# ---------------------------------------------------------------------------

def _ensure_tmp_dir() -> None:
    """Ensure the .tmp directory exists (creates it silently if absent)."""
    os.makedirs(".tmp", exist_ok=True)


def _log_query_cost(
    query_name: str,
    bytes_processed: int,
    elapsed_s: float,
) -> None:
    """Append a per-query cost record to .tmp/query_costs.json.

    Also pushes to st.session_state['query_costs'] and increments
    st.session_state['total_gb_scanned'] when Streamlit is available.
    """
    _ensure_tmp_dir()
    gb = bytes_processed / 1e9
    record = {
        "query_name": query_name,
        "gb":         round(gb, 4),
        "elapsed_s":  round(elapsed_s, 3),
        "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # -- Write to JSON log -------------------------------------------------
    log_path = settings.QUERY_COST_LOG_PATH
    try:
        _ensure_tmp_dir()
        existing: list = []
        if os.path.exists(log_path):
            with open(log_path) as fh:
                try:
                    existing = json.load(fh)
                except (json.JSONDecodeError, ValueError):
                    existing = []
        existing.append(record)
        with open(log_path, "w") as fh:
            json.dump(existing, fh, indent=2)
    except Exception as exc:
        logger.warning("Could not write query cost log: %s", exc)

    # -- Push to Streamlit session state (non-blocking) -------------------
    try:
        import streamlit as st
        if "query_costs" not in st.session_state:
            st.session_state["query_costs"] = []
        st.session_state["query_costs"].append(record)
        prev = st.session_state.get("total_gb_scanned", 0.0) or 0.0
        st.session_state["total_gb_scanned"] = round(prev + gb, 4)
    except Exception:
        pass  # Streamlit not available — ignore


class PatentRetriever:
    """
    Retrieves patent records from Google BigQuery patents-public-data.

    Usage::

        retriever = PatentRetriever()
        detail_df, landscape_df = retriever.search(search_strategy)
    """

    def __init__(self, bq_client: "bigquery.Client | None" = None) -> None:
        """
        Args:
            bq_client: Optional pre-built BigQuery client. When None (default),
                       a client is created using the ambient application
                       credentials (GOOGLE_APPLICATION_CREDENTIALS env var or
                       Workload Identity). Pass a pre-built client when running
                       on Streamlit Cloud with secrets-based credentials.
        """
        if bq_client is not None:
            self._client = bq_client
        else:
            self._client = bigquery.Client(project=settings.BIGQUERY_PROJECT)
        # Primary job config — capped at configured limit
        self._job_config = bigquery.QueryJobConfig(
            maximum_bytes_billed=settings.BQ_MAX_BYTES_BILLED
        )
        # Fallback job config — lower cap for title-only searches
        self._fallback_job_config = bigquery.QueryJobConfig(
            maximum_bytes_billed=settings.BQ_FALLBACK_BYTES_BILLED
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self, search_strategy: dict
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run detail + landscape queries and return (detail_df, landscape_df).

        Args:
            search_strategy: Dict produced by QueryBuilder that must contain
                             a ``text_filter`` key with a BigQuery-ready
                             WHERE expression string.

        Returns:
            ``(detail_df, landscape_df)`` — both enriched with cpc_code lists,
            assignee_name lists, parsed date strings, relevance scores, and
            Google Patents URLs.
        """
        text_filter = search_strategy.get("text_filter", "TRUE")
        # ── Debug: always print the filter so we can inspect it in the terminal
        print("\n" + "=" * 72)
        print("[PatentRetriever] text_filter:\n" + text_filter)
        print("=" * 72 + "\n")
        logger.debug("text_filter:\n%s", text_filter)

        predicted_cpc = [
            c.get("code", "") for c in search_strategy.get("cpc_codes", [])
        ]

        # -- Detail patents -----------------------------------------------
        detail_df = self._fetch_detail(text_filter)
        detail_df = self._enrich_with_metadata(detail_df, fetch_claims=True)
        self._add_computed_columns(detail_df)
        self._score_relevance(detail_df, predicted_cpc, search_strategy)

        # -- Landscape patents --------------------------------------------
        landscape_df = self._fetch_landscape(text_filter)
        landscape_df = self._enrich_with_metadata(landscape_df, fetch_claims=False)
        self._add_computed_columns(landscape_df)

        logger.info(
            "PatentRetriever: detail=%d rows  landscape=%d rows",
            len(detail_df),
            len(landscape_df),
        )
        return detail_df, landscape_df

    # ------------------------------------------------------------------
    # Query builders
    # ------------------------------------------------------------------

    def _step1_query(
        self,
        text_filter: str,
        limit: int,
        include_claims: bool = False,  # always ignored — claims fetched in Step 2
        date_cutoff: int = _DATE_CUTOFF,
    ) -> str:
        """Return a BigQuery SQL for the first step (core patent fields).

        Claims are intentionally excluded here: UNNEST(claims_localized) on the
        full publications table costs ~350 GB per query.  Claims are fetched
        cheaply in a targeted Step-2 query once we have a short patent list.

        Args:
            date_cutoff: integer YYYYMMDD lower bound for filing_date.  Using a
                         more recent cutoff (e.g. 20100101) is the primary
                         mechanism for reducing bytes scanned when the full
                         range query hits the billing cap.
        """
        return f"""
        SELECT DISTINCT
            publication_number,
            title.text    AS title,
            abstract.text AS abstract,
            filing_date,
            grant_date,
            publication_date,
            country_code,
            family_id
        FROM
            `patents-public-data.patents.publications`,
            UNNEST(title_localized)    AS title,
            UNNEST(abstract_localized) AS abstract
        WHERE
            title.language    = 'en'
            AND abstract.language = 'en'
            AND country_code  = 'US'
            AND grant_date    > 0
            AND filing_date   > {date_cutoff}
            AND ({text_filter})
        ORDER BY publication_date DESC
        LIMIT {limit}
        """

    def _meta_query(self, publication_numbers: list[str]) -> str:
        """Return SQL that fetches CPC codes + assignees for given patents."""
        patents_in = ", ".join(f"'{p}'" for p in publication_numbers)
        return f"""
        SELECT
            publication_number,
            assignee.name AS assignee_name,
            cpc.code      AS cpc_code
        FROM
            `patents-public-data.patents.publications`,
            UNNEST(assignee_harmonized) AS assignee,
            UNNEST(cpc) AS cpc
        WHERE
            publication_number IN ({patents_in})
        """

    def _claims_query(self, publication_numbers: list[str]) -> str:
        """Return SQL that fetches English claims text for a short list of patents.

        This is cheap because it is filtered to a small IN-list of specific
        publication_numbers rather than scanning the full table.
        """
        patents_in = ", ".join(f"'{p}'" for p in publication_numbers)
        return f"""
        SELECT
            publication_number,
            STRING_AGG(claims.text, ' | ' ORDER BY claims.text) AS claims_text
        FROM
            `patents-public-data.patents.publications`,
            UNNEST(claims_localized) AS claims
        WHERE
            publication_number IN ({patents_in})
            AND claims.language = 'en'
        GROUP BY publication_number
        """

    # ------------------------------------------------------------------
    # Fetch helpers
    # ------------------------------------------------------------------

    def _run_query(
        self,
        sql: str,
        stage_label: str = "",
        job_config: "bigquery.QueryJobConfig | None" = None,
    ) -> pd.DataFrame:
        """Execute *sql* with billing cap and return a DataFrame.

        ``create_bqstorage_client=False`` forces the standard paginated REST
        downloader rather than the BigQuery Storage Read API, avoiding the need
        for the ``bigquery.readsessions.create`` IAM permission.

        Per-query bytes processed and elapsed time are appended to the cost log.
        """
        if job_config is None:
            job_config = self._job_config

        t0 = time.time()
        job = self._client.query(sql, job_config=job_config)
        df  = job.to_dataframe(create_bqstorage_client=False)
        elapsed = time.time() - t0

        bytes_billed    = job.total_bytes_billed    or 0
        bytes_processed = job.total_bytes_processed or 0
        label = f"[{stage_label}] " if stage_label else ""
        logger.info(
            "%sQuery billed=%.2f MB  processed=%.2f MB  rows=%d  elapsed=%.2fs",
            label, bytes_billed / 1e6, bytes_processed / 1e6, len(df), elapsed,
        )
        print(
            f"[BQ]{f' {stage_label}' if stage_label else ''} "
            f"billed={bytes_billed/1e6:.2f} MB  "
            f"processed={bytes_processed/1e6:.2f} MB  "
            f"elapsed={elapsed:.2f}s  rows={len(df)}"
        )
        # Log costs for monitoring / cost acceptance-criteria check
        _log_query_cost(
            query_name=stage_label or "unknown",
            bytes_processed=bytes_processed,
            elapsed_s=elapsed,
        )
        return df

    def _fetch_detail(self, text_filter: str) -> pd.DataFrame:
        """Fetch full detail records including claims text, with fallbacks.

        Fallback chain:
          1. Primary regex on title+abstract (primary cap, full date range)
          2. Same regex, restricted to filing_date > 2010 (reduces scan size)
          3. Title-only regex (fallback cap — 5 GB)
          4. Broad LIKE on title+abstract (fallback cap)
        """
        # ── Attempt 1 — full regex, primary cap ──────────────────────────
        try:
            sql = self._step1_query(text_filter, limit=settings.BQ_QUERY_LIMIT_DETAIL)
            print("\n[PatentRetriever] Detail attempt 1 (regex, primary cap):\n" + sql)
            df = self._run_query(sql, stage_label="detail-regex")
            logger.info("Detail attempt 1 (regex): %d rows", len(df))
            if not df.empty:
                return df
            print("[PatentRetriever] attempt 1 returned 0 rows — fallback")
        except (GoogleAPICallError, Exception) as exc:
            msg = str(exc)
            if "bytesBilledLimitExceeded" in msg or "exceeds" in msg.lower():
                print(f"[PatentRetriever] attempt 1 HIT BILLING CAP — trying narrower date window")
            else:
                print(f"[PatentRetriever] attempt 1 FAILED: {exc}")
            logger.warning("Detail regex query failed (%s)", exc)

        # ── Attempt 2 — narrower date window (2010+) ─────────────────────
        like_filter = _regex_filter_to_like(text_filter)
        try:
            sql2 = self._step1_query(
                text_filter,
                limit=settings.BQ_QUERY_LIMIT_DETAIL,
                date_cutoff=_DATE_CUTOFF_FB1,
            )
            print("[PatentRetriever] Detail attempt 2 (regex, 2010+ date window):\n" + sql2)
            df = self._run_query(sql2, stage_label="detail-regex-2010", job_config=self._job_config)
            logger.info("Detail attempt 2 (2010+ window): %d rows", len(df))
            if not df.empty:
                return df
            print("[PatentRetriever] attempt 2 returned 0 rows — fallback")
        except (GoogleAPICallError, Exception) as exc:
            logger.warning("Detail 2010+ window query failed (%s)", exc)
            print(f"[PatentRetriever] attempt 2 FAILED: {exc}")

        # ── Attempt 3 — title-only keyword regex, fallback cap ────────────
        if like_filter:
            title_filter = f"REGEXP_CONTAINS(title.text, r'(?i)({like_filter})')"
            try:
                sql3 = self._step1_query(
                    title_filter,
                    limit=settings.BQ_QUERY_LIMIT_DETAIL,
                )
                print("[PatentRetriever] Detail attempt 3 (title-only, fallback cap):\n" + sql3)
                df = self._run_query(
                    sql3,
                    stage_label="detail-kw-title",
                    job_config=self._fallback_job_config,
                )
                logger.info("Detail attempt 3 (title keyword regex): %d rows", len(df))
                if not df.empty:
                    return df
                print("[PatentRetriever] attempt 3 returned 0 rows — LIKE fallback")
            except (GoogleAPICallError, Exception) as exc:
                logger.warning("Detail title regex query failed (%s)", exc)
                print(f"[PatentRetriever] attempt 3 FAILED: {exc}")

        # ── Attempt 4 — broad LIKE on title + abstract, fallback cap ──────
        broad_keywords = _broad_keywords_from_filter(text_filter)
        if broad_keywords:
            abstract_likes = " OR ".join(
                f"abstract.text LIKE '%{kw}%'" for kw in broad_keywords
            )
            title_likes = " OR ".join(
                f"title.text LIKE '%{kw}%'" for kw in broad_keywords
            )
            broad_filter = f"(({abstract_likes}) OR ({title_likes}))"
            sql4 = self._step1_query(
                broad_filter,
                limit=settings.BQ_QUERY_LIMIT_DETAIL,
            )
            print("[PatentRetriever] Detail attempt 4 (LIKE broad, fallback cap):\n" + sql4)
            try:
                df = self._run_query(
                    sql4,
                    stage_label="detail-LIKE-broad",
                    job_config=self._fallback_job_config,
                )
                logger.info("Detail attempt 4 (LIKE broad): %d rows", len(df))
                if not df.empty:
                    return df
                print("[PatentRetriever] attempt 4 returned 0 rows")
            except (GoogleAPICallError, Exception) as exc:
                logger.warning("Detail LIKE broad query failed (%s)", exc)
                print(f"[PatentRetriever] attempt 4 FAILED: {exc}")

        logger.warning("All detail query attempts returned 0 rows for filter: %s", text_filter)
        return pd.DataFrame()

    def _fetch_landscape(self, text_filter: str) -> pd.DataFrame:
        """Fetch broader landscape set (no claims) with fallbacks.

        Uses a separate, slimmer query (no abstract UNNEST text match needed)
        so the cost is lower than the detail query.
        """
        # ── Attempt 1 — regex search, primary cap ────────────────────────
        landscape_sql = f"""
        SELECT DISTINCT
            publication_number,
            title.text   AS title,
            filing_date,
            grant_date,
            publication_date,
            country_code
        FROM
            `patents-public-data.patents.publications`,
            UNNEST(title_localized)    AS title,
            UNNEST(abstract_localized) AS abstract
        WHERE
            title.language    = 'en'
            AND abstract.language = 'en'
            AND country_code  = 'US'
            AND grant_date    > 0
            AND filing_date   > {_DATE_CUTOFF}
            AND ({text_filter})
        ORDER BY publication_date DESC
        LIMIT {settings.BQ_QUERY_LIMIT_LANDSCAPE}
        """
        print("[PatentRetriever] Landscape attempt 1 (regex, primary cap):\n" + landscape_sql)
        try:
            df = self._run_query(landscape_sql, stage_label="landscape-regex")
            logger.info("Landscape attempt 1 (regex): %d rows", len(df))
            if not df.empty:
                return df
            print("[PatentRetriever] landscape attempt 1 returned 0 rows — fallback")
        except (GoogleAPICallError, Exception) as exc:
            msg = str(exc)
            if "bytesBilledLimitExceeded" in msg or "exceeds" in msg.lower():
                print("[PatentRetriever] landscape attempt 1 HIT BILLING CAP — reducing to title-only")
            else:
                print(f"[PatentRetriever] landscape attempt 1 FAILED: {exc}")
            logger.warning("Landscape regex query failed (%s)", exc)

        # ── Attempt 2 — title-only keyword regex, fallback cap ─────────
        like_filter = _regex_filter_to_like(text_filter)
        if like_filter:
            fallback_sql = f"""
            SELECT DISTINCT
                publication_number,
                title.text   AS title,
                filing_date,
                grant_date,
                publication_date,
                country_code
            FROM
                `patents-public-data.patents.publications`,
                UNNEST(title_localized) AS title
            WHERE
                title.language = 'en'
                AND country_code = 'US'
                AND grant_date   > 0
                AND filing_date  > {_DATE_CUTOFF}
                AND REGEXP_CONTAINS(title.text, r'(?i)({like_filter})')
            ORDER BY publication_date DESC
            LIMIT {settings.BQ_QUERY_LIMIT_LANDSCAPE}
            """
            print("[PatentRetriever] Landscape attempt 2 (title keyword regex, fallback cap):\n" + fallback_sql)
            try:
                df = self._run_query(
                    fallback_sql,
                    stage_label="landscape-kw-title",
                    job_config=self._fallback_job_config,
                )
                logger.info("Landscape attempt 2 (title keyword regex): %d rows", len(df))
                if not df.empty:
                    return df
                print("[PatentRetriever] landscape attempt 2 returned 0 rows — LIKE fallback")
            except (GoogleAPICallError, Exception) as exc:
                logger.warning("Landscape keyword title query failed (%s)", exc)
                print(f"[PatentRetriever] landscape attempt 2 FAILED: {exc}")

        # ── Attempt 3 — broad LIKE on title.text, fallback cap ────────
        broad_keywords = _broad_keywords_from_filter(text_filter)
        if broad_keywords:
            title_likes = " OR ".join(
                f"title.text LIKE '%{kw}%'" for kw in broad_keywords
            )
            broad_sql = f"""
            SELECT DISTINCT
                publication_number,
                title.text   AS title,
                filing_date,
                grant_date,
                publication_date,
                country_code
            FROM
                `patents-public-data.patriots.publications`,
                UNNEST(title_localized) AS title
            WHERE
                title.language = 'en'
                AND country_code = 'US'
                AND grant_date   > 0
                AND filing_date  > {_DATE_CUTOFF}
                AND ({title_likes})
            ORDER BY publication_date DESC
            LIMIT {settings.BQ_QUERY_LIMIT_LANDSCAPE}
            """
            # Fix: correct dataset name
            broad_sql = broad_sql.replace(
                "`patents-public-data.patriots.publications`",
                "`patents-public-data.patents.publications`",
            )
            print("[PatentRetriever] Landscape attempt 3 (LIKE broad, fallback cap):\n" + broad_sql)
            try:
                df = self._run_query(
                    broad_sql,
                    stage_label="landscape-LIKE-broad",
                    job_config=self._fallback_job_config,
                )
                logger.info("Landscape attempt 3 (LIKE broad): %d rows", len(df))
                if not df.empty:
                    return df
                print("[PatentRetriever] landscape attempt 3 returned 0 rows")
            except (GoogleAPICallError, Exception) as exc:
                logger.warning("Landscape LIKE broad query failed (%s)", exc)
                print(f"[PatentRetriever] landscape attempt 3 FAILED: {exc}")

        logger.warning("All landscape query attempts returned 0 rows for filter: %s", text_filter)
        return pd.DataFrame()

    def _enrich_with_metadata(
        self, df: pd.DataFrame, fetch_claims: bool = False
    ) -> pd.DataFrame:
        """Attach aggregated CPC code lists, assignee lists, and optionally
        claims text to *df*.

        Claims are fetched via a targeted IN-list query (cheap) rather than
        the full-table UNNEST used in Step 1.
        """
        if df.empty:
            return df

        pub_numbers = df["publication_number"].tolist()
        try:
            meta_df = self._run_query(self._meta_query(pub_numbers), stage_label="meta")
        except (GoogleAPICallError, Exception) as exc:
            logger.warning("Metadata query failed (%s) — skipping enrichment", exc)
            df["cpc_code"]      = [[] for _ in range(len(df))]
            df["assignee_name"] = [[] for _ in range(len(df))]
            df["claims_text"]   = [""] * len(df)
            return df

        if not meta_df.empty:
            cpc_agg = (
                meta_df.groupby("publication_number")["cpc_code"]
                .apply(list)
                .reset_index()
            )
            assignee_agg = (
                meta_df.groupby("publication_number")["assignee_name"]
                .apply(lambda x: list(set(x)))
                .reset_index()
            )
            df = df.merge(cpc_agg, on="publication_number", how="left")
            df = df.merge(assignee_agg, on="publication_number", how="left")
        else:
            df["cpc_code"]      = [[] for _ in range(len(df))]
            df["assignee_name"] = [[] for _ in range(len(df))]

        # Fetch claims text via targeted query (cheap — filtered by pub number)
        if fetch_claims and pub_numbers:
            try:
                claims_df = self._run_query(
                    self._claims_query(pub_numbers), stage_label="claims"
                )
                if not claims_df.empty:
                    df = df.merge(
                        claims_df[["publication_number", "claims_text"]],
                        on="publication_number",
                        how="left",
                    )
                else:
                    df["claims_text"] = ""
            except (GoogleAPICallError, Exception) as exc:
                logger.warning("Claims query failed (%s) — leaving blank", exc)
                df["claims_text"] = ""
        elif "claims_text" not in df.columns:
            df["claims_text"] = ""

        # Ensure list columns contain actual lists (not NaN after left join)
        for col in ("cpc_code", "assignee_name"):
            df[col] = df[col].apply(
                lambda v: v if isinstance(v, list) else []
            )
        # Ensure claims_text is always a string (not NaN after left join)
        if "claims_text" in df.columns:
            df["claims_text"] = df["claims_text"].fillna("").astype(str)
        return df

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _add_computed_columns(self, df: pd.DataFrame) -> None:
        """Add patent_url and human-readable date columns in-place."""
        if df.empty:
            return
        df["patent_url"] = df["publication_number"].apply(
            self._format_patent_url
        )
        for col in ("filing_date", "grant_date", "publication_date"):
            if col in df.columns:
                df[col + "_str"] = df[col].apply(self._parse_date)

    def _score_relevance(
        self,
        df: pd.DataFrame,
        predicted_cpc: list[str],
        search_strategy: dict,
    ) -> None:
        """
        Add ``relevance_score`` column and sort *df* descending (in-place).

        Scoring breakdown:
        - 0.40 — CPC overlap fraction with predicted codes
        - 0.30 — Recency (year normalised between 2000–2026)
        - 0.30 — Keyword density in abstract
        """
        if df.empty:
            return

        # Pre-compute keyword set from search_terms
        keywords: set[str] = set()
        for term in search_strategy.get("search_terms", []):
            primary = term.get("primary", "")
            if primary:
                keywords.add(primary.lower())
            for syn in term.get("synonyms", []):
                keywords.add(syn.lower())

        # Normalise years 2000–2026
        min_year, max_year = 2000, 2026
        year_range = max_year - min_year or 1

        scores: list[float] = []
        for _, row in df.iterrows():
            # --- CPC overlap ----------------------------------------
            pat_cpcs = row.get("cpc_code") if isinstance(row.get("cpc_code"), list) else []
            cpc_score = 0.0
            if predicted_cpc and pat_cpcs:
                matches = sum(
                    1
                    for pc in predicted_cpc
                    if any(
                        pc[:4].upper() in str(ac).upper()
                        for ac in pat_cpcs
                    )
                )
                cpc_score = min(matches / max(len(predicted_cpc), 1), 1.0)

            # --- Recency -------------------------------------------
            pub_date = row.get("publication_date", 0) or 0
            year = int(str(pub_date)[:4]) if pub_date else min_year
            year_score = (year - min_year) / year_range

            # --- Keyword density -----------------------------------
            abstract = str(row.get("abstract", "")).lower()
            kw_hits  = sum(1 for kw in keywords if kw in abstract)
            kw_score = (
                min(kw_hits / max(len(keywords), 1), 1.0) if keywords else 0.0
            )

            score = 0.4 * cpc_score + 0.3 * year_score + 0.3 * kw_score
            scores.append(round(score, 4))

        df["relevance_score"] = scores
        df.sort_values("relevance_score", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_date(date_int) -> Optional[str]:
        """Convert YYYYMMDD integer to 'YYYY-MM-DD' string."""
        try:
            if not date_int or int(date_int) == 0:
                return None
            s = str(int(date_int))
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _format_patent_url(publication_number: str) -> str:
        """Convert 'US-7479949-B2' → 'https://patents.google.com/patent/US7479949B2'."""
        clean = str(publication_number).replace("-", "")
        return f"https://patents.google.com/patent/{clean}"


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _regex_filter_to_like(text_filter: str) -> str:
    """
    Extract plain ASCII words from a BigQuery REGEXP_CONTAINS filter string
    and return a pipe-joined alternation suitable for a simpler
    REGEXP_CONTAINS fallback.

    Returns empty string if extraction yields nothing useful.
    """
    literals = re.findall(r"r'([^']*)'", text_filter)
    words: list[str] = []
    for lit in literals:
        found = re.findall(r"[a-zA-Z]{4,}", lit)
        words.extend(found[:3])          # max 3 words per pattern
    unique_words = list(dict.fromkeys(w.lower() for w in words))[:8]
    return "|".join(unique_words)


def _broad_keywords_from_filter(text_filter: str) -> list[str]:
    """
    Extract up to 5 short, common keywords from *text_filter* for use in
    LIKE '%keyword%' fallback clauses.

    Prefers shorter words (more likely to appear in titles/abstracts) and
    skips common regex/SQL noise tokens.
    """
    _NOISE = {"true", "false", "regexp", "contains", "abstract", "title", "text"}
    literals = re.findall(r"r'([^']*)'", text_filter)
    raw_words: list[str] = []
    for lit in literals:
        raw_words.extend(re.findall(r"[a-zA-Z]{4,}", lit))
    # Also mine any bare words from the filter (e.g. TRUE fallback)
    raw_words.extend(re.findall(r"[a-zA-Z]{4,}", text_filter))
    unique = list(dict.fromkeys(
        w.lower() for w in raw_words if w.lower() not in _NOISE
    ))
    # Sort by length ascending (shorter = more general) and take top 5
    unique.sort(key=len)
    return unique[:5]
