"""
PatentScout — Patent Retriever Module

Executes BigQuery queries against the patents-public-data public dataset
and returns normalised DataFrames for the Prior Art and Landscape tabs.

Design notes
------------
* Two-phase approach to minimise bytes scanned:
    Phase 1 — CPC-filtered scout: pub_numbers + dates + CPC/assignee (~22 GB)
    Phase 2 — Title fetch for matched patents via IN-list (~19 GB)
    Phase 3 — Python-side text regex filter on titles (0 GB)
* Abstract is NOT fetched from BQ (~202 GB column). Embedding uses titles.
* Claims are optionally fetched with a separate high cap (~123 GB) and
  are skipped gracefully if too expensive.
* Per-query bytes processed and elapsed time are appended to
  .tmp/query_costs.json and to st.session_state['query_costs'] when
  Streamlit is active (non-blocking — failures silently suppressed).
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
_DATE_CUTOFF      = settings.BQ_MIN_FILING_DATE   # primary (from settings)
_DATE_CUTOFF_FB1  = 20100101   # fallback #1 — narrower date window
_DATE_CUTOFF_FB2  = 20150101   # fallback #2 — even narrower

# Solar-domain keywords for sanity checks (case-insensitive)
_SOLAR_KEYWORDS = [
    "solar", "photovoltaic", "charger", "battery", "usb",
    "power bank", "portable", "foldable",
]


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
        Two-phase retrieval: CPC scout → title fetch → Python filter.

        Phase 1 (CPC Scout, ~22 GB): publication_numbers + dates +
            aggregated CPC codes & assignee names, filtered by CPC EXISTS.
        Phase 2 (Title Fetch, ~19 GB): English titles for matched patents.
        Phase 3 (Python, 0 GB): Apply text regex from ``text_filter`` to
            titles locally.  Split results into detail (top N) and
            landscape (remaining).

        Returns:
            ``(detail_df, landscape_df)`` — both enriched with cpc_code lists,
            assignee_name lists, parsed date strings, relevance scores, and
            Google Patents URLs.
        """
        text_filter  = search_strategy.get("text_filter", "TRUE")

        # Always rebuild CPC EXISTS clause from cpc_codes — the strategy
        # dict may contain a cpc_filter in the old single-query format
        # (cpc_item.code LIKE ...) which is incompatible with the two-phase
        # scout query that uses UNNEST(cpc) AS c.
        cpc_prefixes = self._extract_cpc_prefixes(search_strategy)
        cpc_filter   = self._build_cpc_exists_clause(cpc_prefixes)

        print("\n" + "=" * 72)
        print("[PatentRetriever] text_filter:\n" + text_filter)
        print("[PatentRetriever] cpc_filter:\n" + cpc_filter)
        print("=" * 72 + "\n")
        logger.debug("text_filter:\n%s", text_filter)
        logger.debug("cpc_filter:\n%s", cpc_filter)

        # Save for diagnostics
        try:
            import streamlit as st
            st.session_state["last_text_filter"] = text_filter
            st.session_state["last_cpc_filter"]  = cpc_filter
        except Exception:
            pass

        predicted_cpc = [
            c.get("code", "") for c in search_strategy.get("cpc_codes", [])
        ]

        # ── Phase 1: CPC Scout ──────────────────────────────────────────
        scout_df = self._cpc_scout(cpc_filter)
        if scout_df.empty:
            logger.warning("CPC scout returned 0 rows — returning empty")
            empty = pd.DataFrame()
            return empty, empty

        # ── Phase 2: Title Fetch ────────────────────────────────────────
        pub_numbers = scout_df["publication_number"].tolist()
        try:
            title_df = self._run_query(
                self._title_query(pub_numbers), stage_label="title-fetch",
            )
            if not title_df.empty:
                scout_df = scout_df.merge(title_df, on="publication_number", how="left")
        except (GoogleAPICallError, Exception) as exc:
            logger.warning("Title fetch failed (%s)", exc)
            print(f"[PatentRetriever] title fetch FAILED: {exc}")
        if "title" not in scout_df.columns:
            scout_df["title"] = ""

        # ── Parse CPC/assignee aggregation ──────────────────────────────
        self._parse_scout_aggregations(scout_df)

        # ── Phase 3: Python text filter on titles ───────────────────────
        filtered_df = self._python_text_filter(scout_df, text_filter)
        n_before, n_after = len(scout_df), len(filtered_df)
        print(f"  Text filter: {n_after}/{n_before} rows passed")

        # If text filter removed too many, relax to CPC-only
        if n_after < 5 and n_before >= 5:
            print("  Text filter too strict — using CPC-only results")
            filtered_df = scout_df

        # Relevance sanity check
        frac = self._check_relevance(filtered_df)
        print(f"  Relevance: {frac:.1%} ({len(filtered_df)} rows)")

        # Abstract not fetched from BQ (too expensive at 202 GB)
        if "abstract" not in filtered_df.columns:
            filtered_df["abstract"] = ""

        # Sort by publication_date DESC
        if "publication_date" in filtered_df.columns:
            filtered_df = filtered_df.sort_values("publication_date", ascending=False)
        filtered_df = filtered_df.reset_index(drop=True)

        # ── Split into detail + landscape ───────────────────────────────
        detail_limit = settings.BQ_QUERY_LIMIT_DETAIL
        detail_df    = filtered_df.head(detail_limit).copy()
        landscape_df = filtered_df.iloc[detail_limit:].copy().reset_index(drop=True)

        # Ensure claims_text column exists (may be fetched later)
        if "claims_text" not in detail_df.columns:
            detail_df["claims_text"] = ""
        if "claims_text" not in landscape_df.columns:
            landscape_df["claims_text"] = ""

        # ── Post-processing ─────────────────────────────────────────────
        self._add_computed_columns(detail_df)
        self._score_relevance(detail_df, predicted_cpc, search_strategy)
        self._add_computed_columns(landscape_df)

        # ── Optional: Fetch claims for top detail patents ───────────────
        if not detail_df.empty:
            self._try_fetch_claims(detail_df)

        logger.info(
            "PatentRetriever: detail=%d rows  landscape=%d rows",
            len(detail_df), len(landscape_df),
        )
        return detail_df, landscape_df

    # ------------------------------------------------------------------
    # CPC helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_cpc_prefixes(search_strategy: dict) -> list[str]:
        """Extract unique CPC group prefixes from search_strategy['cpc_codes']."""
        prefixes: list[str] = []
        for cpc in search_strategy.get("cpc_codes", []):
            code = cpc.get("code", "").strip()
            if code:
                prefix = re.split(r"[/.]", code)[0]
                if prefix and prefix not in prefixes:
                    prefixes.append(prefix)
        return prefixes

    @staticmethod
    def _build_cpc_exists_clause(cpc_prefixes: list[str]) -> str:
        """Build a BigQuery EXISTS(SELECT 1 FROM UNNEST(cpc)...) clause."""
        if not cpc_prefixes:
            return ""
        likes = [f"c.code LIKE '{p}%'" for p in cpc_prefixes]
        return (
            "EXISTS (SELECT 1 FROM UNNEST(cpc) AS c WHERE\n      "
            + "\n      OR ".join(likes)
            + "\n    )"
        )

    @staticmethod
    def _check_relevance(
        df: pd.DataFrame,
        keywords: list[str] | None = None,
    ) -> float:
        """Return the fraction of rows whose title or abstract contains
        at least one keyword from *keywords* (case-insensitive).

        Handles both pre-rename (pub_title/pub_abstract) and post-rename
        (title/abstract) column names.
        """
        if df.empty:
            return 0.0
        if keywords is None:
            keywords = _SOLAR_KEYWORDS
        # Support both column naming conventions
        title_col = "pub_title" if "pub_title" in df.columns else "title"
        abs_col = "pub_abstract" if "pub_abstract" in df.columns else "abstract"
        hits = 0
        for _, row in df.iterrows():
            text = (str(row.get(title_col, "")) + " " + str(row.get(abs_col, ""))).lower()
            if any(kw in text for kw in keywords):
                hits += 1
        return hits / len(df)

    # ------------------------------------------------------------------
    # Query builders
    # ------------------------------------------------------------------

    def _cpc_scout_query(
        self,
        cpc_filter: str,
        limit: int = 500,
        date_cutoff: int = _DATE_CUTOFF,
    ) -> str:
        """Phase 1 SQL: CPC-filtered patent discovery.

        Scans ~22 GB (publication_number + dates + cpc.code + assignee.name).
        Does NOT access title_localized or abstract_localized.
        Includes STRING_AGG for CPC codes and assignees so no separate
        metadata query is needed.
        """
        cpc_clause = f"AND {cpc_filter}" if cpc_filter else ""
        return f"""
        SELECT
            publication_number,
            filing_date,
            grant_date,
            publication_date,
            country_code,
            family_id,
            (SELECT STRING_AGG(c.code, ' | ') FROM UNNEST(cpc) c) AS cpc_codes_agg,
            (SELECT STRING_AGG(a.name, ' | ') FROM UNNEST(assignee_harmonized) a) AS assignees_agg
        FROM `patents-public-data.patents.publications`
        WHERE country_code = 'US'
          AND grant_date > 0
          AND filing_date > {date_cutoff}
          {cpc_clause}
        ORDER BY publication_date DESC
        LIMIT {limit}
        """

    def _title_query(self, publication_numbers: list[str]) -> str:
        """Phase 2 SQL: Fetch English titles for specific patents (~19 GB).

        Uses IN-list filtering on publication_number.  Only reads the
        publication_number and title_localized columns.
        """
        in_list = ", ".join(f"'{p}'" for p in publication_numbers)
        return f"""
        SELECT
            publication_number,
            (SELECT t.text FROM UNNEST(title_localized) t
             WHERE t.language = 'en' LIMIT 1) AS title
        FROM `patents-public-data.patents.publications`
        WHERE publication_number IN ({in_list})
        """

    def _meta_query(self, publication_numbers: list[str]) -> str:
        """Return SQL that fetches CPC codes + assignees for given patents.

        Uses separate ARRAY subqueries instead of UNNEST cross-join so that
        rows with N assignees × M CPC codes aren't duplicated N×M times.
        """
        patents_in = ", ".join(f"'{p}'" for p in publication_numbers)
        return f"""
        SELECT
            publication_number,
            (SELECT STRING_AGG(a.name, ' | ') FROM UNNEST(assignee_harmonized) a) AS assignee_name,
            (SELECT STRING_AGG(c.code, ' | ') FROM UNNEST(cpc) c)                AS cpc_code
        FROM
            `patents-public-data.patents.publications`
        WHERE
            publication_number IN ({patents_in})
        """

    def _claims_query(self, publication_numbers: list[str]) -> str:
        """Return SQL that fetches English claims text.

        WARNING: Scans ~123 GB (claims_localized column) regardless of
        IN-list size.  Use with a high billing cap or skip if too expensive.
        """
        patents_in = ", ".join(f"'{p}'" for p in publication_numbers)
        return f"""
        SELECT
            publication_number,
            (SELECT STRING_AGG(c.text, ' | ')
             FROM UNNEST(claims_localized) c
             WHERE c.language = 'en') AS claims_text
        FROM
            `patents-public-data.patents.publications`
        WHERE
            publication_number IN ({patents_in})
        """

    # ------------------------------------------------------------------
    # Core query runner
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
        _log_query_cost(
            query_name=stage_label or "unknown",
            bytes_processed=bytes_processed,
            elapsed_s=elapsed,
        )
        return df

    # ------------------------------------------------------------------
    # Fetch orchestrators
    # ------------------------------------------------------------------

    def _cpc_scout(self, cpc_filter: str) -> pd.DataFrame:
        """Run CPC scout with date-fallback chain.

        Attempts:
          1. Full date range (filing_date > settings.BQ_MIN_FILING_DATE)
          2. 2010+ date window
          3. 2015+ date window
        """
        for attempt, cutoff in enumerate(
            [_DATE_CUTOFF, _DATE_CUTOFF_FB1, _DATE_CUTOFF_FB2], 1
        ):
            try:
                sql = self._cpc_scout_query(cpc_filter, date_cutoff=cutoff)
                print(f"[PatentRetriever] CPC scout attempt {attempt} (date>{cutoff})")
                df = self._run_query(sql, stage_label=f"cpc-scout-{attempt}")
                if not df.empty:
                    print(f"  Scout: {len(df)} CPC-matched patents")
                    return df
                print(f"  Scout returned 0 rows — tightening date")
            except (GoogleAPICallError, Exception) as exc:
                msg = str(exc)
                if "bytesBilledLimitExceeded" in msg:
                    print(f"[PatentRetriever] scout {attempt} HIT BILLING CAP")
                else:
                    print(f"[PatentRetriever] scout {attempt} FAILED: {exc}")
                logger.warning("Scout attempt %d failed: %s", attempt, exc)

        logger.warning("All CPC scout attempts returned 0 rows")
        return pd.DataFrame()

    def _try_fetch_claims(self, df: pd.DataFrame) -> None:
        """Attempt to fetch claims for patents in *df* (in-place).

        Uses a high billing cap (BQ_CLAIMS_BYTES_BILLED).  If the query
        exceeds the cap or fails, claims_text is left as empty string.
        """
        if df.empty:
            return
        pub_numbers = df["publication_number"].head(20).tolist()
        claims_cap = int(os.environ.get("BQ_CLAIMS_BYTES_BILLED", str(130_000_000_000)))
        claims_config = bigquery.QueryJobConfig(maximum_bytes_billed=claims_cap)
        try:
            sql = self._claims_query(pub_numbers)
            print(f"[PatentRetriever] Claims fetch for {len(pub_numbers)} patents (cap={claims_cap/1e9:.0f} GB)")
            claims_df = self._run_query(sql, stage_label="claims", job_config=claims_config)
            if not claims_df.empty:
                df.drop(columns=["claims_text"], errors="ignore", inplace=True)
                merged = df.merge(
                    claims_df[["publication_number", "claims_text"]],
                    on="publication_number",
                    how="left",
                )
                # Update df in-place
                for col in merged.columns:
                    df[col] = merged[col].values
                df["claims_text"] = df["claims_text"].fillna("").astype(str)
                print(f"  Claims fetched for {claims_df['publication_number'].nunique()} patents")
        except (GoogleAPICallError, Exception) as exc:
            logger.warning("Claims query failed (%s) — skipping", exc)
            print(f"[PatentRetriever] Claims fetch FAILED: {exc}")
            if "claims_text" not in df.columns:
                df["claims_text"] = ""

    @staticmethod
    def _parse_scout_aggregations(df: pd.DataFrame) -> None:
        """Convert CPC/assignee STRING_AGG columns to list columns in-place."""
        if "cpc_codes_agg" in df.columns:
            df["cpc_code"] = df["cpc_codes_agg"].apply(
                lambda v: [x.strip() for x in v.split(" | ")]
                if isinstance(v, str) and v else []
            )
            df.drop(columns=["cpc_codes_agg"], inplace=True)
        elif "cpc_code" not in df.columns:
            df["cpc_code"] = [[] for _ in range(len(df))]

        if "assignees_agg" in df.columns:
            df["assignee_name"] = df["assignees_agg"].apply(
                lambda v: list(set(x.strip() for x in v.split(" | ")))
                if isinstance(v, str) and v else []
            )
            df.drop(columns=["assignees_agg"], inplace=True)
        elif "assignee_name" not in df.columns:
            df["assignee_name"] = [[] for _ in range(len(df))]

    @staticmethod
    def _python_text_filter(df: pd.DataFrame, text_filter: str) -> pd.DataFrame:
        """Apply BQ-style REGEXP_CONTAINS filter on titles in Python.

        Extracts regex patterns from the BQ text_filter and checks if
        any pattern matches the ``title`` column locally.
        """
        if df.empty or not text_filter or text_filter.strip().startswith("TRUE"):
            return df

        patterns = re.findall(r"r'([^']*)'", text_filter)
        if not patterns:
            return df

        compiled = []
        for p in patterns:
            try:
                compiled.append(re.compile(p, re.IGNORECASE))
            except re.error:
                pass

        if not compiled:
            return df

        mask = df["title"].apply(
            lambda t: any(r.search(str(t)) for r in compiled) if pd.notna(t) else False
        )
        return df[mask].reset_index(drop=True)

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

            # --- Keyword density (title + abstract) ─────────────────
            text = (
                str(row.get("title", "")).lower() + " "
                + str(row.get("abstract", "")).lower()
            )
            kw_hits  = sum(1 for kw in keywords if kw in text)
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
