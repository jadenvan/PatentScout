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

# Secondary / generic terms that should NEVER drive title-only matching
# because they appear in thousands of unrelated patents.
_SECONDARY_GENERIC_TERMS = {
    "portable", "compact", "foldable", "folding", "fold", "collapsible",
    "lightweight", "usb", "battery", "charger", "charging", "power",
    "energy", "storage", "bank", "connection", "integrated", "mobile",
    "wireless", "device", "apparatus", "system", "method", "module",
    "unit", "mechanism", "assembly", "structure", "housing", "case",
    "cover", "panel", "connector", "adapter", "converter", "output",
    "input", "controller", "sensor", "detector", "display",
}


def filter_topical_relevance(df, primary_terms):
    """
    Remove patents that contain NONE of the primary technology terms in
    either title or abstract. These are guaranteed irrelevant.
    """
    if df.empty or not primary_terms:
        return df
    primary_lower = [t.lower() for t in primary_terms if len(t) >= 3]
    if not primary_lower:
        return df

    def has_primary_term(row):
        text = (
            str(row.get('title', '')).lower() + ' ' +
            str(row.get('abstract', '')).lower()
        )
        return any(term in text for term in primary_lower)

    before = len(df)
    df = df[df.apply(has_primary_term, axis=1)]
    after = len(df)
    if before != after:
        logger.info("topical filter: %d -> %d (%d removed)", before, after, before - after)
    return df.reset_index(drop=True)


def _is_primary_technology_term(term: str) -> bool:
    """Return True if term describes a core technology, not a generic feature."""
    t = term.lower().strip()
    # Split multi-word terms and check if ALL words are generic
    words = t.split()
    if all(w in _SECONDARY_GENERIC_TERMS for w in words):
        return False
    # Single-word check
    if len(words) == 1 and t in _SECONDARY_GENERIC_TERMS:
        return False
    return True



# Cost logging helpers

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
        logger.warning("could not write query cost log: %s", exc)

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
        # Title search config — needs higher budget since title_localized adds ~19 GB
        self._title_search_config = bigquery.QueryJobConfig(
            maximum_bytes_billed=max(settings.BQ_MAX_BYTES_BILLED, 50_000_000_000)
        )
        # Fallback job config — lower cap for title-only searches
        self._fallback_job_config = bigquery.QueryJobConfig(
            maximum_bytes_billed=settings.BQ_FALLBACK_BYTES_BILLED
        )

    # Public API

    def search(
        self,
        search_strategy: dict,
        user_description: str = "",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Three-tier retrieval with guaranteed topical relevance.

        Tier 1 — Title-keyword search: REGEXP_CONTAINS on title_localized
                 with terms extracted from search_strategy.
        Tier 2 — CPC-filtered search + title fetch + Python text filter.
        Tier 3 — Broadened keyword search (simplest core terms).

        Post-retrieval:
        - Abstract fetch for matched patents (IN-list, cheap)
        - Semantic re-ranking by cosine similarity to user_description
        - CPC/assignee metadata fetch
        - Claims fetch for top 20

        NO tier ever returns unfiltered patents.  If all tiers return 0
        results the method returns two empty DataFrames.

        Returns:
            ``(detail_df, landscape_df)``
        """
        primary_terms: list[str] = []
        for st_group in search_strategy.get("search_terms", []):
            p = st_group.get("primary", "").strip()
            if p:
                primary_terms.append(p)
            for syn in st_group.get("synonyms", []):
                s = syn.strip()
                if s and len(s) > 2:
                    primary_terms.append(s)
        # Deduplicate preserving order
        _seen_t: set[str] = set()
        _unique_primary: list[str] = []
        for t in primary_terms:
            tl = t.lower()
            if tl not in _seen_t:
                _seen_t.add(tl)
                _unique_primary.append(t)
        primary_terms = _unique_primary

        all_keywords: list[str] = []
        for feature in search_strategy.get("features", []):
            all_keywords.extend(feature.get("keywords", []))
        all_keywords = list(set(k.lower() for k in all_keywords if k))[:15]

        text_filter = search_strategy.get("text_filter", "TRUE")
        predicted_cpc = [
            c.get("code", "") for c in search_strategy.get("cpc_codes", [])
        ]
        cpc_prefixes = self._extract_cpc_prefixes(search_strategy)
        cpc_filter = self._build_cpc_exists_clause(cpc_prefixes)

        def _terms_to_bq_regex(terms: list[str]) -> str:
            cleaned: list[str] = []
            for t in terms:
                t = t.lower().strip()
                if len(t) < 3:
                    continue
                t = re.escape(t).replace(r"\ ", r"\s+")
                cleaned.append(t)
            if not cleaned:
                return ""
            return "(" + "|".join(cleaned) + ")"

        # Title regex: ONLY primary technology terms (not generic features).
        # Generic terms like "fold", "portable", "battery" match thousands
        # of unrelated patents and must be excluded from title-only matching.
        title_terms_raw = [
            g.get("primary", "")
            for g in search_strategy.get("search_terms", [])
            if g.get("primary")
        ]
        if not title_terms_raw:
            title_terms_raw = primary_terms[:5]

        # Filter to primary technology terms only
        title_terms = [
            t for t in title_terms_raw if _is_primary_technology_term(t)
        ]
        if not title_terms:
            # If filtering removed everything, keep the longest term
            # (most likely to be domain-specific)
            title_terms_raw.sort(key=len, reverse=True)
            title_terms = title_terms_raw[:2]

        title_regex = _terms_to_bq_regex(title_terms)

        # Validate regex
        if title_regex:
            try:
                re.compile(title_regex)
            except re.error:
                title_regex = _terms_to_bq_regex(primary_terms[:3])


        # Save for diagnostics
        try:
            import streamlit as st
            st.session_state["last_title_regex"] = title_regex
            st.session_state["last_text_filter"] = text_filter
            st.session_state["last_cpc_filter"] = cpc_filter
        except Exception:
            pass

        detail_df = pd.DataFrame()

        # TIER 1 — title-keyword search via REGEXP_CONTAINS on title_localized
        if title_regex:
            tier1_sql = f"""
            SELECT DISTINCT
                publication_number,
                t.text AS title,
                filing_date,
                grant_date,
                publication_date
            FROM
                `patents-public-data.patents.publications`,
                UNNEST(title_localized) AS t
            WHERE
                country_code = 'US'
                AND filing_date > 20050101
                AND grant_date > 0
                AND t.language = 'en'
                AND REGEXP_CONTAINS(LOWER(t.text), r'{title_regex}')
            ORDER BY filing_date DESC
            LIMIT 300
            """
            try:
                detail_df = self._run_query(
                    tier1_sql,
                    stage_label="tier1-title-keyword",
                    job_config=self._title_search_config,
                )
                if not detail_df.empty:
                    for _, row in detail_df.head(5).iterrows():
                        logger.debug("tier1 sample: %s", row.get("title", "")[:60])
            except (GoogleAPICallError, Exception) as exc:
                logger.warning("tier 1 title search failed: %s", exc)
                detail_df = pd.DataFrame()

        # TIER 2 — CPC-filtered scout + title fetch + Python text filter
        if len(detail_df) < 10 and cpc_filter:
            logger.info(
                "tier 2: cpc-filtered search (tier 1 gave %d)", len(detail_df)
            )
            scout_df = self._cpc_scout(cpc_filter)
            if not scout_df.empty:
                # Fetch titles for CPC-matched patents
                pub_nums = scout_df["publication_number"].tolist()
                try:
                    title_df = self._run_query(
                        self._title_query(pub_nums),
                        stage_label="tier2-title-fetch",
                    )
                    if not title_df.empty:
                        scout_df = scout_df.merge(
                            title_df, on="publication_number", how="left"
                        )
                except Exception as exc:
                    logger.warning("tier 2 title fetch failed: %s", exc)
                if "title" not in scout_df.columns:
                    scout_df["title"] = ""
                self._parse_scout_aggregations(scout_df)

                # Python-side text filter on titles
                filtered = self._python_text_filter(scout_df, text_filter)
                if len(filtered) >= 5:
                    scout_df = filtered

                # Merge with Tier 1 results
                if not detail_df.empty:
                    detail_df = pd.concat(
                        [detail_df, scout_df], ignore_index=True,
                    ).drop_duplicates(
                        subset=["publication_number"], keep="first",
                    ).reset_index(drop=True)
                else:
                    detail_df = scout_df

        # TIER 3 — broadened keyword search on simplest core terms
        if len(detail_df) < 10 and primary_terms:
            logger.info(
                "tier 3: broadened keyword search (found %d so far)", len(detail_df)
            )
            simple_terms = primary_terms[:3]
            simple_regex = "|".join(
                re.escape(t.lower()) for t in simple_terms if t
            )
            if simple_regex:
                tier3_sql = f"""
                SELECT DISTINCT
                    publication_number,
                    t.text AS title,
                    filing_date, grant_date, publication_date
                FROM
                    `patents-public-data.patents.publications`,
                    UNNEST(title_localized) AS t
                WHERE
                    country_code = 'US'
                    AND filing_date > 20000101
                    AND grant_date > 0
                    AND t.language = 'en'
                    AND REGEXP_CONTAINS(
                        LOWER(t.text), r'({simple_regex})'
                    )
                ORDER BY filing_date DESC
                LIMIT 200
                """
                try:
                    tier3_df = self._run_query(
                        tier3_sql,
                        stage_label="tier3-broadened",
                        job_config=self._title_search_config,
                    )
                    detail_df = pd.concat(
                        [detail_df, tier3_df], ignore_index=True,
                    ).drop_duplicates(
                        subset=["publication_number"], keep="first",
                    ).reset_index(drop=True)
                except Exception as exc:
                    logger.warning("tier 3 broadened search failed: %s", exc)

        # Early exit if all tiers returned nothing
        if detail_df.empty:
            empty = pd.DataFrame()
            return empty, empty

        # Dedup
        before_dedup = len(detail_df)
        detail_df = detail_df.drop_duplicates(
            subset=["publication_number"], keep="first",
        ).reset_index(drop=True)
        if before_dedup != len(detail_df):
            logger.info("dedup: %d -> %d", before_dedup, len(detail_df))

        # Remove design / plant patents (US-D*, US-PP*)
        before_dp_filter = len(detail_df)
        detail_df = detail_df[
            ~detail_df['publication_number'].str.match(
                r'US-D|US-PP', na=False
            )
        ].reset_index(drop=True)
        if before_dp_filter != len(detail_df):
            logger.info(
                "design/plant filter: %d -> %d",
                before_dp_filter, len(detail_df),
            )

        # Abstract fetch — abstract_localized column is ~202 GB; needs high cap
        _abs_cap = int(os.environ.get(
            "BQ_ABSTRACT_BYTES_BILLED", str(210_000_000_000),
        ))
        _abs_config = bigquery.QueryJobConfig(maximum_bytes_billed=_abs_cap)
        if "abstract" not in detail_df.columns:
            detail_df["abstract"] = ""
        abstract_pubs = detail_df["publication_number"].tolist()[:200]
        if abstract_pubs:
            logger.info(
                "fetching abstracts for %d patents (cap=%d GB)",
                len(abstract_pubs), int(_abs_cap / 1e9),
            )
            try:
                abstract_df = self._run_query(
                    self._abstract_query(abstract_pubs),
                    stage_label="abstract-fetch",
                    job_config=_abs_config,
                )
                if not abstract_df.empty:
                    # Drop existing empty abstract column before merge
                    detail_df.drop(columns=["abstract"], errors="ignore", inplace=True)
                    detail_df = detail_df.merge(
                        abstract_df, on="publication_number", how="left",
                    )
                    n_abs = detail_df["abstract"].notna().sum()
                    logger.info("abstracts fetched: %d/%d", n_abs, len(detail_df))
            except Exception as exc:
                logger.warning("abstract fetch failed: %s", exc)
        if "abstract" not in detail_df.columns:
            detail_df["abstract"] = ""
        detail_df["abstract"] = detail_df["abstract"].fillna("")

        # Metadata fetch (CPC codes + assignee names)
        needs_meta = (
            "cpc_code" not in detail_df.columns
            or "assignee_name" not in detail_df.columns
        )
        if needs_meta:
            meta_pubs = detail_df["publication_number"].tolist()[:200]
            if meta_pubs:
                try:
                    meta_df = self._run_query(
                        self._meta_query(meta_pubs),
                        stage_label="metadata-fetch",
                    )
                    if not meta_df.empty:
                        # Parse pipe-separated strings into lists
                        if "cpc_code" in meta_df.columns:
                            meta_df["cpc_code"] = meta_df["cpc_code"].apply(
                                lambda v: [x.strip() for x in v.split(" | ")]
                                if isinstance(v, str) and v else []
                            )
                        if "assignee_name" in meta_df.columns:
                            meta_df["assignee_name"] = meta_df["assignee_name"].apply(
                                lambda v: list(set(x.strip() for x in v.split(" | ")))
                                if isinstance(v, str) and v else []
                            )
                        detail_df.drop(
                            columns=["cpc_code", "assignee_name"],
                            errors="ignore", inplace=True,
                        )
                        detail_df = detail_df.merge(
                            meta_df, on="publication_number", how="left",
                        )
                except Exception as exc:
                    logger.warning("metadata fetch failed: %s", exc)

        if "cpc_code" not in detail_df.columns:
            detail_df["cpc_code"] = [[] for _ in range(len(detail_df))]
        if "assignee_name" not in detail_df.columns:
            detail_df["assignee_name"] = [[] for _ in range(len(detail_df))]

        # Remove patents with ZERO primary technology terms in title+abstract.
        # This catches noise from generic terms like "folding mechanism".
        _primary_tech_terms = [
            t for t in primary_terms if _is_primary_technology_term(t)
        ]
        if _primary_tech_terms:
            detail_df = filter_topical_relevance(detail_df, _primary_tech_terms)

        if user_description:
            detail_df = self._semantic_rerank(detail_df, user_description)
        else:
            self._score_relevance(detail_df, predicted_cpc, search_strategy)

        # Build landscape from the same re-ranked data (saves BQ cost).
        # Filter to patents sharing CPC class prefixes with the predicted codes.
        _predicted_cpc_classes: set[str] = set()
        for _cpc_entry in search_strategy.get("cpc_codes", []):
            _code = _cpc_entry.get("code", "").strip()
            if _code and len(_code) >= 3:
                _predicted_cpc_classes.add(_code[:3].upper())

        if (
            _predicted_cpc_classes
            and "cpc_code" in detail_df.columns
            and len(detail_df) > 20
        ):
            def _has_relevant_cpc(row):
                cpc_list = row.get("cpc_code", [])
                if not isinstance(cpc_list, list):
                    return False
                for c in cpc_list:
                    if isinstance(c, str):
                        for prefix in _predicted_cpc_classes:
                            if c.upper().startswith(prefix):
                                return True
                return False

            _cpc_mask = detail_df.apply(_has_relevant_cpc, axis=1)
            _cpc_filtered = detail_df[_cpc_mask]
            if len(_cpc_filtered) >= 5:
                landscape_df = _cpc_filtered.copy().reset_index(drop=True)
                logger.info(
                    "landscape: %d patents (from %d total, CPC classes: %s)",
                    len(landscape_df), len(detail_df), _predicted_cpc_classes,
                )
            else:
                landscape_df = detail_df.head(min(len(detail_df), 30)).copy()
                logger.info(
                    "landscape: cpc filter too aggressive (%d matched %s), using top 30",
                    len(_cpc_filtered), _predicted_cpc_classes,
                )
        else:
            landscape_df = detail_df.copy()

        if not landscape_df.empty and 'assignee_name' in landscape_df.columns:
            _ls_assignees = landscape_df[
                landscape_df['assignee_name'].apply(
                    lambda x: isinstance(x, list) and len(x) > 0
                )
            ].explode('assignee_name')
            if not _ls_assignees.empty:
                _top_asgn = _ls_assignees.groupby('assignee_name').size().nlargest(5)
                logger.debug("top assignees: %s", _top_asgn.to_dict())
        if not landscape_df.empty and 'cpc_code' in landscape_df.columns:
            _ls_cpc = landscape_df[
                landscape_df['cpc_code'].apply(
                    lambda x: isinstance(x, list) and len(x) > 0
                )
            ].explode('cpc_code')
            if not _ls_cpc.empty:
                _top_cpc = _ls_cpc['cpc_code'].apply(
                    lambda x: str(x)[:4] if pd.notna(x) else None
                ).dropna().value_counts().head(5)
                logger.debug("top CPC groups: %s", _top_cpc.to_dict())

        detail_limit = settings.BQ_QUERY_LIMIT_DETAIL
        detail_out = detail_df.head(detail_limit).copy()

        for df in (detail_out, landscape_df):
            if "claims_text" not in df.columns:
                df["claims_text"] = ""

        self._add_computed_columns(detail_out)
        self._add_computed_columns(landscape_df)

        if not detail_out.empty:
            self._try_fetch_claims(detail_out)

        if "relevance_score" in detail_out.columns:
            for _, row in detail_out.head(10).iterrows():
                score = row.get("relevance_score", 0)
                title = row.get("title", "N/A")[:70]
                logger.debug("top result: %.3f  %s", score, title)

        # Relevance sanity check
        frac = self._check_relevance(detail_out)
        logger.info(
            "PatentRetriever: detail=%d rows  landscape=%d rows  relevance=%.2f",
            len(detail_out), len(landscape_df), frac,
        )
        return detail_out, landscape_df


    # CPC helpers

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

    # Query builders

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

    def _abstract_query(self, publication_numbers: list[str]) -> str:
        """Return SQL that fetches English abstracts for specific patents.

        Uses IN-list filtering on publication_number so only matching
        rows are read — much cheaper than a full-table scan of
        abstract_localized (~202 GB).
        """
        patents_in = ", ".join(f"'{p}'" for p in publication_numbers)
        return f"""
        SELECT
            publication_number,
            (SELECT a.text FROM UNNEST(abstract_localized) a
             WHERE a.language = 'en' LIMIT 1) AS abstract
        FROM
            `patents-public-data.patents.publications`
        WHERE
            publication_number IN ({patents_in})
        """

    # Core query runner

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
        _log_query_cost(
            query_name=stage_label or "unknown",
            bytes_processed=bytes_processed,
            elapsed_s=elapsed,
        )
        return df

    # Fetch orchestrators

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
                df = self._run_query(sql, stage_label=f"cpc-scout-{attempt}")
                if not df.empty:
                    return df
            except (GoogleAPICallError, Exception) as exc:
                msg = str(exc)
                if "bytesBilledLimitExceeded" in msg:
                    logger.warning("scout attempt %d: bytes billed limit exceeded", attempt)
                else:
                    logger.warning("Scout attempt %d failed: %s", attempt, exc)

        logger.warning("all CPC scout attempts returned 0 rows")
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
        except (GoogleAPICallError, Exception) as exc:
            logger.warning("claims query failed (%s) — skipping", exc)
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

    # Post-processing

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

    # Utilities

    def rerank_by_relevance(
        self,
        df: pd.DataFrame,
        user_description: str,
        text_column: str = "abstract",
        top_n: int = 0,
    ) -> pd.DataFrame:
        """
        Re-rank retrieved patents by semantic similarity to the user's
        invention description using sentence-transformers embeddings.

        This is the winning strategy (Strategy G) from retrieval experiments,
        which boosted Top-10 precision from 10% to 60%.

        Parameters
        ----------
        df : pd.DataFrame
            Patents to re-rank (must have *text_column*).
        user_description : str
            The user's invention description text.
        text_column : str
            Column to use for similarity comparison ('abstract' or 'title').
        top_n : int
            If > 0, return only the top N most relevant patents.

        Returns
        -------
        pd.DataFrame
            Sorted by ``semantic_relevance`` descending, with the new column
            added in-place.
        """
        if df.empty or not user_description:
            return df

        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            logger.warning("sentence-transformers not installed — skipping re-rank")
            return df

        # Pick best available text column
        if text_column not in df.columns or df[text_column].isna().all():
            for fallback_col in ("abstract", "title"):
                if fallback_col in df.columns and df[fallback_col].notna().any():
                    text_column = fallback_col
                    break
            else:
                return df

        work_df = df[df[text_column].notna() & (df[text_column] != "")].copy()
        if work_df.empty:
            return df

        model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)

        user_emb = model.encode([user_description])
        text_embs = model.encode(work_df[text_column].tolist(), show_progress_bar=False)

        scores = cosine_similarity(user_emb, text_embs)[0]
        work_df["semantic_relevance"] = scores
        work_df = work_df.sort_values("semantic_relevance", ascending=False)

        if top_n > 0:
            work_df = work_df.head(top_n)

        logger.info(
            "Re-ranked %d patents by semantic relevance (top score: %.3f)",
            len(work_df),
            scores.max() if len(scores) > 0 else 0,
        )
        return work_df.reset_index(drop=True)

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

    # Semantic re-ranking

    def _semantic_rerank(
        self,
        df: pd.DataFrame,
        user_description: str,
    ) -> pd.DataFrame:
        """
        Re-rank retrieved patents by semantic similarity to the user
        description using sentence-transformers embeddings on
        title + abstract.

        Updates ``relevance_score`` column in-place and returns the
        sorted DataFrame.
        """
        if df.empty or not user_description:
            return df

        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        except ImportError:
            logger.warning("sentence-transformers not installed — skipping semantic rerank")
            return df

        # Combine title + abstract for encoding
        texts: list[str] = []
        valid_indices: list[int] = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            title = str(row.get("title", "") or "")
            abstract = str(row.get("abstract", "") or "")
            combined = f"{title}. {abstract}".strip()
            if len(combined) > 10:
                texts.append(combined)
                valid_indices.append(idx)

        if not texts:
            df["relevance_score"] = 0.0
            return df

        model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        user_emb = model.encode([user_description], show_progress_bar=False)
        text_embs = model.encode(texts, show_progress_bar=False, batch_size=32)
        scores = cos_sim(user_emb, text_embs)[0]

        df["relevance_score"] = 0.0
        for i, idx in enumerate(valid_indices):
            df.iloc[idx, df.columns.get_loc("relevance_score")] = round(
                float(scores[i]), 4
            )

        df = df.sort_values("relevance_score", ascending=False).reset_index(
            drop=True
        )

        for _, row in df.head(5).iterrows():
            logger.debug(
                "rerank: %.3f  %s",
                row.get("relevance_score", 0),
                row.get("title", "")[:60],
            )

        return df



# Module-level helper

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
