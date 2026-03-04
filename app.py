"""
PatentScout - MVP Streamlit Application
Entry point for the PatentScout patent analysis tool.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from google.cloud import bigquery

from modules.claim_parser import ClaimParser
from modules.element_mapper import ElementMapper
from modules.embedding_engine import EmbeddingEngine
from modules.input_handler import validate_input
from modules.landscape_analyzer import LandscapeAnalyzer
from modules.patent_retriever import PatentRetriever
from modules.query_builder import QueryBuilder
from modules.report_generator import ReportGenerator
from modules.whitespace_finder import WhiteSpaceFinder

load_dotenv()

logger = logging.getLogger(__name__)


# Credential helpers — support both local .env and Streamlit Cloud secrets


def _get_gemini_api_key() -> str:
    """Return Gemini API key from env or Streamlit secrets."""
    key = os.getenv("GEMINI_API_KEY", "")
    if key:
        return key
    try:
        return st.secrets.get("general", {}).get("GEMINI_API_KEY", "") or \
               st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""


@st.cache_resource(show_spinner="Connecting to BigQuery...")
def _get_bigquery_client() -> bigquery.Client:
    """
    Build and cache a BigQuery client.

    Resolution order:
      1. GOOGLE_APPLICATION_CREDENTIALS env var pointing to a local JSON file
      2. credentials/service-account.json (local dev convenience)
      3. ``gcp_service_account`` block in Streamlit secrets (Cloud deployment)
    """
    from config import settings

    # 1. Explicit env var
    local_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    if local_creds and os.path.exists(local_creds):
        return bigquery.Client(project=settings.BIGQUERY_PROJECT)

    # 2. Fallback local JSON
    fallback = os.path.join(os.path.dirname(__file__), "credentials", "service-account.json")
    if os.path.exists(fallback):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = fallback
        return bigquery.Client(project=settings.BIGQUERY_PROJECT)

    # 3. Streamlit secrets
    try:
        secret_block = dict(st.secrets["gcp_service_account"])
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_info(
            secret_block,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        project = secret_block.get("project_id", settings.BIGQUERY_PROJECT)
        return bigquery.Client(credentials=creds, project=project)
    except (KeyError, Exception):
        pass

    raise ValueError(
        "No Google Cloud credentials found. "
        "Set GOOGLE_APPLICATION_CREDENTIALS or configure gcp_service_account in "
        "Streamlit secrets."
    )


@st.cache_resource(show_spinner="Loading embedding model...")
def _get_embedding_engine() -> "EmbeddingEngine":
    """Load and cache the sentence-transformers model (expensive, load once)."""
    return EmbeddingEngine()


# Page configuration — must be the very first Streamlit call

st.set_page_config(
    page_title="PatentScout",
    page_icon=":mag:",
    layout="wide",
)


# Session-state initialisation — runs before any widget is rendered


def _init_session_state() -> None:
    """Initialise all session-state keys with safe defaults."""
    defaults: dict = {
        "invention_text": "",
        "invention_image": None,   # raw bytes of the uploaded image
        "sketch_used": False,        # True when doorbell demo with sketch is loaded
        "is_demo": False,            # True when a demo session is loaded
        "analysis_timestamp": "",   # datetime string when analysis completed
        "analysis_complete": False,
        "prior_art_results": None,
        "claim_analysis": None,
        "landscape_data": None,
        "report_data": None,
        "search_strategy": None,
        "detail_patents": None,    # pd.DataFrame from PatentRetriever
        "landscape_patents": None, # pd.DataFrame from PatentRetriever
        "parsed_claims": None,     # dict from ClaimParser.parse_all()
        "similarity_results": None,   # dict from EmbeddingEngine
        "comparison_matrix": None,     # list[dict] from ElementMapper
        "landscape_figures": None,    # dict of Plotly figures
        "chart_images": {},           # dict of PNG bytes for PDF export
        "white_spaces": None,         # list[dict] from WhiteSpaceFinder
        "_pdf_bytes": None,            # bytes from ReportGenerator.generate()
        # Cost monitoring
        "query_costs": [],             # list of per-query cost records
        "total_gb_scanned": 0.0,       # cumulative GB scanned this session
        # Query result cache (key → {"detail_df": ..., "landscape_df": ...})
        "query_cache": {},
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


_init_session_state()


# Global CSS — sticky card + visual refinements

st.markdown("""
<style>
/* Fix 1: Sticky Analyzed Invention card */
[data-testid="stVerticalBlockBorderWrapper"]:has(.analyzed-card-marker) {
    position: sticky !important;
    top: 0px;
    z-index: 50;
    background: white;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
/* Fix 3: Visual hierarchy refinements */
.patentscout-header h1 {
    font-size: 34px !important;
    font-weight: 700 !important;
    margin-bottom: 8px !important;
}
.patentscout-subheader {
    font-size: 16px;
    font-weight: 600;
    max-width: 720px;
    margin-bottom: 32px;
}
.how-it-works-step h4 {
    margin-bottom: 4px !important;
}
</style>
""", unsafe_allow_html=True)


# Query cache helpers

import hashlib

def get_cache_key(search_strategy: dict) -> str:
    """Build a stable cache key from the search_terms primary values."""
    terms = search_strategy.get("search_terms", [])
    key_parts = sorted(t.get("primary", "") for t in terms)
    raw = json.dumps(key_parts, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]



# Demo session helpers  (use shared demo_data module for PDF consistency)

from demo_data import build_solar_demo_data, build_doorbell_demo_data


def _populate_session_from_data(data: dict) -> None:
    """Populate all session-state keys from a demo data dict.

    This ensures that app tabs see the EXACT same data that the PDF
    report generator uses — no more inconsistency.
    """
    import pandas as pd

    from datetime import datetime
    st.session_state["invention_text"]     = data.get("invention_text", "")
    st.session_state["invention_image"]    = data.get("invention_image")
    st.session_state["sketch_used"]        = data.get("sketch_used", False)
    st.session_state["is_demo"]            = True
    st.session_state["analysis_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.session_state["search_strategy"]    = data.get("search_strategy")
    st.session_state["similarity_results"] = data.get("similarity_results")
    st.session_state["comparison_matrix"]  = data.get("comparison_matrix")
    st.session_state["white_spaces"]       = data.get("white_spaces")
    st.session_state["query_costs"]        = data.get("query_costs", [])
    st.session_state["total_gb_scanned"]   = data.get("total_gb_scanned", 0.0)
    st.session_state["_pdf_bytes"]         = None

    # detail_patents — convert list→DataFrame
    raw_detail = data.get("detail_patents")
    if isinstance(raw_detail, list):
        st.session_state["detail_patents"] = pd.DataFrame(raw_detail)
    elif isinstance(raw_detail, pd.DataFrame):
        st.session_state["detail_patents"] = raw_detail
    else:
        st.session_state["detail_patents"] = None

    # landscape_patents — convert list→DataFrame and build figures
    raw_landscape = data.get("landscape_patents")
    if isinstance(raw_landscape, list):
        landscape_df = pd.DataFrame(raw_landscape)
    elif isinstance(raw_landscape, pd.DataFrame):
        landscape_df = raw_landscape
    else:
        landscape_df = None
    st.session_state["landscape_patents"] = landscape_df

    if landscape_df is not None and not landscape_df.empty:
        try:
            _analyzer = LandscapeAnalyzer(landscape_df)
            _figs = {
                "filing_trends":    _analyzer.filing_trends(),
                "top_assignees":    _analyzer.top_assignees(),
                "cpc_distribution": _analyzer.cpc_distribution(),
            }
            st.session_state["landscape_figures"] = _figs
            st.session_state["chart_images"] = (
                _analyzer.export_charts_as_images(_figs)
            )
        except Exception as _la_exc:
            logger.warning("LandscapeAnalyzer failed during demo load: %s", _la_exc)
            st.session_state["landscape_figures"] = {}
            st.session_state["chart_images"] = {}
    else:
        st.session_state["landscape_figures"] = None
        st.session_state["chart_images"] = {}

    # parsed_claims — normalise list form to {"summary": ..., "results": ...} dict
    raw_claims = data.get("parsed_claims")
    if isinstance(raw_claims, list):
        st.session_state["parsed_claims"] = {
            "summary": {
                "attempted": len(raw_claims),
                "successful": len(raw_claims),
                "skipped": 0,
                "failed": 0,
            },
            "results": raw_claims,
        }
    else:
        st.session_state["parsed_claims"] = raw_claims

    st.session_state["analysis_complete"] = True


def _load_solar_demo() -> bool:
    """Build and load the solar charger demo into session state."""
    try:
        data = build_solar_demo_data()
        _populate_session_from_data(data)
        return True
    except Exception as exc:
        logger.warning("Could not build solar demo data: %s", exc)
        return False


def _load_doorbell_demo() -> bool:
    """Build and load the smart doorbell demo (with sketch) into session state."""
    try:
        data = build_doorbell_demo_data()
        _populate_session_from_data(data)
        if not data.get("sketch_used"):
            st.warning(
                "Sketch file not found at assets/demo/doorbell_sketch.* — "
                "loading text-only demo"
            )
        return True
    except Exception as exc:
        logger.warning("Could not build doorbell demo data: %s", exc)
        return False



# Helper: highlight overlapping terms between two texts


def highlight_overlapping_terms(text: str, reference_text: str, min_word_length: int = 4) -> str:
    """Highlight words in *text* that also appear in *reference_text*."""
    ref_words = set(
        w.lower() for w in re.findall(r'\b\w+\b', reference_text)
        if len(w) >= min_word_length
    )
    stopwords = {
        "that", "this", "with", "from", "have", "been", "were", "will",
        "said", "each", "which", "their", "than", "into", "more", "also",
        "configured", "wherein", "thereof", "therein", "herein",
    }
    ref_words -= stopwords

    def _replace(m):
        word = m.group()
        if word.lower() in ref_words:
            return (
                f'<span style="background-color: #FFEB3B; padding: 1px 3px; '
                f'border-radius: 2px; font-weight: bold;">{word}</span>'
            )
        return word

    return re.sub(r'\b\w+\b', _replace, text)



# Helper: Executive Summary tab renderer


def _render_executive_summary() -> None:
    """Render the Executive Summary dashboard tab."""
    strategy = st.session_state.get("search_strategy") or {}
    features = strategy.get("features", [])
    detail_df = st.session_state.get("detail_patents")
    sim = st.session_state.get("similarity_results") or {}
    white_spaces = st.session_state.get("white_spaces") or []
    stats = sim.get("stats", {})
    matches = sim.get("matches", [])

    total_detail_patents = len(detail_df) if detail_df is not None else 0
    high_count = stats.get("high_matches", 0)
    feature_count = len(features)
    ws_count = len(white_spaces)

    # Row 1 — Metric cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patents Analyzed", total_detail_patents)
    c2.metric("High Matches", high_count)
    c3.metric("Features Mapped", feature_count)
    c4.metric("White Space Opportunities", ws_count)

    st.divider()

    # Row 2 — Overall risk assessment
    if feature_count > 0 and matches:
        features_with_high = set()
        for m in matches:
            if m.get("similarity_level") in ("HIGH", "MODERATE"):
                # count features that have at least one high/mod match
                features_with_high.add(m.get("feature_label", ""))
        risk_n = len(features_with_high)
        risk_pct = risk_n / feature_count if feature_count else 0
        if risk_pct >= 0.5:
            st.error(
                f"\U0001f534 **HIGH RISK** — Significant prior art overlap "
                f"found in {risk_n} of {feature_count} features"
            )
        elif risk_pct >= 0.25:
            st.warning(
                f"\U0001f7e1 **MODERATE RISK** — Some prior art overlap "
                f"detected in {risk_n} of {feature_count} features"
            )
        else:
            st.success(
                "\U0001f7e2 **LOW RISK** — Limited prior art overlap found"
            )
    else:
        st.info("Insufficient data to compute risk assessment.")

    st.divider()

    # Row 3 — Feature coverage table
    if features and total_detail_patents > 0:
        st.subheader("Feature Prior Art Exposure")
        feature_rows = []
        for feat in features:
            label = feat.get("label", "")
            matching_patents: set[str] = set()
            top_score = 0.0
            for m in matches:
                if (
                    m.get("feature_label") == label
                    and m.get("similarity_level") in ("HIGH", "MODERATE")
                ):
                    matching_patents.add(m.get("patent_number", ""))
                    top_score = max(top_score, m.get("similarity_score", 0))
            n_match = len(matching_patents)
            coverage_pct = n_match / total_detail_patents * 100
            if coverage_pct >= 60:
                risk_label = "\U0001f534 HIGH"
                action = "Design-around needed"
            elif coverage_pct >= 30:
                risk_label = "\U0001f7e1 MED"
                action = "Further investigation"
            else:
                risk_label = "\U0001f7e2 LOW"
                action = "Potential opportunity"
            feature_rows.append({
                "Feature": label,
                "Coverage": f"{coverage_pct:.0f}% ({n_match}/{total_detail_patents})",
                "Top Score": f"{top_score:.3f}" if top_score > 0 else "—",
                "Risk": risk_label,
                "Suggested Action": action,
                "coverage_pct": coverage_pct,
                "label": label,
            })

        import pandas as _pd_es
        display_rows = [
            {k: v for k, v in r.items() if k not in ("coverage_pct", "label")}
            for r in feature_rows
        ]
        st.dataframe(
            _pd_es.DataFrame(display_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        feature_rows = []

    st.divider()

    # Row 4 — Top 3 closest patents
    if matches:
        st.subheader("Closest Prior Art")
        # Build patent title lookup
        _title_map: dict[str, str] = {}
        if detail_df is not None:
            for _, row in detail_df.iterrows():
                pn = row.get("publication_number", "")
                t = row.get("title", "")
                if pn and t:
                    _title_map[str(pn)] = str(t)

        seen_patents: set[str] = set()
        top3: list[dict] = []
        for m in sorted(matches, key=lambda x: x.get("similarity_score", 0), reverse=True):
            pn = m.get("patent_number", "")
            if pn not in seen_patents:
                seen_patents.add(pn)
                top3.append(m)
            if len(top3) >= 3:
                break

        for i, m in enumerate(top3, 1):
            pat_title = _title_map.get(m.get("patent_number", ""), "")
            title_suffix = f" — {pat_title}" if pat_title else ""
            st.markdown(
                f"**{i}. {m['patent_number']}** "
                f"(Score: {m['similarity_score']:.3f}){title_suffix}"
            )

    st.divider()

    # Row 5 — Key recommendation
    if feature_rows:
        st.subheader("Key Recommendation")
        low_coverage = [f for f in feature_rows if f["coverage_pct"] < 30]
        high_risk = [f for f in feature_rows if f["coverage_pct"] >= 60]

        if low_coverage:
            opportunity_names = ", ".join(f["label"] for f in low_coverage[:2])
            st.info(
                f"Strongest differentiation opportunity lies in "
                f"**{opportunity_names}**, which show limited prior art "
                f"coverage. Consider emphasizing these in patent claims."
            )
        if high_risk:
            risk_names = ", ".join(f["label"] for f in high_risk[:2])
            st.warning(
                f"**{risk_names}** have significant prior art overlap. "
                f"Design-around strategies should be explored before filing."
            )
        if not low_coverage and not high_risk:
            st.info(
                "All features show moderate prior art presence. "
                "A detailed freedom-to-operate analysis is recommended."
            )

    st.divider()

    # Row 6 — Download button
    if st.session_state.get("_pdf_bytes"):
        st.download_button(
            "\U0001f4e5 Download Full Report",
            data=st.session_state["_pdf_bytes"],
            file_name="patentscout_report.pdf",
            mime="application/pdf",
            key="exec_summary_download_btn",
        )
    else:
        _retrieval_done = st.session_state.get("detail_patents") is not None
        if _retrieval_done:
            if st.button("\U0001f4c4 Generate Report", key="exec_summary_gen_btn"):
                with st.spinner("Generating PDF..."):
                    try:
                        _rg = ReportGenerator()
                        _pdf_bytes = _rg.generate(dict(st.session_state))
                        st.session_state["_pdf_bytes"] = _pdf_bytes
                        st.rerun()
                    except Exception as _rg_exc:
                        st.error(f"Report generation failed: {_rg_exc}")



# Helper: demo progress bar


def _show_demo_progress() -> None:
    """Show a brief staged progress bar when loading demo data."""
    stages = [
        "\U0001f50d Loading extracted features...",
        "\U0001f5c4\ufe0f Loading cached patent data...",
        "\U0001f4cb Loading parsed claims...",
        "\U0001f9ee Loading similarity results...",
        "\U0001f916 Loading AI analysis...",
        "\U0001f4ca Loading landscape data...",
        "\u2705 Demo loaded!",
    ]
    progress_bar = st.progress(0)
    for i, stage in enumerate(stages):
        progress_bar.progress((i + 1) / len(stages), text=stage)
        time.sleep(0.15)
    progress_bar.empty()



# Helper: results renderer  (defined BEFORE any code that calls it)


def _render_analyzed_invention_card() -> None:
    """Render a persistent card above tabs showing what was analyzed."""
    with st.container(border=True):
        st.markdown('<div class="analyzed-card-marker"></div>', unsafe_allow_html=True)
        st.markdown("**Analyzed Invention**")

        # Show description (truncated)
        desc = st.session_state.get("invention_text", "")
        if desc:
            truncated = desc[:300] + "..." if len(desc) > 300 else desc
            st.markdown(f'"{truncated}"')
            if len(desc) > 300:
                with st.expander("Show full description"):
                    st.markdown(desc)

        # Show sketch thumbnail + expander if present
        if st.session_state.get("invention_image"):
            col_thumb, col_expand = st.columns([1, 3])
            with col_thumb:
                st.image(st.session_state["invention_image"], width=120)
            with col_expand:
                with st.expander("View full-size sketch"):
                    st.image(st.session_state["invention_image"], use_container_width=True)

        # Metadata row
        has_text = bool(st.session_state.get("invention_text"))
        has_sketch = bool(st.session_state.get("invention_image"))
        if has_text and has_sketch:
            sub_type = "Text + Sketch"
        elif has_sketch:
            sub_type = "Sketch"
        else:
            sub_type = "Text"

        source = "Demo" if st.session_state.get("is_demo") else "User"
        feature_count = len(st.session_state.get("search_strategy", {}).get("features", []))

        meta_cols = st.columns(4)
        meta_cols[0].caption(f"Submission: {sub_type}")
        meta_cols[1].caption(f"Source: {source}")
        meta_cols[2].caption(f"{feature_count} features extracted")
        meta_cols[3].caption(f"{st.session_state.get('analysis_timestamp', '')}")

        # Demo badge
        if source == "Demo":
            st.caption("Demo Example")


def _render_extracted_features_tab() -> None:
    """Render the Extracted Features tab content."""
    st.header("Extracted Features")
    st.caption("AI-identified technical features from your invention description")

    # Show full-size sketch at top of features tab when sketch was used
    if st.session_state.get("sketch_used") and st.session_state.get("invention_image"):
        st.subheader("Design Sketch Input")
        st.image(st.session_state["invention_image"], use_container_width=True)
        st.caption("Features enhanced by sketch analysis are marked with a badge")
        st.divider()

    features = st.session_state.get("search_strategy", {}).get("features", [])
    similarity_results = st.session_state.get("similarity_results", {}) or {}
    matches = similarity_results.get("matches", [])

    for feature in features:
        source = feature.get("source", "text")

        with st.container(border=True):
            # Header row
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"### {feature['label']}")
            with col2:
                if source == "sketch":
                    st.markdown(
                        '<span style="background-color:#E3F2FD;padding:2px 8px;'
                        'border-radius:10px;font-size:0.8em;">Sketch</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<span style="background-color:#F5F5F5;padding:2px 8px;'
                        'border-radius:10px;font-size:0.8em;">Text</span>',
                        unsafe_allow_html=True,
                    )

            # Description
            st.markdown(feature.get("description", ""))

            # Keywords
            keywords = feature.get("keywords", [])
            if keywords:
                kw_str = " \u00b7 ".join(keywords[:6])
                st.caption(f"Keywords: {kw_str}")

            # Matching patents (expandable)
            feature_matches = [m for m in matches if m.get("feature_label") == feature.get("label")]
            high_mod = [m for m in feature_matches if m.get("similarity_level") in ("HIGH", "MODERATE")]

            if high_mod:
                with st.expander(f"View matching claim elements ({len(high_mod)} patents)"):
                    for m in sorted(high_mod, key=lambda x: x.get("similarity_score", 0), reverse=True)[:5]:
                        score = m.get("similarity_score", 0)
                        level = m.get("similarity_level", "")
                        color = "#FF4444" if level == "HIGH" else "#FF9800"
                        st.markdown(
                            f"**{m.get('patent_number', '')}** "
                            f'<span style="color:{color}">({score:.3f} {level})</span>: '
                            f"*{m.get('element_text', '')[:120]}...*",
                            unsafe_allow_html=True,
                        )
            else:
                st.caption("No strong prior art matches found for this feature")


def _render_results() -> None:
    """Render the six-tab results layout."""

    # Analyzed Invention card — above tabs
    _render_analyzed_invention_card()

    tab_summary, tab_features, tab_prior, tab_claims, tab_landscape, tab_report = st.tabs(
        ["Executive Summary", "Extracted Features", "Prior Art Results", "Claim Analysis", "Landscape", "Report"]
    )

    # Tab 0 — Executive Summary
    with tab_summary:
        st.subheader("Executive Summary")
        _render_executive_summary()

    # Tab 1 — Extracted Features
    with tab_features:
        _render_extracted_features_tab()

    # Tab 2 — Prior Art Results
    with tab_prior:
        st.subheader("Retrieved Patents")
        detail_df = st.session_state.get("detail_patents")
        if detail_df is None or (hasattr(detail_df, 'empty') and detail_df.empty):
            st.info("Patent results will appear here after analysis.")
        else:
            # --- Metric cards -------------------------------------------
            try:
                unique_assignees = set()
                for asg in detail_df["assignee_name"]:
                    if isinstance(asg, list):
                        for a in asg:
                            if a:
                                unique_assignees.add(a)
                    elif isinstance(asg, str) and asg:
                        unique_assignees.add(asg)

                import pandas as _pd_metric
                dates = _pd_metric.to_numeric(
                    detail_df["publication_date"], errors="coerce"
                ).dropna()
                dates = dates[dates > 0]
                if not dates.empty:
                    oldest = str(int(dates.min()))[:4]
                    newest = str(int(dates.max()))[:4]
                    date_range = f"{oldest}–{newest}"
                else:
                    date_range = "N/A"

                m1, m2, m3 = st.columns(3)
                m1.metric("Patents Found", len(detail_df))
                m2.metric("Unique Assignees", len(unique_assignees))
                m3.metric("Date Range", date_range)
            except Exception:
                pass

            st.divider()

            # --- Clickable table (top 20) --------------------------------
            display_cols = ["publication_number", "title", "assignee_name",
                            "publication_date_str", "relevance_score"]
            display_cols = [c for c in display_cols if c in detail_df.columns]
            top20 = detail_df.head(20)[display_cols].copy()
            # Build markdown links for patent numbers
            if "publication_number" in top20.columns and "patent_url" in detail_df.columns:
                url_map = dict(
                    zip(detail_df["publication_number"], detail_df["patent_url"])
                )
                top20["patent_link"] = top20["publication_number"].apply(
                    lambda n: f"[{n}]({url_map.get(n, '#')})")
                top20 = top20.drop(columns=["publication_number"])
                cols_order = ["patent_link"] + [
                    c for c in top20.columns if c != "patent_link"
                ]
                top20 = top20[cols_order]

            # Flatten assignee lists for display
            if "assignee_name" in top20.columns:
                top20["assignee_name"] = top20["assignee_name"].apply(
                    lambda v: "; ".join(v[:2]) if isinstance(v, list) else str(v or "")
                )

            st.markdown("**Top 20 Most Relevant Patents**")
            st.dataframe(
                top20,
                use_container_width=True,
                column_config={
                    "patent_link": st.column_config.LinkColumn(
                        "Patent Number", display_text=r"(US-[\w-]+)"
                    )
                } if "patent_link" in top20.columns else {},
            )

            # --- Abstract expanders (top 10) ----------------------------
            st.divider()
            st.markdown("**Abstract Detail — Top 10 Patents**")
            for _, row in detail_df.head(10).iterrows():
                pub_num  = row.get("publication_number", "")
                url       = row.get("patent_url", "#")
                title_txt = row.get("title", "Untitled")
                abstract  = row.get("abstract") or "No abstract available (title-only retrieval)."
                with st.expander(f"{pub_num} — {title_txt[:80]}"):
                    st.markdown(f"**[Open on Google Patents ↗]({url})**")
                    st.write(abstract)

            # --- Success banner -----------------------------------------
            st.success(
                f"Retrieved {len(detail_df)} patents from Google BigQuery "
                "Patents Database (100M+ publications)"
            )

    # Tab 3 — Claim Analysis
    with tab_claims:
        st.subheader("Feature vs. Claim Comparison")
        claim_data = st.session_state.get("parsed_claims")
        if claim_data is None:
            st.info("Claim mapping will appear here after analysis.")
        else:
            # parsed_claims may be stored as a list (from demo session) or dict
            if isinstance(claim_data, list):
                claim_data = {"summary": {"attempted": len(claim_data), "successful": len(claim_data), "skipped": 0, "failed": 0}, "results": claim_data}
            summary = claim_data.get("summary", {})
            attempted  = summary.get("attempted", 0)
            successful = summary.get("successful", 0)
            skipped    = summary.get("skipped", 0)
            failed     = summary.get("failed", 0)

            _status_parts = []
            if skipped > 0:
                _status_parts.append(f"{skipped} had no claims data")
            if failed > 0:
                _status_parts.append(f"{failed} failed to parse")
            _status_suffix = f" ({', '.join(_status_parts)})" if _status_parts else ""
            st.markdown(
                f"Parsed claims for **{successful}** out of **{attempted}** patents"
                + _status_suffix
            )

            # ----------------------------------------------------------------
            # Similarity analysis results — two-layer view
            # ----------------------------------------------------------------
            sim  = st.session_state.get("similarity_results")
            cmat = st.session_state.get("comparison_matrix")  # enriched list

            if sim is not None:
                st.divider()
                st.subheader("Similarity Analysis")

                # --- Legend -----------------------------------------------
                with st.expander("How to read this table", expanded=False):
                    st.markdown(
                        """
**Two-layer analysis:** each match is scored by both an embedding model
(semantic similarity) and Gemini (contextual/technical analysis).

| Badge | Meaning |
|-------|---------|
| HIGH | Strong overlap in both layers |
| MODERATE | Partial overlap; notable differences present |
| LOW | Incidental or surface-level similarity only |
| DIVERGENCE | The two layers disagree — human review recommended |

**What this tool cannot do:** determine infringement, assess claim validity,
or replace professional patent counsel.
                        """
                    )

                if sim.get("error"):
                    st.warning(f"{sim['error']}")
                else:
                    stats = sim.get("stats", {})
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Comparisons", stats.get("total_comparisons", 0))
                    c2.metric("High Similarity", stats.get("high_matches", 0))
                    c3.metric("Moderate", stats.get("moderate_matches", 0))
                    c4.metric("Low", stats.get("low_matches", 0))

                    matches = sim.get("matches", [])
                    if not matches:
                        st.info(
                            "No matches exceeded the low-similarity threshold "
                            f"({0.30:.0%}). Patent language often differs greatly from "
                            "natural language descriptions."
                        )
                    elif cmat is not None and len(cmat) > 0:
                        # ---- ENRICHED two-layer view (grouped by patent) --
                        from modules.report_helpers import group_matches_by_patent

                        st.markdown("**Contextual Analysis — Top Matches (AI-enriched)**")
                        _bg   = {"HIGH": "#ffd6d6", "MODERATE": "#ffe4b5", "LOW": "#f0f0f0"}
                        _icon = {"HIGH": "[H]", "MODERATE": "[M]", "LOW": "[L]"}
                        _conf_colour = {"HIGH": "[H]", "MODERATE": "[M]", "LOW": "[L]"}

                        grouped = group_matches_by_patent(cmat)
                        for pat_group in grouped:
                            overall = pat_group.get("overall_confidence", "LOW")
                            div_flag = pat_group.get("divergence_flag", False)
                            pat_num   = pat_group.get("patent_number", "")
                            best_score = pat_group.get("best_score", 0.0)
                            n_features = pat_group.get("feature_count", 1)

                            # Build expander label — one per patent
                            div_prefix = "[!] " if div_flag else ""
                            icon = _icon.get(overall, "[L]")
                            feature_names = ", ".join(
                                f.get("feature_label", "")
                                for f in pat_group.get("all_features", [])[:3]
                            )
                            if n_features > 3:
                                feature_names += f" (+{n_features - 3} more)"
                            exp_label = (
                                f"{div_prefix}{icon} **{pat_num}** — "
                                f"Best: {best_score:.3f} ({overall}) — "
                                f"{n_features} feature(s): {feature_names}"
                            )

                            with st.expander(exp_label, expanded=False):
                                # Divergence warning banner
                                if div_flag:
                                    st.warning(
                                        f"**Divergence detected:** "
                                        f"{pat_group.get('divergence_note', '')}"
                                    )

                                # Show each feature matched to this patent — side-by-side view
                                for fi, feat in enumerate(pat_group.get("all_features", [])):
                                    feat_conf = feat.get("overall_confidence", "LOW")
                                    feat_icon = _icon.get(feat_conf, "[L]")
                                    st.markdown(
                                        f"{feat_icon} **{feat.get('feature_label', '')}** — "
                                        f"Claim {feat.get('claim_number', '?')} — "
                                        f"Score: `{feat.get('similarity_score', 0):.3f}` ({feat_conf})"
                                    )

                                    # Side-by-side comparison columns
                                    left_col, right_col = st.columns(2)
                                    feat_desc = feat.get("feature_description", "")
                                    elem_txt = feat.get("element_text", "")
                                    with left_col:
                                        st.markdown("**Your Feature**")
                                        st.markdown(f"**{feat.get('feature_label', '')}**")
                                        if feat_desc and elem_txt:
                                            st.markdown(
                                                highlight_overlapping_terms(feat_desc, elem_txt),
                                                unsafe_allow_html=True,
                                            )
                                        elif feat_desc:
                                            st.markdown(feat_desc)
                                    with right_col:
                                        st.markdown("**Matched Claim Element**")
                                        st.markdown(f"*{feat.get('patent_number', '')}*")
                                        if elem_txt and feat_desc:
                                            st.markdown(
                                                highlight_overlapping_terms(elem_txt, feat_desc),
                                                unsafe_allow_html=True,
                                            )
                                        elif elem_txt:
                                            st.markdown(elem_txt[:300])

                                    # Gemini analysis per-feature
                                    if feat.get("gemini_assessment"):
                                        st.markdown(f"**Analysis:** {feat['gemini_assessment']}")
                                    if feat.get("key_distinctions"):
                                        st.markdown("**Key Distinctions:**")
                                        for _d in feat["key_distinctions"]:
                                            st.markdown(f"- {_d}")

                                    if fi < len(pat_group.get("all_features", [])) - 1:
                                        st.divider()

                                # Show Gemini analysis from best match (patent-level)
                                gemini_expl = pat_group.get("gemini_explanation", "")
                                gemini_asmt = pat_group.get("gemini_assessment", "")
                                if gemini_expl:
                                    st.markdown("**What this claim element legally requires:**")
                                    st.info(gemini_expl)
                                if gemini_asmt:
                                    st.markdown("**Technical Comparison:**")
                                    st.write(gemini_asmt)

                                key_dist = pat_group.get("key_distinctions", [])
                                if key_dist:
                                    st.markdown("**Key Distinctions (Patent-level):**")
                                    for d in key_dist:
                                        st.markdown(f"- {d}")

                                cannot = pat_group.get("cannot_determine", "")
                                if cannot:
                                    st.markdown("**Cannot Determine Without Expert Review:**")
                                    st.caption(cannot)

                        # Embedding-only matches (LOW only) in a compact table
                        low_only = [
                            m for m in matches
                            if m["similarity_level"] == "LOW"
                        ]
                        if low_only:
                            with st.expander(
                                f"{len(low_only)} low-similarity match(es) "
                                "(embedding-only, not analysed by Gemini)"
                            ):
                                import pandas as _pd2
                                _df2 = _pd2.DataFrame(low_only)[[
                                    "feature_label", "patent_number", "claim_number",
                                    "element_id", "similarity_score", "element_text",
                                ]]
                                st.dataframe(_df2, use_container_width=True)

                    else:
                        # ---- Basic embedding-only view -------------------
                        seen: set = set()
                        top_rows: list[dict] = []
                        for m in matches:
                            lbl = m["feature_label"]
                            if lbl not in seen:
                                seen.add(lbl)
                                top_rows.append(m)

                        _bg   = {"HIGH": "#ffd6d6", "MODERATE": "#ffe4b5", "LOW": "#f0f0f0"}
                        _icon = {"HIGH": "[H]", "MODERATE": "[M]", "LOW": "[L]"}

                        header_cells = "".join(
                            f"<th style='padding:6px 10px;background:#f8f8f8;"
                            f"border-bottom:2px solid #ccc;text-align:left'>{h}</th>"
                            for h in ["Feature", "Top Match Element", "Patent #",
                                      "Claim", "Score", "Level"]
                        )
                        row_html = ""
                        for r in top_rows:
                            bg = _bg.get(r["similarity_level"], "#ffffff")
                            icon = _icon.get(r["similarity_level"], "")
                            elem_txt = r["element_text"]
                            if len(elem_txt) > 120:
                                elem_txt = elem_txt[:117] + "…"
                            cells = "".join(
                                f"<td style='padding:5px 10px;vertical-align:top'>{v}</td>"
                                for v in [
                                    f"<strong>{r['feature_label']}</strong>",
                                    elem_txt,
                                    r["patent_number"],
                                    r["claim_number"],
                                    f"{r['similarity_score']:.3f}",
                                    f"{icon} {r['similarity_level']}",
                                ]
                            )
                            row_html += f"<tr style='background:{bg}'>{cells}</tr>\n"

                        st.markdown("**Top Match per Feature**")
                        st.markdown(
                            "<div style='overflow-x:auto'>"
                            "<table style='border-collapse:collapse;width:100%;"
                            "font-size:0.875rem'>"
                            f"<thead><tr>{header_cells}</tr></thead>"
                            f"<tbody>{row_html}</tbody>"
                            "</table></div>",
                            unsafe_allow_html=True,
                        )

                        with st.expander(f"See all {len(matches)} match entries"):
                            import pandas as _pd
                            _df = _pd.DataFrame(matches)[[
                                "feature_label", "patent_number", "claim_number",
                                "element_id", "similarity_score", "similarity_level",
                                "element_text",
                            ]]
                            st.dataframe(_df, use_container_width=True)

                    # --- Unmatched features --------------------------------
                    unmatched = sim.get("unmatched_features", [])
                    if unmatched:
                        st.divider()
                        st.markdown("**Features With No Strong Prior Art Match**")
                        for uf in unmatched:
                            st.markdown(
                                f"**{uf['label']}** — {uf.get('description', '')}"
                            )
                    elif matches:
                        st.success(
                            "All features have at least one HIGH or MODERATE match "
                            "in the retrieved patents."
                        )

            # ----------------------------------------------------------------
            # Parsed claim details (expandable, below similarity summary)
            # ----------------------------------------------------------------
            results = claim_data.get("results", [])
            if not results:
                st.warning("No structured claims could be extracted from the retrieved patents.")
            else:
                st.divider()
                st.markdown("**Parsed Claim Detail**")
                for parsed in results:
                    pub   = parsed.get("patent_number", "")
                    ind   = parsed.get("independent_claims", [])
                    total = parsed.get("total_claims_found", 0)
                    conf  = parsed.get("parsing_confidence", "")
                    conf_colour = {"HIGH": "[H]", "MODERATE": "[M]", "LOW": "[L]"}.get(conf, "")
                    label = (
                        f"{conf_colour} {pub} — "
                        f"{len(ind)} independent claim(s) of {total} total "
                        f"[{conf}]"
                    )
                    with st.expander(label):
                        for claim in ind:
                            c_num  = claim.get("claim_number", "?")
                            preamble = claim.get("preamble", "")
                            transition = claim.get("transitional_phrase", "")
                            elements = claim.get("elements", [])
                            plain = claim.get("plain_english", "")

                            st.markdown(f"**Claim {c_num}**")
                            if preamble:
                                st.markdown(f"*Preamble:* {preamble}")
                            if transition:
                                st.markdown(f"*Transition:* `{transition}`")
                            if elements:
                                st.markdown("**Elements:**")
                                for el in elements:
                                    st.markdown(
                                        f"- `{el['id']}` {el['text']}"
                                    )
                            if plain:
                                st.caption(f"Plain English: {plain}")
                            st.divider()

            # ----------------------------------------------------------------
            # White Space Analysis section (below parsed claim details)
            # ----------------------------------------------------------------
            st.divider()
            st.subheader("White Space Analysis")
            white_spaces = st.session_state.get("white_spaces")

            if white_spaces is None:
                st.info("White space analysis will appear here after analysis.")
            elif len(white_spaces) == 0:
                st.info(
                    "All described features showed moderate or high similarity "
                    "to existing patents. This may indicate a crowded technology "
                    "space. Professional analysis recommended."
                )
            else:
                _ws_conf_icon = {
                    "HIGH": "[H]", "MODERATE": "[M]",
                    "LOW": "[L]", "INSUFFICIENT": "[?]",
                }
                _ws_type_icon = {
                    "Feature Gap": "[ ]",
                    "Classification Gap": "[ ]",
                    "Classification Density": "[ ]",
                    "Combination Novelty": "[ ]",
                }
                st.markdown(
                    f"Found **{len(white_spaces)}** potential white-space "
                    f"area(s). These are research signals, not legal findings."
                )
                for ws in white_spaces:
                    conf = ws.get("confidence", {})
                    conf_level = conf.get("level", "LOW")
                    conf_icon = _ws_conf_icon.get(conf_level, "[?]")
                    type_icon = _ws_type_icon.get(ws["type"], "[ ]")
                    exp_label = (
                        f"{type_icon} **{ws['type']}** {conf_icon} {conf_level} "
                        f"— {ws['title']}"
                    )
                    with st.expander(exp_label, expanded=False):
                        st.markdown(ws["description"])

                        boundary = ws.get("boundary_patents", [])
                        if boundary:
                            st.markdown("**Nearest Prior Art:**")
                            import pandas as _pd_ws
                            _bp_rows = []
                            for bp in boundary:
                                if isinstance(bp, dict):
                                    score_str = (
                                        f"{bp['score']:.3f}"
                                        if bp.get("score") is not None
                                        else "—"
                                    )
                                    _bp_rows.append({
                                        "Patent": bp.get("patent", ""),
                                        "Score": score_str,
                                        "Claim Element": bp.get("element", ""),
                                    })
                                else:
                                    # boundary_patents may be plain patent number strings
                                    _bp_rows.append({
                                        "Patent": str(bp),
                                        "Score": "—",
                                        "Claim Element": "",
                                    })
                            st.dataframe(
                                _pd_ws.DataFrame(_bp_rows),
                                use_container_width=True,
                                hide_index=True,
                            )

                        st.markdown(
                            f"**Confidence:** {conf_icon} {conf_level}"
                        )
                        st.caption(conf.get("rationale", ""))
                        st.caption(ws.get("data_completeness", ""))
                        st.caption(f"{ws.get('disclaimer', '')}")

    # Tab 4 — Landscape
    with tab_landscape:
        st.subheader("Patent Landscape")
        landscape_figs = st.session_state.get("landscape_figures")
        landscape_df_raw = st.session_state.get("landscape_patents")
        if not landscape_figs:
            st.info("Landscape charts will appear here after analysis.")
        else:
            n_patents = len(landscape_df_raw) if landscape_df_raw is not None else 0

            st.plotly_chart(
                landscape_figs["filing_trends"],
                use_container_width=True,
            )
            st.caption(
                "Filing trends show how patent activity in this technology area "
                "has evolved year-over-year based on the filing date recorded in "
                "the USPTO database."
            )

            st.plotly_chart(
                landscape_figs["top_assignees"],
                use_container_width=True,
            )
            st.caption(
                "Top patent holders ranked by the number of relevant patents "
                "retrieved.  Assignee names are taken from the primary applicant "
                "listed on each patent."
            )

            st.plotly_chart(
                landscape_figs["cpc_distribution"],
                use_container_width=True,
            )
            cpc_available = (
                landscape_df_raw["cpc_code"]
                .dropna()
                .apply(lambda x: isinstance(x, list) and len(x) > 0)
                .sum()
                if landscape_df_raw is not None and "cpc_code" in landscape_df_raw.columns
                else 0
            )
            st.caption(
                f"Technology classification distribution based on CPC codes. "
                f"CPC data available for {cpc_available} of {n_patents} patents."
            )

            st.divider()
            st.markdown(
                f"Analysis based on **{n_patents}** patents from the USPTO database."
            )

    # Tab 5 — Report
    with tab_report:
        st.subheader("Download Report")

        # Determine which phases are complete
        _phases = {
            "Patent Retrieval":       st.session_state.get("detail_patents") is not None,
            "Claim Parsing":          st.session_state.get("parsed_claims") is not None,
            "Similarity Analysis":    st.session_state.get("similarity_results") is not None,
            "Landscape Analysis":     st.session_state.get("landscape_figures") is not None,
            "White-Space Finder":     st.session_state.get("white_spaces") is not None,
        }
        _all_done = all(_phases.values())

        if not _all_done:
            st.warning(
                "Some analysis phases have not yet completed. "
                "Run the full analysis first for the most complete report."
            )
            for phase_name, done in _phases.items():
                icon = "[x]" if done else "[ ]"
                st.markdown(f"{icon} {phase_name}")

        # Allow generation even if landscape/whitespace incomplete
        _retrieval_done = st.session_state.get("detail_patents") is not None
        if _retrieval_done:
            if st.button("Generate Report", type="primary", key="gen_report_btn"):
                with st.spinner("Generating PDF..."):
                    try:
                        _rg = ReportGenerator()
                        _pdf_bytes = _rg.generate(dict(st.session_state))
                        st.session_state["_pdf_bytes"] = _pdf_bytes
                    except Exception as _rg_exc:
                        st.error(f"Report generation failed: {_rg_exc}")
                        logger.exception("ReportGenerator.generate() failed")
                        st.session_state["_pdf_bytes"] = None

        if st.session_state.get("_pdf_bytes"):
            st.download_button(
                label="Download PDF Report",
                data=st.session_state["_pdf_bytes"],
                file_name="patentscout_report.pdf",
                mime="application/pdf",
                key="download_report_btn",
            )
            _size_kb = len(st.session_state["_pdf_bytes"]) / 1024
            st.caption(f"Report size: {_size_kb:.1f} KB")

            # Executive summary preview
            st.divider()
            st.subheader("Executive Summary Preview")
            _strategy_prev   = st.session_state.get("search_strategy") or {}
            _detail_prev     = st.session_state.get("detail_patents")
            _sim_prev        = st.session_state.get("similarity_results") or {}
            _ws_prev         = st.session_state.get("white_spaces") or []
            _stats_prev      = _sim_prev.get("stats", {})
            _n_pat           = len(_detail_prev) if _detail_prev is not None else 0
            _high            = _stats_prev.get("high_matches", 0)
            _mod             = _stats_prev.get("moderate_matches", 0)
            _ws_n            = len(_ws_prev)
            st.markdown(
                f"PatentScout retrieved and analysed **{_n_pat}** patents from the "
                f"Google BigQuery Patents database. Similarity scoring found "
                f"**{_high}** high-overlap and **{_mod}** moderate-overlap match(es). "
                f"White-space analysis identified **{_ws_n}** potential opportunity area(s)."
            )
        elif not _retrieval_done:
            st.info(
                "Run the analysis from the sidebar to retrieve patents, "
                "then return here to generate and download your PDF report."
            )



# Sidebar — input panel

with st.sidebar:
    st.subheader("Describe Your Invention")

    # Text description
    invention_text = st.text_area(
        label="Invention description",
        label_visibility="collapsed",
        value=st.session_state["invention_text"],
        placeholder="Example: A portable solar panel that folds into a compact case and charges mobile phones via USB-C...",
        height=200,
    )
    st.session_state["invention_text"] = invention_text

    # Image upload
    uploaded_file = st.file_uploader(
        "Upload a Design Sketch (optional)",
        type=["png", "jpg", "jpeg"],
        help="Max 10 MB. Image will be resized to 1024 × 1024 if larger.",
    )

    if uploaded_file is not None:
        raw_bytes = uploaded_file.getvalue()
        if len(raw_bytes) > 10 * 1024 * 1024:
            st.error("Image exceeds 10 MB. Please upload a smaller file.")
        else:
            st.session_state["invention_image"] = raw_bytes
            st.image(
                st.session_state["invention_image"],
                use_container_width=True,
            )
            st.caption(f"✓ {uploaded_file.name}")
    else:
        # Only reset stored bytes if not loaded from a demo session
        if not st.session_state.get("is_demo"):
            st.session_state["invention_image"] = None

    # Show demo-loaded sketch in sidebar (file uploader not active)
    if uploaded_file is None and st.session_state.get("invention_image"):
        st.image(
            st.session_state["invention_image"],
            use_container_width=True,
        )
        st.caption("✓ doorbell_sketch.png")

    # Action buttons
    col_analyze, col_clear = st.columns([2, 1])
    with col_analyze:
        analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)
    with col_clear:
        clear_clicked = st.button("Clear", use_container_width=True)

    st.caption("PatentScout is a research tool, not legal advice.")

    # -- Demo load buttons (collapsible) --------------------------------
    st.divider()
    with st.expander("Explore Example Analyses", expanded=False):
        if st.button("Solar Charger Demo", use_container_width=True, key="solar_demo"):
            if _load_solar_demo():
                _show_demo_progress()
                st.rerun()
            else:
                st.error("Could not load solar charger demo data.")
        if st.button("Doorbell Demo (Text + Sketch)", use_container_width=True, key="doorbell_demo"):
            if _load_doorbell_demo():
                _show_demo_progress()
                st.rerun()
            else:
                st.error("Could not load doorbell demo data.")

    # -- "Try Another Invention" button ------------------------------------
    st.divider()
    if st.session_state.get("analysis_complete"):
        if st.button("Try Another Invention", use_container_width=True):
            keys_to_clear = [
                "invention_text", "invention_image", "sketch_used",
                "is_demo", "analysis_timestamp",
                "detail_patents", "landscape_patents", "parsed_claims",
                "similarity_results", "comparison_matrix", "white_spaces",
                "search_strategy", "landscape_figures", "chart_images",
                "analysis_complete", "_pdf_bytes", "query_cache",
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    # -- Query cost widget -------------------------------------------------
    total_gb = st.session_state.get("total_gb_scanned", 0.0) or 0.0
    if total_gb > 0:
        st.caption(f"Total GB scanned this session: **{total_gb:.2f} GB**")
        if total_gb > 15:
            st.warning(
                f"{total_gb:.1f} GB scanned — approaching 20 GB budget cap."
            )

    # Handle Clear
    if clear_clicked:
        st.session_state["invention_text"] = ""
        st.session_state["invention_image"] = None
        st.session_state["sketch_used"] = False
        st.session_state["is_demo"] = False
        st.session_state["analysis_timestamp"] = ""
        st.session_state["analysis_complete"] = False
        st.session_state["prior_art_results"]  = None
        st.session_state["claim_analysis"]       = None
        st.session_state["landscape_data"]       = None
        st.session_state["report_data"]          = None
        st.session_state["search_strategy"]      = None
        st.session_state["detail_patents"]       = None
        st.session_state["landscape_patents"]    = None
        st.session_state["parsed_claims"]        = None
        st.session_state["similarity_results"]   = None
        st.session_state["comparison_matrix"]    = None
        st.session_state["landscape_figures"]    = None
        st.session_state["chart_images"]         = {}
        st.session_state["white_spaces"]         = None
        st.session_state["_pdf_bytes"]            = None



# Main area — welcome or results

if not analyze_clicked and not st.session_state["analysis_complete"]:
    # Welcome / empty state
    st.markdown(
        '<h1 style="font-size:40px; font-weight:700; margin-bottom:0; padding-bottom:0;">'
        'PatentScout</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:18px; color:#6c757d; margin-top:4px; margin-bottom:24px;">'
        'Patent Prior Art Research Tool</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "**AI-powered patent landscape analysis.** PatentScout helps inventors and product "
        "teams understand the prior art landscape around their ideas before investing in development."
    )

    st.divider()
    st.markdown("### How It Works")

    cols = st.columns(4, gap="medium")
    steps = [
        ("\u2460 Describe", "Enter your invention concept and optionally upload a design sketch"),
        ("\u2461 Extract", "AI identifies key technical features and relevant patent classifications"),
        ("\u2462 Analyze", "Patents are retrieved, claims parsed, and compared against your features"),
        ("\u2463 Report", "Download a detailed prior art landscape report with actionable insights"),
    ]
    for col, (title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"**{title}**")
            st.caption(desc)

    st.divider()
    st.markdown("### \u2192 Describe your invention in the sidebar and click **Analyze**")
    st.caption("Or explore a pre-built example \u2014 expand *Explore Example Analyses* in the sidebar")
    st.caption("Powered by: Gemini 2.0 Flash \u00b7 Google BigQuery Patents \u00b7 sentence-transformers")

elif analyze_clicked:
    # Validate input
    is_valid, error_msg = validate_input(
        st.session_state["invention_text"],
        st.session_state["invention_image"],
    )

    if not is_valid:
        st.error(error_msg)
    else:
        from datetime import datetime as _dt_pipeline
        st.session_state["is_demo"] = False
        st.session_state["analysis_timestamp"] = _dt_pipeline.now().strftime("%Y-%m-%d %H:%M")
        _pipeline_start = time.time()
        try:
            with st.spinner("Extracting features and building search strategy..."):
                qb = QueryBuilder(api_key=_get_gemini_api_key())
                strategy = qb.extract_features(
                    st.session_state["invention_text"],
                    st.session_state["invention_image"],
                )
                # Attach BigQuery WHERE clause fragments directly to strategy
                where = qb.build_bigquery_where_clause(strategy)
                strategy.update(where)
                st.session_state["search_strategy"] = strategy

            n_features = len(strategy.get("features", []))
            n_cpc = len(strategy.get("cpc_codes", []))
            n_terms = len(strategy.get("search_terms", []))
            st.success(
                f"Extracted {n_features} features, predicted {n_cpc} CPC "
                f"codes, generated {n_terms} search term groups"
            )

            # -- Feature reformulation into patent language ----------------
            with st.spinner("Reformulating features into patent language..."):
                try:
                    strategy["features"] = qb.reformulate_features_for_patent_language(
                        strategy.get("features", [])
                    )
                    st.session_state["search_strategy"] = strategy
                    reformulated = sum(
                        1 for f in strategy["features"] if f.get("patent_language")
                    )
                    st.caption(
                        f"Patent-language reformulation: "
                        f"{reformulated}/{n_features} features rewritten."
                    )
                except Exception as _ref_exc:
                    logger.warning("Feature reformulation failed: %s", _ref_exc)
                    st.caption(
                        "Reformulation skipped — using original descriptions for similarity."
                    )

            # -- Patent retrieval with query cache -------------------------
            _cache_key = get_cache_key(strategy)
            _cache = st.session_state.get("query_cache", {})

            retrieval_error: str | None = None
            if _cache_key in _cache:
                detail_df   = _cache[_cache_key]["detail_df"]
                landscape_df = _cache[_cache_key]["landscape_df"]
                st.session_state["detail_patents"]    = detail_df
                st.session_state["landscape_patents"] = landscape_df
                st.session_state["analysis_complete"] = True
                st.caption("Results loaded from session cache (no new BigQuery charges).")
            else:
                with st.spinner("Searching patent database (usually 15–30 s)..."):
                    try:
                        _bq_client = _get_bigquery_client()
                        retriever = PatentRetriever(bq_client=_bq_client)
                        _desc = st.session_state.get("invention_text", "")
                        detail_df, landscape_df = retriever.search(
                            strategy, user_description=_desc,
                        )

                        st.session_state["detail_patents"]    = detail_df
                        st.session_state["landscape_patents"] = landscape_df
                        st.session_state["analysis_complete"] = True

                        # Cache results to avoid re-charging on re-runs
                        _cache[_cache_key] = {
                            "detail_df":    detail_df,
                            "landscape_df": landscape_df,
                        }
                        st.session_state["query_cache"] = _cache

                        # -- Landscape visualisations ----------------------
                        if landscape_df is not None and not landscape_df.empty:
                            try:
                                _analyzer = LandscapeAnalyzer(landscape_df)
                                _figs = {
                                    "filing_trends": _analyzer.filing_trends(),
                                    "top_assignees": _analyzer.top_assignees(),
                                    "cpc_distribution": _analyzer.cpc_distribution(),
                                }
                                st.session_state["landscape_figures"] = _figs
                                st.session_state["chart_images"] = (
                                    _analyzer.export_charts_as_images(_figs)
                                )
                            except Exception as _la_exc:
                                logger.warning(
                                    "LandscapeAnalyzer failed: %s", _la_exc
                                )
                                st.session_state["landscape_figures"] = {}
                                st.session_state["chart_images"] = {}

                    except Exception as exc:
                        retrieval_error = str(exc)
                        st.session_state["analysis_complete"] = True

            if retrieval_error:
                quota_hint = (
                    "You may have exceeded the BigQuery free-tier quota "
                    "(1 TB/month). Check the Google Cloud Console for quota usage."
                    if "quota" in retrieval_error.lower() or "billing" in retrieval_error.lower()
                    else ""
                )
                st.error(
                    f"Patent retrieval failed: {retrieval_error}\n\n{quota_hint}"
                )
            elif st.session_state["detail_patents"] is not None and \
                    st.session_state["detail_patents"].empty:
                st.warning(
                    "No patents found matching your description. "
                    "Try broadening your description or using more general technical terms."
                )

            # -- Claim parsing ------------------------------------------
            _gemini_client = None   # initialised here; may be overwritten below
            if not retrieval_error and (
                st.session_state["detail_patents"] is not None
                and not st.session_state["detail_patents"].empty
            ):
                _detail = st.session_state["detail_patents"]
                n_to_parse = min(20, len(_detail))
                parse_progress = st.progress(0, text="Parsing patent claims...")
                try:
                    from google import genai as _genai
                    _gemini_client = _genai.Client(
                        api_key=_get_gemini_api_key()
                    )
                except Exception:
                    _gemini_client = None

                parser = ClaimParser(gemini_client=_gemini_client)
                _rows = _detail.head(n_to_parse)
                results_list: list = []
                attempted = skipped = failed = 0

                for _i, (_idx, _row) in enumerate(_rows.iterrows()):
                    _pub  = str(_row.get("publication_number", "UNKNOWN"))
                    _text = _row.get("claims_text", None)
                    if not _text or str(_text).strip() in ("", "nan", "None"):
                        skipped += 1
                    else:
                        attempted += 1
                        try:
                            _parsed = parser.parse_claims(_pub, str(_text))
                            if _parsed.get("independent_claims"):
                                results_list.append(_parsed)
                            else:
                                failed += 1
                        except Exception as _exc:
                            logger.warning("Claim parse error for %s: %s", _pub, _exc)
                            failed += 1

                    pct = int((_i + 1) / n_to_parse * 100)
                    parse_progress.progress(
                        pct,
                        text=f"Parsing claims... {_i + 1}/{n_to_parse}",
                    )

                parse_progress.empty()
                st.session_state["parsed_claims"] = {
                    "results": results_list,
                    "summary": {
                        "attempted": attempted,
                        "successful": len(results_list),
                        "skipped": skipped,
                        "failed": failed,
                    },
                }

            # -- Similarity computation ------------------------------------
            _features = strategy.get("features", [])
            _parsed_list = (
                st.session_state["parsed_claims"].get("results", [])
                if st.session_state.get("parsed_claims")
                else []
            )
            if _features and _parsed_list:
                with st.spinner("Computing similarity analysis..."):
                    try:
                        _engine = _get_embedding_engine()
                        _sim = _engine.compute_similarity_matrix(
                            _features, _parsed_list
                        )
                        st.session_state["similarity_results"] = _sim
                    except Exception as _sim_exc:
                        logger.warning("EmbeddingEngine failed: %s", _sim_exc)
                        st.warning(
                            f"Similarity analysis failed: {_sim_exc}. "
                            "Continuing with remaining phases."
                        )
                        st.session_state["similarity_results"] = {"error": str(_sim_exc), "matches": [], "stats": {}}

            # -- Gemini contextual analysis --------------------------------
            _sim_results = st.session_state.get("similarity_results")
            _high_mod_count = len([
                m for m in (_sim_results or {}).get("matches", [])
                if m["similarity_level"] in ("HIGH", "MODERATE")
            ]) if _sim_results else 0

            if _sim_results and _high_mod_count > 0 and _gemini_client:
                _n_batches = -(-min(_high_mod_count, 15) // 3)  # ceil(n/3)
                _gem_progress = st.progress(
                    0, text="Running contextual AI analysis on top matches..."
                )
                with st.spinner(
                    f"Running contextual AI analysis on "
                    f"{min(_high_mod_count, 15)} pair(s) "
                    f"({_n_batches} batch(es))..."
                ):
                    try:
                        _mapper = ElementMapper(gemini_client=_gemini_client)
                        _enriched = _mapper.analyze_matches(
                            _sim_results,
                            invention_description=st.session_state.get("invention_text", ""),
                            detail_patents=st.session_state.get("detail_patents"),
                        )
                        st.session_state["comparison_matrix"] = _enriched
                    except Exception as _gem_exc:
                        logger.warning(
                            "ElementMapper failed: %s", _gem_exc
                        )
                        st.session_state["comparison_matrix"] = []
                _gem_progress.empty()
            elif _sim_results and _high_mod_count > 0 and not _gemini_client:
                st.warning(
                    "Gemini client unavailable — contextual analysis skipped."
                )

            # -- White space identification --------------------------------
            _ws_sim = st.session_state.get("similarity_results")
            _ws_strategy = st.session_state.get("search_strategy")
            _ws_detail = st.session_state.get("detail_patents")
            _ws_landscape = st.session_state.get("landscape_patents")
            _ws_landscape_size = (
                len(_ws_landscape) if _ws_landscape is not None else 0
            )
            if (
                _ws_sim
                and _ws_strategy
                and not _ws_sim.get("error")
            ):
                with st.spinner("Identifying white spaces..."):
                    try:
                        _ws_finder = WhiteSpaceFinder(
                            gemini_client=_gemini_client
                        )
                        _white_spaces = _ws_finder.identify_gaps(
                            features=_ws_strategy.get("features", []),
                            similarity_results=_ws_sim,
                            landscape_df_size=_ws_landscape_size,
                            search_strategy=_ws_strategy,
                            detail_df=_ws_detail,
                        )
                        st.session_state["white_spaces"] = _white_spaces
                    except Exception as _ws_exc:
                        logger.warning(
                            "WhiteSpaceFinder failed: %s", _ws_exc
                        )
                        st.session_state["white_spaces"] = []

            _render_results()
            _elapsed = time.time() - _pipeline_start
            logger.info("Full pipeline completed in %.1f s", _elapsed)
            _total_gb = st.session_state.get("total_gb_scanned", 0.0) or 0.0
            st.caption(
                f"Analysis completed in {_elapsed:.0f} seconds.  "
                f"Total GB scanned: {_total_gb:.2f} GB"
            )
        except Exception as exc:
            logger.exception("Pipeline top-level exception")
            st.error(
                f"Analysis failed during feature extraction: {exc}\n\n"
                "Please check your Gemini API key and try again. "
                "If the problem persists, try a shorter description."
            )

else:
    # analysis_complete is True from a previous run
    _render_results()
