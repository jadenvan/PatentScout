"""
PatentScout - MVP Streamlit Application
Entry point for the PatentScout patent analysis tool.
"""

from __future__ import annotations

import io
import json
import logging
import os
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

# ---------------------------------------------------------------------------
# Credential helpers — support both local .env and Streamlit Cloud secrets
# ---------------------------------------------------------------------------


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

# ---------------------------------------------------------------------------
# Page configuration — must be the very first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PatentScout",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session-state initialisation — runs before any widget is rendered
# ---------------------------------------------------------------------------


def _init_session_state() -> None:
    """Initialise all session-state keys with safe defaults."""
    defaults: dict = {
        "invention_text": "",
        "invention_image": None,   # raw bytes of the uploaded image
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
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


_init_session_state()

# ---------------------------------------------------------------------------
# Helper: results renderer  (defined BEFORE any code that calls it)
# ---------------------------------------------------------------------------


def _render_results() -> None:
    """Render the four-tab results layout."""
    tab_prior, tab_claims, tab_landscape, tab_report = st.tabs(
        ["Prior Art Results", "Claim Analysis", "Landscape", "Report"]
    )

    # Tab 1 — Prior Art Results
    with tab_prior:
        st.subheader("Retrieved Patents")
        detail_df = st.session_state.get("detail_patents")
        if detail_df is None or (hasattr(detail_df, 'empty') and detail_df.empty):
            st.info("Patent results will appear here after analysis.")
        else:
            # --- Metric cards -------------------------------------------
            try:
                unique_assignees = set(
                    a
                    for asg_list in detail_df["assignee_name"]
                    if isinstance(asg_list, list)
                    for a in asg_list
                    if a
                )
                dates = detail_df["publication_date"].dropna()
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
                abstract  = row.get("abstract", "No abstract available.")
                with st.expander(f"{pub_num} — {title_txt[:80]}"):
                    st.markdown(f"**[Open on Google Patents ↗]({url})**")
                    st.write(abstract)

            # --- Success banner -----------------------------------------
            st.success(
                f"✅ Retrieved {len(detail_df)} patents from Google BigQuery "
                "Patents Database (100M+ publications)"
            )

    # Tab 2 — Claim Analysis
    with tab_claims:
        st.subheader("Feature vs. Claim Comparison")
        claim_data = st.session_state.get("parsed_claims")
        if claim_data is None:
            st.info("Claim mapping will appear here after analysis.")
        else:
            summary = claim_data.get("summary", {})
            attempted  = summary.get("attempted", 0)
            successful = summary.get("successful", 0)
            skipped    = summary.get("skipped", 0)
            failed     = summary.get("failed", 0)

            st.markdown(
                f"Parsed claims for **{successful}** out of **{attempted}** patents "
                f"({skipped} had no claims data, {failed} failed to parse)"
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
                with st.expander("ℹ️ How to read this table", expanded=False):
                    st.markdown(
                        """
**Two-layer analysis:** each match is scored by both an embedding model
(semantic similarity) and Gemini (contextual/technical analysis).

| Badge | Meaning |
|-------|---------|
| 🔴 HIGH | Strong overlap in both layers |
| 🟠 MODERATE | Partial overlap; notable differences present |
| ⚪ LOW | Incidental or surface-level similarity only |
| ⚠️ DIVERGENCE | The two layers disagree — human review recommended |

**What this tool cannot do:** determine infringement, assess claim validity,
or replace professional patent counsel.
                        """
                    )

                if sim.get("error"):
                    st.warning(f"⚠️ {sim['error']}")
                else:
                    stats = sim.get("stats", {})
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Comparisons", stats.get("total_comparisons", 0))
                    c2.metric("🔴 High Similarity", stats.get("high_matches", 0))
                    c3.metric("🟠 Moderate", stats.get("moderate_matches", 0))
                    c4.metric("⚪ Low", stats.get("low_matches", 0))

                    matches = sim.get("matches", [])
                    if not matches:
                        st.info(
                            "No matches exceeded the low-similarity threshold "
                            f"({0.30:.0%}). Patent language often differs greatly from "
                            "natural language descriptions."
                        )
                    elif cmat is not None and len(cmat) > 0:
                        # ---- ENRICHED two-layer view ---------------------
                        st.markdown("**Contextual Analysis — Top Matches (AI-enriched)**")
                        _bg   = {"HIGH": "#ffd6d6", "MODERATE": "#ffe4b5", "LOW": "#f0f0f0"}
                        _icon = {"HIGH": "🔴", "MODERATE": "🟠", "LOW": "⚪"}
                        _conf_colour = {"HIGH": "🟢", "MODERATE": "🟡", "LOW": "🔴"}

                        for em in cmat:
                            overall = em.get("overall_confidence", "LOW")
                            div_flag = em.get("divergence_flag", False)
                            embed_lvl = em.get("similarity_level", "LOW")
                            gem_conf  = em.get("gemini_confidence", "LOW")
                            score     = em.get("similarity_score", 0.0)
                            pat_num   = em.get("patent_number", "")
                            claim_num = em.get("claim_number", "")

                            # Build expander label
                            div_prefix = "⚠️ " if div_flag else ""
                            icon = _icon.get(overall, "⚪")
                            exp_label = (
                                f"{div_prefix}{icon} **{em['feature_label']}** — "
                                f"{em['element_text'][:80]}{'…' if len(em['element_text']) > 80 else ''} "
                                f"| {pat_num} Cl.{claim_num} | "
                                f"Embed: {score:.3f} | Overall: {overall}"
                            )

                            with st.expander(exp_label, expanded=False):
                                # Divergence warning banner
                                if div_flag:
                                    st.warning(
                                        f"⚠️ **Divergence detected:** "
                                        f"{em.get('divergence_note', '')}"
                                    )

                                # Confidence badges row
                                b1, b2, b3 = st.columns(3)
                                b1.markdown(
                                    f"**Embedding Layer**  \n"
                                    f"{_icon.get(embed_lvl, '⚪')} {embed_lvl}  \n"
                                    f"Score: `{score:.3f}`"
                                )
                                b2.markdown(
                                    f"**Gemini Layer**  \n"
                                    f"{_conf_colour.get(gem_conf, '🔴')} {gem_conf}"
                                )
                                b3.markdown(
                                    f"**Overall**  \n"
                                    f"{_icon.get(overall, '⚪')} {overall}"
                                )

                                st.divider()

                                st.markdown(
                                    "**What this claim element legally requires:**"
                                )
                                st.info(em.get("gemini_explanation", "—"))

                                st.markdown("**Technical Comparison:**")
                                st.write(em.get("gemini_assessment", "—"))

                                key_dist = em.get("key_distinctions", [])
                                if key_dist:
                                    st.markdown("**Key Distinctions:**")
                                    for d in key_dist:
                                        st.markdown(f"- {d}")

                                cannot = em.get("cannot_determine", "")
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
                                f"⚪ {len(low_only)} low-similarity match(es) "
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
                        _icon = {"HIGH": "🔴", "MODERATE": "🟠", "LOW": "⚪"}

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
                                f"✅ **{uf['label']}** — {uf.get('description', '')}"
                            )
                    elif matches:
                        st.success(
                            "✅ All features have at least one HIGH or MODERATE match "
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
                    conf_colour = {"HIGH": "🟢", "MODERATE": "🟡", "LOW": "🔴"}.get(conf, "")
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
                    "HIGH": "🟢", "MODERATE": "🟡",
                    "LOW": "🔴", "INSUFFICIENT": "⚫",
                }
                _ws_type_icon = {
                    "Feature Gap": "🔍",
                    "Classification Gap": "📂",
                    "Combination Novelty": "✨",
                }
                st.markdown(
                    f"Found **{len(white_spaces)}** potential white-space "
                    f"area(s). These are research signals, not legal findings."
                )
                for ws in white_spaces:
                    conf = ws.get("confidence", {})
                    conf_level = conf.get("level", "LOW")
                    conf_icon = _ws_conf_icon.get(conf_level, "⚫")
                    type_icon = _ws_type_icon.get(ws["type"], "🔹")
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
                        st.caption(f"⚠️ {ws.get('disclaimer', '')}")

    # Tab 3 — Landscape
    with tab_landscape:
        st.subheader("Patent Landscape")
        landscape_figs = st.session_state.get("landscape_figures")
        landscape_df_raw = st.session_state.get("landscape_patents")
        if landscape_figs is None:
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

    # Tab 4 — Report
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
                icon = "✅" if done else "⏳"
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
                label="📥 Download PDF Report",
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


# ---------------------------------------------------------------------------
# Sidebar — input panel
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Describe Your Invention")

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
                Image.open(io.BytesIO(raw_bytes)),
                caption="Uploaded sketch",
                width=250,
            )
    else:
        # User cleared the uploader — reset stored bytes
        st.session_state["invention_image"] = None

    # Text description
    invention_text = st.text_area(
        "Describe your invention",
        value=st.session_state["invention_text"],
        placeholder=(
            "Describe what your invention does, how it works, "
            "and what problem it solves..."
        ),
        height=200,
    )
    st.session_state["invention_text"] = invention_text

    # Action buttons
    col_analyze, col_clear = st.columns([2, 1])
    with col_analyze:
        analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)
    with col_clear:
        clear_clicked = st.button("Clear", use_container_width=True)

    st.caption("PatentScout is a research tool, not legal advice.")

    # Handle Clear
    if clear_clicked:
        st.session_state["invention_text"] = ""
        st.session_state["invention_image"] = None
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
    # ---------------------------------------------------------------------------
    # Extracted features expander — shown after a successful analysis
    # ---------------------------------------------------------------------------
    strategy = st.session_state.get("search_strategy")
    if strategy:
        st.divider()
        with st.expander("🔍 Extracted Features", expanded=True):
            features = strategy.get("features", [])
            if features:
                for feat in features:
                    st.markdown(f"**{feat['label']}**")
                    st.caption(feat.get("description", ""))
                    keywords = feat.get("keywords", [])
                    if keywords:
                        st.write("Keywords: " + " · ".join(keywords))
                    st.write("")

            cpc_list = strategy.get("cpc_codes", [])
            if cpc_list:
                st.markdown("**Predicted CPC Codes**")
                for cpc in cpc_list:
                    st.markdown(f"- `{cpc['code']}` — {cpc.get('rationale', '')}")

# ---------------------------------------------------------------------------
# Main area — welcome or results
# ---------------------------------------------------------------------------

if not analyze_clicked and not st.session_state["analysis_complete"]:
    # Welcome message
    st.title("PatentScout")
    st.markdown(
        """
        **PatentScout** helps inventors and product teams understand the patent
        landscape around a new idea — before you invest in development.

        **What it does:**
        - Retrieves semantically similar prior-art patents from Google Patents
        - Maps your invention's features against existing patent claims
        - Visualises the patent landscape so you can spot white space
        - Generates a downloadable analysis report

        **To get started**, describe your invention in the sidebar and click
        **Analyze**.
        """
    )

elif analyze_clicked:
    # Validate input
    is_valid, error_msg = validate_input(
        st.session_state["invention_text"],
        st.session_state["invention_image"],
    )

    if not is_valid:
        st.error(error_msg)
    else:
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
                f"✅ Extracted {n_features} features, predicted {n_cpc} CPC "
                f"codes, generated {n_terms} search term groups"
            )

            # -- Patent retrieval ------------------------------------------
            retrieval_error: str | None = None
            with st.spinner("Searching patent database (⏳ usually 15–30 s)..."):
                try:
                    _bq_client = _get_bigquery_client()
                    retriever = PatentRetriever(bq_client=_bq_client)
                    detail_df, landscape_df = retriever.search(strategy)
                    st.session_state["detail_patents"]    = detail_df
                    st.session_state["landscape_patents"] = landscape_df
                    st.session_state["analysis_complete"] = True

                    # -- Landscape visualisations --------------------------
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
                            f"⚠️ Similarity analysis failed: {_sim_exc}. "
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
                        _enriched = _mapper.analyze_matches(_sim_results)
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
            st.caption(f"⏱ Analysis completed in {_elapsed:.0f} seconds.")
        except Exception as exc:
            logger.exception("Pipeline top-level exception")
            st.error(
                f"❌ Analysis failed during feature extraction: {exc}\n\n"
                "Please check your Gemini API key and try again. "
                "If the problem persists, try a shorter description."
            )

else:
    # analysis_complete is True from a previous run
    _render_results()
