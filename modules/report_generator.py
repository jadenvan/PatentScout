"""
PatentScout — Report Generator Module

Builds a multi-section downloadable PDF analysis report using ReportLab.
All session data produced by the analysis pipeline is accepted via a single
`session_data` dict so the caller (app.py) does not have to know internals.
"""

from __future__ import annotations

import logging
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)

from assets.report_styles import (
    PRIMARY, SECONDARY, ACCENT, WARNING, SUCCESS, LIGHT_GRAY, WHITE,
    CONF_COLORS, BASE_TABLE_STYLE, SCORE_BAR_FG, SCORE_BAR_BG,
    title_style, subtitle_style, date_style,
    heading_style, heading2_style,
    body_style, bullet_style, small_style, disclaimer_style,
    mono_style, small_bold, feature_header_style, recommendation_style,
)
from modules.report_helpers import (
    format_google_patent_url,
    highlight_snippet,
    safe_text_for_pdf,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sanitisation helper — strips characters that break ReportLab's latin-1
# codec fallback while preserving readability.
# ---------------------------------------------------------------------------
def _safe(text: Any, maxlen: int = 0) -> str:
    """Return a ReportLab-safe string, optionally truncated."""
    s = str(text) if text is not None else ''
    # encode/decode to strip non-latin-1 chars
    s = s.encode('latin-1', 'replace').decode('latin-1')
    if maxlen and len(s) > maxlen:
        s = s[:maxlen] + '...'
    return s


def _para(text: Any, style=None, maxlen: int = 0) -> Paragraph:
    """Convenience: build a Paragraph with sanitised text."""
    if style is None:
        style = body_style
    return Paragraph(_safe(text, maxlen), style)


# ---------------------------------------------------------------------------
# Score-bar flowable helper
# ---------------------------------------------------------------------------
def _render_score_bar(score: float, width: float = 1.5 * inch) -> Table:
    """Return a tiny coloured bar Table proportional to *score* (0-1)."""
    score = max(0.0, min(1.0, float(score)))
    filled_w = max(2, score * width)
    empty_w = max(0, width - filled_w)
    bar = Table(
        [['', '']],
        colWidths=[filled_w, empty_w],
        rowHeights=[10],
    )
    bar.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), SCORE_BAR_FG),
        ('BACKGROUND', (1, 0), (1, 0), SCORE_BAR_BG),
        ('TOPPADDING',    (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('LEFTPADDING',   (0, 0), (-1, -1), 0),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
    ]))
    return bar


# ---------------------------------------------------------------------------
# Page-number callback
# ---------------------------------------------------------------------------
def _add_page_number(canvas, doc):
    """Draw page number in the footer of every page."""
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(SECONDARY)
    canvas.drawCentredString(
        letter[0] / 2.0, 0.45 * inch,
        f"Page {doc.page}",
    )
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Consistency helper — fill missing fields in matches
# ---------------------------------------------------------------------------
def _normalise_matches(session_data: dict) -> list[dict]:
    """
    Return the matches list with every expected key guaranteed present.
    Also attempt to resolve missing ``element_text`` from detail_patents.
    """
    sim = session_data.get('similarity_results') or {}
    matches = list(sim.get('matches', []))

    # Build a quick lookup: publication_number -> detail patent dict
    detail_patents = session_data.get('detail_patents') or []
    pat_lookup: dict[str, dict] = {}
    if isinstance(detail_patents, list):
        for p in detail_patents:
            if isinstance(p, dict):
                pn = p.get('publication_number', '')
                if pn:
                    pat_lookup[pn] = p

    _DEFAULTS = {
        'feature_label': 'Unknown Feature',
        'feature_description': '',
        'element_text': '',
        'patent_number': '',
        'patent_title': '',
        'patent_assignee': '',
        'publication_date': '',
        'similarity_score': 0.0,
        'similarity_level': 'LOW',
        'gemini_explanation': '',
        'gemini_assessment': '',
        'key_distinctions': [],
        'cannot_determine': '',
        'overall_confidence': 'LOW',
        'divergence_flag': False,
    }

    for m in matches:
        for key, default in _DEFAULTS.items():
            m.setdefault(key, default)

        # Resolve missing element_text from detail_patents claims
        if not m.get('element_text'):
            pat = pat_lookup.get(m.get('patent_number', ''))
            if pat and pat.get('claims_text'):
                m['element_text'] = pat['claims_text'][:500]
            else:
                m['element_text'] = '[claim text not available]'
                logger.warning(
                    "element_text missing for patent %s, feature %s",
                    m.get('patent_number'), m.get('feature_label'),
                )

        # Resolve patent_title / patent_assignee from detail_patents
        if not m.get('patent_title'):
            pat = pat_lookup.get(m.get('patent_number', ''))
            if pat:
                m['patent_title'] = pat.get('title', '')
                asgn = pat.get('assignee_name', '')
                if isinstance(asgn, list):
                    asgn = '; '.join(str(a) for a in asgn[:2])
                m['patent_assignee'] = str(asgn)
                m['publication_date'] = str(pat.get('publication_date', ''))

    return matches


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------
class ReportGenerator:
    """
    Generates a professional 8-section PDF report from PatentScout session data.

    Usage::

        rg = ReportGenerator()
        pdf_bytes = rg.generate(st.session_state)
    """

    # Total usable width with 0.75" margins on letter paper
    _PAGE_W = 7.0 * inch

    def generate(self, session_data: dict) -> bytes:  # noqa: C901  (complexity OK here)
        """
        Build the full PDF and return the raw bytes.

        Args:
            session_data: A dict containing all keys populated by the
                          PatentScout analysis pipeline (mirrors st.session_state).

        Returns:
            PDF file as a bytes object.
        """
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )

        elements: list = []

        # Pre-compute shared data  ----------------------------------------
        strategy     = session_data.get('search_strategy') or {}
        detail_list  = session_data.get('detail_patents') or []
        # detail_patents may be a pandas DataFrame or a list of dicts
        import pandas as pd
        if isinstance(detail_list, pd.DataFrame):
            detail_df = detail_list
        elif isinstance(detail_list, list):
            detail_df = pd.DataFrame(detail_list) if detail_list else pd.DataFrame()
        else:
            detail_df = pd.DataFrame()

        sim_results  = session_data.get('similarity_results') or {}
        white_spaces = session_data.get('white_spaces') or []
        cmat         = session_data.get('comparison_matrix') or []
        matches      = _normalise_matches(session_data)
        features     = strategy.get('features', [])
        keywords     = strategy.get('keywords', [])

        # Extract plain-string keyword terms for snippet highlighting
        _kw_strings: list[str] = []
        for k in keywords:
            _kw_strings.append(str(k) if not isinstance(k, dict) else k.get('label', ''))
        for f in features:
            if isinstance(f, dict):
                _kw_strings.append(f.get('label', ''))
                _kw_strings.extend(f.get('keywords', []))
            else:
                _kw_strings.append(str(f))
        _kw_strings = [t for t in _kw_strings if t]

        n_patents   = len(detail_df) if not detail_df.empty else 0
        stats       = sim_results.get('stats', {})
        high_count  = stats.get('high_matches', 0)
        mod_count   = stats.get('moderate_matches', 0)
        ws_count    = len(white_spaces)

        # ================================================================== #
        #  PAGE 1 — Title Page + Disclaimer                                   #
        # ================================================================== #
        elements.append(Spacer(1, 0.5 * inch))
        elements.append(_para('PATENTSCOUT', title_style))
        elements.append(_para('Preliminary Research Report', subtitle_style))
        elements.append(Spacer(1, 12))
        elements.append(_para(
            f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
            date_style,
        ))
        elements.append(Spacer(1, 24))

        disclaimer_text = (
            "<b>DISCLAIMER AND SCOPE</b><br/><br/>"
            "This report is an automated preliminary research summary generated "
            "by PatentScout. It is <b>NOT</b> a legal opinion, freedom-to-operate "
            "analysis, patentability opinion, or infringement analysis. All findings "
            "are based on automated search and comparison against the Google Patents "
            "Public Dataset via BigQuery and should be reviewed and validated by a "
            "qualified patent professional before any business or legal decisions "
            "are made."
        )
        disclaimer_table = Table(
            [[Paragraph(disclaimer_text, disclaimer_style)]],
            colWidths=[self._PAGE_W],
        )
        disclaimer_table.setStyle(TableStyle([
            ('BOX',           (0, 0), (-1, -1), 2, WARNING),
            ('BACKGROUND',    (0, 0), (-1, -1), WARNING),
            ('TOPPADDING',    (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('LEFTPADDING',   (0, 0), (-1, -1), 12),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 12),
        ]))
        elements.append(disclaimer_table)
        elements.append(PageBreak())

        # ================================================================== #
        #  TABLE OF CONTENTS                                                   #
        # ================================================================== #
        elements.append(_para('Table of Contents', heading_style))
        elements.append(Spacer(1, 8))
        toc_items = [
            '1. Executive Summary',
            '2. Search Methodology',
            '3. Prior Art Summary',
            '4. Claim Element Comparison',
            '5. Match Details',
            '6. Patent Landscape',
            '7. White Space Analysis',
            '8. Recommended Next Steps',
            '9. References',
        ]
        for item in toc_items:
            elements.append(_para(item, body_style))
        elements.append(PageBreak())

        # ================================================================== #
        #  SECTION 1 — Executive Summary                                       #
        # ================================================================== #
        elements.append(_para('1. Executive Summary', heading_style))

        concept_overview = (
            session_data.get('invention_text', '') or
            ', '.join(
                (f.get('label', str(f)) if isinstance(f, dict) else str(f))
                for f in features[:5]
            ) or
            'No description provided.'
        )

        exec_summary = (
            f"PatentScout retrieved and analysed <b>{n_patents}</b> patents from the "
            f"Google BigQuery Patents database relevant to the submitted invention "
            f"description. Automated similarity scoring identified "
            f"<b>{high_count}</b> high-overlap match(es) and "
            f"<b>{mod_count}</b> moderate-overlap match(es). "
            f"White-space analysis surfaced <b>{ws_count}</b> potential innovation "
            f"opportunity area(s) where claimed coverage appears limited."
        )
        elements.append(_para(f'<b>Concept Overview:</b> {_safe(concept_overview, 400)}', body_style))
        elements.append(Spacer(1, 6))
        elements.append(_para(exec_summary, body_style))
        elements.append(Spacer(1, 12))

        # Key metrics mini-table
        metrics_data = [
            ['Metric', 'Value'],
            ['Total patents retrieved', str(n_patents)],
            ['High-overlap matches', str(high_count)],
            ['Moderate-overlap matches', str(mod_count)],
            ['White-space opportunities', str(ws_count)],
            ['Features analysed', str(len(features))],
        ]
        metrics_tbl = Table(metrics_data, colWidths=[4 * inch, 3 * inch])
        metrics_tbl.setStyle(TableStyle(BASE_TABLE_STYLE))
        elements.append(metrics_tbl)
        elements.append(PageBreak())

        # ================================================================== #
        #  SECTION 2 — Search Methodology                                     #
        # ================================================================== #
        elements.append(_para('2. Search Methodology', heading_style))

        cpc_codes = strategy.get('cpc_codes', [])
        queries   = strategy.get('queries', [])

        elements.append(_para(
            '<b>Database:</b> Google Patents Public Data via BigQuery '
            '(patents-public-data.patents.publications)',
            body_style,
        ))
        elements.append(_para(
            f'<b>Date of search:</b> {datetime.now().strftime("%B %d, %Y")}',
            body_style,
        ))
        elements.append(_para(
            f'<b>Total patents reviewed:</b> {n_patents}',
            body_style,
        ))
        elements.append(Spacer(1, 8))

        if cpc_codes:
            elements.append(_para('<b>CPC classification codes searched:</b>', body_style))
            for code_item in cpc_codes:
                if isinstance(code_item, dict):
                    code = code_item.get('code', '')
                    desc = code_item.get('description', '')
                    elements.append(_para(f'\u2022 {_safe(code)} \u2014 {_safe(desc, 120)}', bullet_style))
                else:
                    elements.append(_para(f'\u2022 {_safe(code_item)}', bullet_style))

        if keywords:
            elements.append(Spacer(1, 8))
            elements.append(_para('<b>Keyword terms used:</b>', body_style))
            elements.append(_para(', '.join(_safe(k) for k in _kw_strings[:30]), body_style))

        if queries:
            elements.append(Spacer(1, 8))
            elements.append(_para('<b>Search queries executed:</b>', body_style))
            for q in queries[:5]:
                elements.append(_para(f'\u2022 {_safe(q, 200)}', bullet_style))
        elements.append(PageBreak())

        # ================================================================== #
        #  SECTION 3 — Prior Art Summary Table (top 20)                        #
        # ================================================================== #
        elements.append(_para('3. Prior Art Summary', heading_style))

        if not detail_df.empty:
            snippet_terms = _kw_strings[:12]
            prior_art_rows = [['#', 'Patent', 'Title', 'Assignee', 'Date', 'Score', 'Claim Snippet']]
            for idx, (_, row) in enumerate(detail_df.head(20).iterrows(), start=1):
                pub_num  = _safe(row.get('publication_number', ''))
                title    = _safe(row.get('title', 'Untitled'), 50)
                asgn     = row.get('assignee_name', '')
                if isinstance(asgn, list):
                    asgn = '; '.join(_safe(a) for a in asgn[:2])
                else:
                    asgn = _safe(asgn, 35)
                date_val = str(row.get('publication_date', ''))[:4]
                score    = row.get('relevance_score', 0)
                score_s  = f"{float(score):.3f}" if score else '\u2014'

                url = format_google_patent_url(pub_num)
                num_cell = Paragraph(
                    f'<b>{_safe(pub_num)}</b><br/>'
                    f'<font size="6" color="#3498DB"><link href="{url}">{_safe(url, 42)}</link></font>',
                    small_style,
                )

                # Build snippet from claims_text if available
                claims_txt = str(row.get('claims_text', ''))[:300]
                snippet_html = highlight_snippet(claims_txt, snippet_terms, max_len=80)
                snippet_cell = Paragraph(_safe(snippet_html) if not snippet_html else snippet_html, small_style)

                prior_art_rows.append([
                    str(idx),
                    num_cell,
                    _para(title, small_style),
                    _para(asgn, small_style),
                    date_val,
                    score_s,
                    snippet_cell,
                ])

            prior_art_tbl = Table(
                prior_art_rows,
                colWidths=[0.25 * inch, 1.1 * inch, 1.3 * inch, 1.0 * inch,
                           0.45 * inch, 0.5 * inch, 2.4 * inch],
                repeatRows=1,
            )
            prior_art_tbl.setStyle(TableStyle(BASE_TABLE_STYLE))
            elements.append(prior_art_tbl)
        else:
            elements.append(_para('No patent data available.', body_style))
        elements.append(PageBreak())

        # ================================================================== #
        #  SECTION 4 — Claim Element Comparison Matrix                         #
        # ================================================================== #
        elements.append(_para('4. Claim Element Comparison', heading_style))

        if cmat:
            cmp_rows = [['Feature', 'Claim Element', 'Patent #',
                         'Embed Score', 'AI Conf.', 'Overall']]
            divergences = []
            for em in cmat:
                overall   = _safe(em.get('overall_confidence', 'LOW'))
                gem_conf  = _safe(em.get('gemini_confidence', '\u2014'))
                score     = em.get('similarity_score', 0.0)
                score_s   = f"{float(score):.3f}" if score else '\u2014'
                pat_num   = _safe(em.get('patent_number', ''))
                feature   = _safe(em.get('feature_label', em.get('feature', '')), 50)
                claim_el  = _safe(em.get('element_text', em.get('claim_element', '')), 60)
                div_flag  = em.get('divergence_flag', False)

                row = [
                    _para(feature, small_style),
                    _para(claim_el, small_style),
                    _para(pat_num, small_style),
                    score_s,
                    gem_conf,
                    overall,
                ]
                cmp_rows.append(row)

                if div_flag:
                    divergences.append(
                        f"{pat_num}: {feature[:60]} \u2014 embedding vs AI disagreement"
                    )

            cmp_tbl = Table(
                cmp_rows,
                colWidths=[1.2 * inch, 1.8 * inch, 1.0 * inch,
                           0.8 * inch, 0.7 * inch, 0.8 * inch],
                repeatRows=1,
            )
            style_cmds = list(BASE_TABLE_STYLE)
            for r_idx, em in enumerate(cmat, start=1):
                level = em.get('overall_confidence', 'LOW')
                bg    = CONF_COLORS.get(level, WHITE)
                style_cmds.append(('BACKGROUND', (0, r_idx), (-1, r_idx), bg))
            cmp_tbl.setStyle(TableStyle(style_cmds))
            elements.append(cmp_tbl)

            if divergences:
                elements.append(Spacer(1, 12))
                elements.append(_para('<b>Key Divergences (manual review recommended):</b>', body_style))
                for d in divergences:
                    elements.append(_para(f'\u26a0  {d}', bullet_style))
        else:
            elements.append(_para(
                'Comparison matrix data not available. Run the full analysis to populate this section.',
                body_style,
            ))
        elements.append(PageBreak())

        # ================================================================== #
        #  SECTION 5 — Match Detail Pages                                     #
        # ================================================================== #
        elements.append(_para('5. Match Details', heading_style))
        elements.append(_para(
            'This section provides detailed explanations for each high-confidence '
            'and notable moderate-confidence match identified during analysis.',
            body_style,
        ))
        elements.append(Spacer(1, 8))

        # Sort: HIGH first, then MODERATE, then LOW; cap at 20
        def _sort_key(m):
            order = {'HIGH': 0, 'MODERATE': 1, 'LOW': 2}
            return (order.get(m.get('overall_confidence', m.get('similarity_level', 'LOW')), 2),
                    -float(m.get('similarity_score', 0)))
        sorted_matches = sorted(matches, key=_sort_key)[:20]

        if not sorted_matches:
            elements.append(_para('No matches to display.', body_style))

        for mi, m in enumerate(sorted_matches, start=1):
            block: list = []
            fl = safe_text_for_pdf(m.get('feature_label', ''), 'Unknown Feature')
            fd = safe_text_for_pdf(m.get('feature_description', ''))

            conf = m.get('overall_confidence', m.get('similarity_level', 'LOW'))
            conf_color_hex = {'HIGH': '#E74C3C', 'MODERATE': '#F39C12', 'LOW': '#27AE60'}.get(conf, '#666')

            block.append(Paragraph(
                f'<b>Match {mi}:</b> {_safe(fl)}',
                feature_header_style,
            ))
            if fd and fd != 'N/A':
                block.append(_para(f'<i>{_safe(fd, 200)}</i>', small_style))
            block.append(Spacer(1, 4))

            # Patent metadata
            pat_num = _safe(m.get('patent_number', ''))
            pat_title = safe_text_for_pdf(m.get('patent_title', ''))
            pat_asgn = safe_text_for_pdf(m.get('patent_assignee', ''))
            pat_date = _safe(str(m.get('publication_date', ''))[:10])
            url = format_google_patent_url(m.get('patent_number', ''))
            block.append(Paragraph(
                f'<b>Patent:</b> {_safe(pat_num)} &mdash; '
                f'<font color="#3498DB"><link href="{url}">{_safe(url, 50)}</link></font>',
                body_style,
            ))
            if pat_title and pat_title != 'N/A':
                block.append(_para(f'<b>Title:</b> {_safe(pat_title, 120)}', body_style))
            if pat_asgn and pat_asgn != 'N/A':
                block.append(_para(f'<b>Assignee:</b> {_safe(pat_asgn, 80)}', body_style))
            if pat_date:
                block.append(_para(f'<b>Date:</b> {pat_date}', body_style))
            block.append(Spacer(1, 6))

            # Full claim element text
            elem_text = safe_text_for_pdf(m.get('element_text', ''), '[claim text not available]')
            block.append(_para('<b>Claim Element:</b>', small_bold))
            block.append(_para(_safe(elem_text, 500), mono_style))
            block.append(Spacer(1, 4))

            # Highlighted snippet
            snippet = highlight_snippet(elem_text, _kw_strings[:10], max_len=160)
            if snippet:
                block.append(_para('<b>Snippet (key terms highlighted):</b>', small_bold))
                block.append(Paragraph(_safe(snippet), body_style))
                block.append(Spacer(1, 4))

            # Embedding score + bar
            score = float(m.get('similarity_score', 0))
            score_label = f"{score:.3f}"
            level = _safe(m.get('similarity_level', conf))
            block.append(Paragraph(
                f'<b>Embedding Similarity:</b> {score_label} ({level})',
                body_style,
            ))
            block.append(_render_score_bar(score))
            block.append(Spacer(1, 6))

            # Gemini explanation & assessment
            gem_exp = safe_text_for_pdf(m.get('gemini_explanation', ''))
            gem_ass = safe_text_for_pdf(m.get('gemini_assessment', ''))
            if gem_exp and gem_exp != 'N/A':
                block.append(_para('<b>Gemini Explanation:</b>', small_bold))
                block.append(_para(_safe(gem_exp, 600), body_style))
                block.append(Spacer(1, 4))
            if gem_ass and gem_ass != 'N/A':
                block.append(_para('<b>Gemini Assessment:</b>', small_bold))
                block.append(_para(_safe(gem_ass, 600), body_style))
                block.append(Spacer(1, 4))

            # Key distinctions
            distinctions = m.get('key_distinctions', [])
            if distinctions:
                block.append(_para('<b>Key Distinctions:</b>', small_bold))
                for d in distinctions[:6]:
                    block.append(_para(f'\u2022 {_safe(d, 200)}', bullet_style))
                block.append(Spacer(1, 4))

            # Cannot determine
            cd = safe_text_for_pdf(m.get('cannot_determine', ''))
            if cd and cd != 'N/A':
                block.append(_para(f'<b>Cannot Determine:</b> {_safe(cd, 300)}', body_style))
                block.append(Spacer(1, 4))

            # Divergence flag
            if m.get('divergence_flag'):
                div_note = safe_text_for_pdf(m.get('divergence_note', ''))
                block.append(Paragraph(
                    f'<font color="#E74C3C"><b>\u26a0 Divergence Flag:</b></font> '
                    f'Embedding and LLM scores disagree. {_safe(div_note, 200)}',
                    body_style,
                ))
                block.append(Spacer(1, 4))

            # Automated recommendation
            if conf == 'HIGH' and not m.get('divergence_flag'):
                rec = "Potential overlap \u2014 manual review recommended."
            elif m.get('divergence_flag'):
                rec = "Divergence between embedding and LLM \u2014 manual claim construction review recommended."
            else:
                rec = "No strong overlap detected; consider further targeted searching."
            block.append(Paragraph(
                f'<b>Recommendation:</b> {rec}',
                recommendation_style,
            ))

            block.append(Spacer(1, 10))
            elements.append(KeepTogether(block))

        elements.append(PageBreak())

        # ================================================================== #
        #  SECTION 6 — Patent Landscape                                        #
        # ================================================================== #
        elements.append(_para('6. Patent Landscape', heading_style))

        chart_images = session_data.get('chart_images') or {}
        if chart_images:
            for name, img_bytes in chart_images.items():
                if img_bytes:
                    try:
                        img = Image(BytesIO(img_bytes), width=6 * inch, height=3 * inch)
                        elements.append(img)
                        elements.append(_para(_safe(name).replace('_', ' ').title(), small_style))
                        elements.append(Spacer(1, 12))
                    except Exception:
                        elements.append(_para(
                            f'[Chart "{_safe(name)}" could not be embedded]', small_style
                        ))
        else:
            elements.append(_para(
                'Landscape visualisations are available in the interactive dashboard. '
                'PNG export was not available at the time this report was generated.',
                body_style,
            ))
        elements.append(PageBreak())

        # ================================================================== #
        #  SECTION 7 — White Space Analysis                                    #
        # ================================================================== #
        elements.append(_para('7. White Space Analysis', heading_style))
        elements.append(_para(
            '<i>Note: White space findings are based on automated pattern detection '
            'and do not constitute a freedom-to-operate opinion.</i>',
            small_style,
        ))
        elements.append(Spacer(1, 8))

        if white_spaces:
            for i, ws in enumerate(white_spaces, start=1):
                area       = _safe(ws.get('area', ws.get('opportunity', '')), 120)
                confidence = _safe(ws.get('confidence', ''))
                rationale  = _safe(ws.get('rationale', ws.get('description', '')), 300)

                ws_block = [
                    _para(f'<b>{i}. {area}</b>', heading2_style),
                    _para(f'Confidence: <b>{confidence}</b>', small_style),
                ]
                if rationale:
                    ws_block.append(_para(rationale, body_style))
                ws_block.append(Spacer(1, 4))
                elements.append(KeepTogether(ws_block))
        else:
            elements.append(_para('No white space opportunities were identified in this run.', body_style))
        elements.append(PageBreak())

        # ================================================================== #
        #  SECTION 8 — Recommended Next Steps                                  #
        # ================================================================== #
        elements.append(_para('8. Recommended Next Steps', heading_style))

        cpc_str = ', '.join(
            (c['code'] if isinstance(c, dict) else str(c))
            for c in cpc_codes[:6]
        ) or 'see methodology section'

        next_steps = [
            f'Professional prior art search focusing on CPC codes: {cpc_str}',
            'Attorney review of the top-scoring patents for detailed claim construction analysis',
            'Investigation of identified white space areas through expanded search',
            'International patent search (this analysis primarily covered US patents)',
            'Non-patent literature search (academic papers, products, standards documents)',
        ]
        for i, step in enumerate(next_steps, 1):
            elements.append(_para(f'{i}. {_safe(step)}', body_style))
        elements.append(PageBreak())

        # ================================================================== #
        #  SECTION 9 — References                                              #
        # ================================================================== #
        elements.append(_para('9. References', heading_style))
        elements.append(_para(
            'All patents retrieved from the Google Patents Public Dataset '
            '(patents-public-data.patents.publications) via Google BigQuery.',
            body_style,
        ))
        elements.append(Spacer(1, 12))

        if not detail_df.empty:
            for idx, (_, row) in enumerate(detail_df.iterrows(), start=1):
                pub_num = _safe(row.get('publication_number', ''))
                title   = _safe(row.get('title', 'Untitled'), 100)
                asgn    = row.get('assignee_name', '')
                if isinstance(asgn, list):
                    asgn = '; '.join(_safe(a) for a in asgn[:2])
                else:
                    asgn = _safe(asgn, 60)
                url = format_google_patent_url(pub_num)
                ref_text = (
                    f'[{idx}] <b>{pub_num}</b> \u2014 {title}. '
                    f'{asgn}. '
                    f'<font size="8" color="#3498DB"><link href="{url}">{_safe(url, 50)}</link></font>'
                )
                elements.append(_para(ref_text, small_style))
                elements.append(Spacer(1, 2))
        else:
            elements.append(_para('No patent references to list.', body_style))

        # ================================================================== #
        #  Build PDF with page numbers                                         #
        # ================================================================== #
        doc.build(elements, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
        buffer.seek(0)
        return buffer.getvalue()
