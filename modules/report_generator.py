"""
PatentScout — Report Generator Module

Builds a multi-section downloadable PDF analysis report using ReportLab.
All session data produced by the analysis pipeline is accepted via a single
`session_data` dict so the caller (app.py) does not have to know internals.
"""

from __future__ import annotations

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
    CONF_COLORS, BASE_TABLE_STYLE,
    title_style, subtitle_style, date_style,
    heading_style, heading2_style,
    body_style, bullet_style, small_style, disclaimer_style,
)


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

        # ------------------------------------------------------------------ #
        # PAGE 1 — Title Page + Disclaimer                                    #
        # ------------------------------------------------------------------ #
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

        # ------------------------------------------------------------------ #
        # PAGE 2 — Executive Summary                                          #
        # ------------------------------------------------------------------ #
        elements.append(_para('1. Executive Summary', heading_style))

        strategy     = session_data.get('search_strategy') or {}
        detail_df    = session_data.get('detail_patents')
        sim_results  = session_data.get('similarity_results') or {}
        white_spaces = session_data.get('white_spaces') or []
        cmat         = session_data.get('comparison_matrix') or []

        n_patents   = len(detail_df) if detail_df is not None else 0
        stats       = sim_results.get('stats', {})
        high_count  = stats.get('high_matches', 0)
        mod_count   = stats.get('moderate_matches', 0)
        ws_count    = len(white_spaces)

        features = strategy.get('features', [])
        concept_overview = (
            session_data.get('invention_text', '') or
            ', '.join(features[:5]) or
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

        # ------------------------------------------------------------------ #
        # PAGE 3 — Search Methodology                                         #
        # ------------------------------------------------------------------ #
        elements.append(_para('2. Search Methodology', heading_style))

        cpc_codes = strategy.get('cpc_codes', [])
        keywords  = strategy.get('keywords', [])
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
                    elements.append(_para(f'• {_safe(code)} — {_safe(desc, 120)}', bullet_style))
                else:
                    elements.append(_para(f'• {_safe(code_item)}', bullet_style))

        if keywords:
            elements.append(Spacer(1, 8))
            elements.append(_para('<b>Keyword terms used:</b>', body_style))
            elements.append(_para(', '.join(_safe(k) for k in keywords[:30]), body_style))

        if queries:
            elements.append(Spacer(1, 8))
            elements.append(_para('<b>Search queries executed:</b>', body_style))
            for q in queries[:5]:
                elements.append(_para(f'• {_safe(q, 200)}', bullet_style))
        elements.append(PageBreak())

        # ------------------------------------------------------------------ #
        # PAGES 4-5 — Prior Art Table                                         #
        # ------------------------------------------------------------------ #
        elements.append(_para('3. Prior Art Summary', heading_style))

        if detail_df is not None and not detail_df.empty:
            prior_art_rows = [['#', 'Patent Number', 'Title', 'Assignee', 'Date', 'Score']]
            for idx, (_, row) in enumerate(detail_df.head(30).iterrows(), start=1):
                pub_num  = _safe(row.get('publication_number', ''))
                title    = _safe(row.get('title', 'Untitled'), 60)
                asgn     = row.get('assignee_name', '')
                if isinstance(asgn, list):
                    asgn = '; '.join(_safe(a) for a in asgn[:2])
                else:
                    asgn = _safe(asgn, 40)
                date_val = str(row.get('publication_date', ''))[:4]
                score    = row.get('relevance_score', 0)
                score_s  = f"{float(score):.3f}" if score else '—'
                url      = _safe(row.get('patent_url', ''))
                # Stack patent number + URL
                num_cell = Paragraph(
                    f'<b>{pub_num}</b><br/><font size="7" color="#3498DB">{url}</font>',
                    small_style,
                )
                prior_art_rows.append([
                    str(idx),
                    num_cell,
                    _para(title, small_style),
                    _para(asgn, small_style),
                    date_val,
                    score_s,
                ])

            prior_art_tbl = Table(
                prior_art_rows,
                colWidths=[0.3 * inch, 1.2 * inch, 2.1 * inch, 1.5 * inch,
                           0.6 * inch, 0.8 * inch],
                repeatRows=1,
            )
            prior_art_tbl.setStyle(TableStyle(BASE_TABLE_STYLE))
            elements.append(prior_art_tbl)
        else:
            elements.append(_para('No patent data available.', body_style))
        elements.append(PageBreak())

        # ------------------------------------------------------------------ #
        # PAGES 6-7 — Claim Element Comparison Matrix                         #
        # ------------------------------------------------------------------ #
        elements.append(_para('4. Claim Element Comparison', heading_style))

        if cmat:
            cmp_rows = [['Feature', 'Claim Element', 'Patent #',
                         'Embed Score', 'AI Conf.', 'Overall']]
            divergences = []
            for em in cmat:
                overall   = _safe(em.get('overall_confidence', 'LOW'))
                gem_conf  = _safe(em.get('gemini_confidence', '—'))
                score     = em.get('similarity_score', 0.0)
                score_s   = f"{float(score):.3f}" if score else '—'
                pat_num   = _safe(em.get('patent_number', ''))
                feature   = _safe(em.get('feature', ''), 50)
                claim_el  = _safe(em.get('claim_element', ''), 60)
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
                        f"{pat_num}: {feature[:60]} — embedding vs AI disagreement"
                    )

            cmp_tbl = Table(
                cmp_rows,
                colWidths=[1.2 * inch, 1.8 * inch, 1.0 * inch,
                           0.8 * inch, 0.7 * inch, 0.8 * inch],
                repeatRows=1,
            )
            style_cmds = list(BASE_TABLE_STYLE)
            # Colour-code rows by confidence
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
                    elements.append(_para(f'⚠  {d}', bullet_style))
        else:
            elements.append(_para(
                'Comparison matrix data not available. Run the full analysis to populate this section.',
                body_style,
            ))
        elements.append(PageBreak())

        # ------------------------------------------------------------------ #
        # PAGE 8 — Patent Landscape                                           #
        # ------------------------------------------------------------------ #
        elements.append(_para('5. Patent Landscape', heading_style))

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

        # ------------------------------------------------------------------ #
        # PAGE 9 — White Space Analysis                                       #
        # ------------------------------------------------------------------ #
        elements.append(_para('6. White Space Analysis', heading_style))
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
                conf_color = {
                    'HIGH': SUCCESS, 'MODERATE': WARNING,
                }.get(confidence.upper(), SECONDARY)

                block = [
                    _para(f'<b>{i}. {area}</b>', heading2_style),
                    _para(f'Confidence: <b>{confidence}</b>', small_style),
                ]
                if rationale:
                    block.append(_para(rationale, body_style))
                block.append(Spacer(1, 4))
                elements.append(KeepTogether(block))
        else:
            elements.append(_para('No white space opportunities were identified in this run.', body_style))
        elements.append(PageBreak())

        # ------------------------------------------------------------------ #
        # PAGE 10 — Recommended Next Steps                                    #
        # ------------------------------------------------------------------ #
        elements.append(_para('7. Recommended Next Steps', heading_style))

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

        # ------------------------------------------------------------------ #
        # FINAL PAGE — References                                             #
        # ------------------------------------------------------------------ #
        elements.append(_para('8. References', heading_style))
        elements.append(_para(
            'All patents retrieved from the Google Patents Public Dataset '
            '(patents-public-data.patents.publications) via Google BigQuery.',
            body_style,
        ))
        elements.append(Spacer(1, 12))

        if detail_df is not None and not detail_df.empty:
            for idx, (_, row) in enumerate(detail_df.iterrows(), start=1):
                pub_num = _safe(row.get('publication_number', ''))
                title   = _safe(row.get('title', 'Untitled'), 100)
                asgn    = row.get('assignee_name', '')
                if isinstance(asgn, list):
                    asgn = '; '.join(_safe(a) for a in asgn[:2])
                else:
                    asgn = _safe(asgn, 60)
                url = _safe(row.get('patent_url', f'https://patents.google.com/patent/{pub_num}'))
                ref_text = (
                    f'[{idx}] <b>{pub_num}</b> — {title}. '
                    f'{asgn}. '
                    f'<font size="8" color="#3498DB">{url}</font>'
                )
                elements.append(_para(ref_text, small_style))
                elements.append(Spacer(1, 2))
        else:
            elements.append(_para('No patent references to list.', body_style))

        # ------------------------------------------------------------------ #
        # Build PDF                                                            #
        # ------------------------------------------------------------------ #
        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()
