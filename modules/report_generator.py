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
    format_patent_date,
    format_patent_year,
    group_matches_by_patent,
    highlight_snippet,
    safe_text_for_pdf,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sanitisation helper — strips characters that break ReportLab's latin-1
# codec fallback while preserving readability.
# ---------------------------------------------------------------------------

_UNICODE_REPLACEMENTS = {
    '\u2192': '-',    # →
    '\u2190': '-',    # ←
    '\u2014': '--',   # —
    '\u2013': '-',    # –
    '\u2018': "'",    # '
    '\u2019': "'",    # '
    '\u201c': '"',    # \u201c
    '\u201d': '"',    # \u201d
    '\u2022': '*',    # •
    '\u25b8': '-',    # ▸
    '\u25ba': '-',    # ►
    '\u00bb': '-',    # »
    '\u00ab': '-',    # «
    '\u2026': '...',  # …
    '\u26a0': '[!]',  # ⚠
}


def _sanitize_for_pdf(text: str) -> str:
    """Replace Unicode characters that may not render in standard PDF fonts."""
    for char, replacement in _UNICODE_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    return text


def _safe(text: Any, maxlen: int = 0) -> str:
    """Return a ReportLab-safe string, optionally truncated."""
    s = str(text) if text is not None else ''
    # Replace known Unicode characters with ASCII equivalents first
    s = _sanitize_for_pdf(s)
    # encode/decode to strip remaining non-latin-1 chars
    s = s.encode('latin-1', 'replace').decode('latin-1')
    if maxlen and len(s) > maxlen:
        s = s[:maxlen] + '...'
    return s


def clean_concept_text(raw_text: str) -> str:
    """
    Remove test-harness metadata, formatting artifacts, and excessive
    length from a concept overview destined for the executive summary.
    """
    lines = raw_text.split('\n')
    clean_lines: list[str] = []
    for line in lines:
        line = line.strip()
        # Skip divider lines (===, ---, ***, etc.)
        if line and set(line) <= {'=', '-', '*', ' ', '?', '\u2014'}:
            continue
        # Skip lines that are ALL CAPS and short (likely headers)
        if line.isupper() and len(line) < 80:
            continue
        # Skip known metadata prefixes
        if any(line.lower().startswith(prefix) for prefix in [
            'title:', 'description:', 'key technical', 'test case',
            'patentscout', '===', '---',
        ]):
            continue
        if line:
            clean_lines.append(line)
    text = ' '.join(clean_lines)
    if len(text) > 500:
        text = text[:497] + '...'
    return text


def _para(text: Any, style=None, maxlen: int = 0) -> Paragraph:
    """Convenience: build a Paragraph with sanitised text."""
    if style is None:
        style = body_style
    return Paragraph(_safe(text, maxlen), style)



# Score-bar flowable helper
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



# Page-number callback
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



# Consistency helper — fill missing fields in matches
def _normalise_matches(session_data: dict) -> list[dict]:
    """
    Return the matches list with every expected key guaranteed present.
    Prefers enriched comparison_matrix matches (Gemini analysis) over raw
    similarity_results matches. Also deduplicates by (feature+patent+element).
    """
    # Prefer comparison_matrix (enriched with Gemini) if available
    cmat = session_data.get('comparison_matrix') or []
    sim = session_data.get('similarity_results') or {}
    raw_matches = list(sim.get('matches', []))

    # Build enriched lookup from comparison_matrix
    _enriched_lookup: dict[tuple, dict] = {}
    for cm in cmat:
        key = (
            cm.get('feature_label', ''),
            cm.get('patent_number', ''),
            cm.get('element_id', ''),
        )
        _enriched_lookup[key] = cm

    # Merge: start with raw matches, overlay enriched data
    matches = []
    for m in raw_matches:
        key = (
            m.get('feature_label', ''),
            m.get('patent_number', ''),
            m.get('element_id', ''),
        )
        if key in _enriched_lookup:
            # Use the enriched version, preserving any extra fields from raw
            merged = dict(m)
            merged.update(_enriched_lookup[key])
            matches.append(merged)
        else:
            matches.append(dict(m))

    # Add any enriched entries not in raw matches
    raw_keys = {
        (m.get('feature_label', ''), m.get('patent_number', ''), m.get('element_id', ''))
        for m in raw_matches
    }
    for key, cm in _enriched_lookup.items():
        if key not in raw_keys:
            matches.append(dict(cm))

    # Deduplicate: one entry per (feature_label, patent_number, element_text[:100])
    _dedup_seen: set = set()
    _dedup_matches: list[dict] = []
    for m in matches:
        dedup_key = (
            m.get('feature_label', ''),
            m.get('patent_number', ''),
            str(m.get('element_text', ''))[:100].strip().lower(),
        )
        if dedup_key not in _dedup_seen:
            _dedup_seen.add(dedup_key)
            _dedup_matches.append(m)
    matches = _dedup_matches

    # Build a quick lookup: publication_number -> detail patent dict
    import pandas as pd
    _raw = session_data.get('detail_patents')
    if isinstance(_raw, pd.DataFrame):
        detail_patents = _raw.to_dict('records')
    elif isinstance(_raw, list):
        detail_patents = _raw
    else:
        detail_patents = []
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
        # element_mapper returns key_distinctions (list),
        # but cached/CM data may use gemini_distinction (string).
        if 'gemini_distinction' in m and 'key_distinctions' not in m:
            raw_dist = m['gemini_distinction']
            if isinstance(raw_dist, str) and raw_dist.strip():
                m['key_distinctions'] = [raw_dist.strip()]
            elif isinstance(raw_dist, list):
                m['key_distinctions'] = raw_dist
            else:
                m['key_distinctions'] = []

        # Derive overall_confidence when missing.
        # Prefer similarity_level (already set by embedding engine).
        if 'overall_confidence' not in m:
            sim_level = m.get('similarity_level', '')
            gem_ass = str(m.get('gemini_assessment', '')).upper()
            if sim_level in ('HIGH', 'MODERATE', 'LOW'):
                m['overall_confidence'] = sim_level
            elif 'HIGH' in gem_ass:
                m['overall_confidence'] = 'HIGH'
            elif 'MODERATE' in gem_ass:
                m['overall_confidence'] = 'MODERATE'
            else:
                m['overall_confidence'] = 'LOW'

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

        # Resolve patent_title / patent_assignee / publication_date from detail_patents
        pat = pat_lookup.get(m.get('patent_number', ''))
        if pat:
            if not m.get('patent_title'):
                m['patent_title'] = pat.get('title', '')
            if not m.get('patent_assignee'):
                asgn = pat.get('assignee_name', '')
                if isinstance(asgn, list):
                    asgn = '; '.join(str(a) for a in asgn[:2])
                m['patent_assignee'] = str(asgn)
            if not m.get('publication_date'):
                m['publication_date'] = str(pat.get('publication_date', ''))

    return matches



# ReportGenerator
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
        import pandas as pd

        # Deduplicate comparison_matrix as a safety net
        raw_cmat = session_data.get('comparison_matrix') or []
        if raw_cmat:
            _cm_seen: set = set()
            _cm_unique: list = []
            for _cm in raw_cmat:
                _cm_key = (
                    _cm.get('feature_label', ''),
                    _cm.get('patent_number', ''),
                    _cm.get('element_id', ''),
                )
                if _cm_key not in _cm_seen:
                    _cm_seen.add(_cm_key)
                    _cm_unique.append(_cm)
            session_data = dict(session_data)
            session_data['comparison_matrix'] = _cm_unique

        strategy     = session_data.get('search_strategy') or {}
        _raw_detail  = session_data.get('detail_patents')
        # detail_patents may be a pandas DataFrame or a list of dicts
        if isinstance(_raw_detail, pd.DataFrame):
            detail_df = _raw_detail
        elif isinstance(_raw_detail, list):
            detail_df = pd.DataFrame(_raw_detail) if _raw_detail else pd.DataFrame()
        else:
            detail_df = pd.DataFrame()

        sim_results  = session_data.get('similarity_results') or {}
        white_spaces = session_data.get('white_spaces') or []
        cmat         = session_data.get('comparison_matrix') or []
        # Normalise cmat field names (gemini_distinction → key_distinctions,
        # derive overall_confidence from similarity_level when missing).
        for _cm in cmat:
            if 'gemini_distinction' in _cm and 'key_distinctions' not in _cm:
                raw = _cm['gemini_distinction']
                _cm['key_distinctions'] = [raw.strip()] if isinstance(raw, str) and raw.strip() else (raw if isinstance(raw, list) else [])
            if 'overall_confidence' not in _cm:
                _cm['overall_confidence'] = _cm.get('similarity_level', 'LOW')
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
        # Count landscape patents for differentiation (Fix 9)
        _lp_for_count = session_data.get('landscape_patents')
        if isinstance(_lp_for_count, list):
            n_landscape = len(_lp_for_count)
        elif hasattr(_lp_for_count, '__len__'):
            n_landscape = len(_lp_for_count)
        else:
            n_landscape = n_patents
        stats       = sim_results.get('stats', {})
        high_count  = stats.get('high_matches', 0)
        mod_count   = stats.get('moderate_matches', 0)
        ws_count    = len(white_spaces)

        # page 1 — title page + disclaimer

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
            ('BOX',           (0, 0), (-1, -1), 1.5, PRIMARY),
            ('BACKGROUND',    (0, 0), (-1, -1), LIGHT_GRAY),
            ('TOPPADDING',    (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('LEFTPADDING',   (0, 0), (-1, -1), 12),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 12),
        ]))
        elements.append(disclaimer_table)
        elements.append(PageBreak())

        # table of contents

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

        # section 1 — executive summary

        elements.append(_para('1. Executive Summary', heading_style))

        concept_overview = clean_concept_text(
            session_data.get('invention_text', '') or
            ', '.join(
                (f.get('label', str(f)) if isinstance(f, dict) else str(f))
                for f in features[:5]
            ) or
            'No description provided.'
        )

        exec_summary = (
            f"PatentScout retrieved and analysed <b>{n_patents}</b> patents in detail from the "
            f"Google BigQuery Patents database relevant to the submitted invention "
            f"description (from a broader landscape of <b>{n_landscape}</b> patents). "
            f"Automated similarity scoring identified "
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

        # section 2 — search methodology

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

        # Multi-modal input indicator
        _sketch_used = session_data.get('sketch_used', False)
        if _sketch_used:
            elements.append(_para(
                '<b>Input type:</b> Text description + design sketch '
                '(multi-modal analysis)',
                body_style,
            ))
            elements.append(Spacer(1, 4))
            # Embed sketch thumbnail if available
            _sketch_bytes = session_data.get('invention_image')
            if _sketch_bytes:
                try:
                    from PIL import Image as PILImage
                    _img = PILImage.open(BytesIO(_sketch_bytes))
                    _img.thumbnail((200, 300))
                    _thumb_buf = BytesIO()
                    _img.save(_thumb_buf, format='JPEG')
                    _thumb_buf.seek(0)
                    elements.append(Image(_thumb_buf, width=150, height=150))
                    elements.append(_para(
                        '<i>Design sketch provided by inventor</i>',
                        small_style,
                    ))
                    elements.append(Spacer(1, 8))
                except Exception as _img_exc:
                    logger.warning('Could not embed sketch in PDF: %s', _img_exc)
        else:
            elements.append(_para(
                '<b>Input type:</b> Text description',
                body_style,
            ))
            elements.append(Spacer(1, 4))

        # Feature source annotations for sketch-sourced features
        _sketch_features = [
            f for f in features
            if isinstance(f, dict) and f.get('source') == 'sketch'
        ]
        if _sketch_features:
            elements.append(_para(
                '<b>Features identified from visual input:</b>',
                body_style,
            ))
            for _sf in _sketch_features:
                elements.append(_para(
                    f'- {_safe(_sf.get("label", ""), 80)} '
                    f'<i>(identified from sketch)</i>',
                    bullet_style,
                ))
            elements.append(Spacer(1, 8))

        if cpc_codes:
            elements.append(_para('<b>CPC classification codes searched:</b>', body_style))
            for code_item in cpc_codes:
                if isinstance(code_item, dict):
                    code = code_item.get('code', '')
                    desc = code_item.get('description', '')
                    elements.append(_para(f'- {_safe(code)} - {_safe(desc, 120)}', bullet_style))
                else:
                    elements.append(_para(f'- {_safe(code_item)}', bullet_style))

        if keywords:
            elements.append(Spacer(1, 8))
            elements.append(_para('<b>Keyword terms used:</b>', body_style))
            elements.append(_para(', '.join(_safe(k) for k in _kw_strings[:30]), body_style))

        if queries:
            elements.append(Spacer(1, 8))
            elements.append(_para('<b>Search queries executed:</b>', body_style))
            for q in queries[:5]:
                elements.append(_para(f'- {_safe(q, 200)}', bullet_style))
        elements.append(Spacer(1, 8))
        elements.append(_para(
            '<b>Patent types:</b> Utility patents only. Design patents '
            '(ornamental appearance) and plant patents were excluded from '
            'analysis as they do not contain functional claims.',
            body_style,
        ))
        elements.append(PageBreak())

        # section 3 — prior art summary table (top 20)

        elements.append(_para('3. Prior Art Summary', heading_style))

        if not detail_df.empty:
            snippet_terms = _kw_strings[:12]

            # Build patent -> best_score and best_snippet lookups from similarity_results
            _sim_results = sim_results
            _patent_best_scores: dict[str, float] = {}
            _patent_best_snippet: dict[str, str] = {}
            for _m in _sim_results.get("matches", []):
                _pn = _m.get("patent_number", "")
                _sc = _m.get("similarity_score", 0)
                if _sc > _patent_best_scores.get(_pn, 0):
                    _patent_best_scores[_pn] = _sc
                    _patent_best_snippet[_pn] = _m.get("element_text", "")[:80]

            prior_art_rows = [['#', 'Patent', 'Title', 'Assignee', 'Date', 'Score', 'Claim Snippet']]
            for idx, (_, row) in enumerate(detail_df.head(20).iterrows(), start=1):
                pub_num  = _safe(row.get('publication_number', ''))
                title    = _safe(row.get('title', 'Untitled'), 50)
                asgn     = row.get('assignee_name', '')
                if isinstance(asgn, list):
                    asgn = '; '.join(_safe(a) for a in asgn[:2])
                else:
                    asgn = _safe(asgn, 35)
                date_val = format_patent_year(row.get('publication_date'))
                score    = row.get('relevance_score', 0)
                if not score:
                    score = _patent_best_scores.get(row.get('publication_number', ''), 0)
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
                if not snippet_html or snippet_html.strip() == '':
                    snippet_html = _patent_best_snippet.get(
                        row.get('publication_number', ''), ''
                    )[:80]
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

        # section 4 — claim element comparison matrix

        elements.append(_para('4. Claim Element Comparison', heading_style))

        if cmat:
            # Deduplicate: group matches by patent for a cleaner table
            grouped_cmat = group_matches_by_patent(cmat)
            cmp_rows = [['Patent #', 'Best Feature', '# Features',
                         'Best Score', 'AI Conf.', 'Overall']]
            divergences = []
            for gm in grouped_cmat:
                overall   = _safe(gm.get('overall_confidence', 'LOW'))
                # Derive AI confidence label from gemini_assessment text
                _ga = str(gm.get('gemini_assessment', '')).upper()
                if gm.get('gemini_confidence'):
                    gem_conf = _safe(gm['gemini_confidence'])
                elif 'HIGH' in _ga:
                    gem_conf = 'HIGH'
                elif 'MODERATE' in _ga:
                    gem_conf = 'MODERATE'
                elif 'LOW' in _ga:
                    gem_conf = 'LOW'
                else:
                    gem_conf = '\u2014'
                best_sc   = gm.get('best_score', 0.0)
                score_s   = f"{float(best_sc):.3f}" if best_sc else '\u2014'
                pat_num   = _safe(gm.get('patent_number', ''))
                feature   = _safe(gm.get('feature_label', gm.get('feature', '')), 50)
                n_feats   = str(gm.get('feature_count', 1))
                div_flag  = gm.get('divergence_flag', False)

                row = [
                    _para(pat_num, small_style),
                    _para(feature, small_style),
                    n_feats,
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
                colWidths=[1.1 * inch, 1.8 * inch, 0.6 * inch,
                           0.8 * inch, 0.7 * inch, 0.8 * inch],
                repeatRows=1,
            )
            style_cmds = list(BASE_TABLE_STYLE)
            for r_idx, gm in enumerate(grouped_cmat, start=1):
                level = gm.get('overall_confidence', 'LOW')
                bg    = CONF_COLORS.get(level, WHITE)
                style_cmds.append(('BACKGROUND', (0, r_idx), (-1, r_idx), bg))
            cmp_tbl.setStyle(TableStyle(style_cmds))
            elements.append(cmp_tbl)

            if divergences:
                elements.append(Spacer(1, 12))
                elements.append(_para('<b>Key Divergences (manual review recommended):</b>', body_style))
                for d in divergences:
                    elements.append(_para(f'\u26a0  {d}', bullet_style))
        elif matches:
            # Fallback: build comparison table from normalised matches
            # when comparison_matrix is empty (e.g. doorbell / short runs).
            grouped_fb = group_matches_by_patent(matches)
            fb_rows = [['Patent #', 'Best Feature', '# Features',
                        'Best Score', 'Similarity', 'Overall']]
            for gm in grouped_fb[:20]:
                overall  = _safe(gm.get('overall_confidence', gm.get('similarity_level', 'LOW')))
                sim_lvl  = _safe(gm.get('similarity_level', overall))
                best_sc  = gm.get('best_score', 0.0)
                score_s  = f"{float(best_sc):.3f}" if best_sc else '\u2014'
                pat_num  = _safe(gm.get('patent_number', ''))
                feature  = _safe(gm.get('feature_label', ''), 50)
                n_feats  = str(gm.get('feature_count', 1))
                fb_rows.append([
                    _para(pat_num, small_style),
                    _para(feature, small_style),
                    n_feats, score_s, sim_lvl, overall,
                ])
            fb_tbl = Table(
                fb_rows,
                colWidths=[1.1 * inch, 1.8 * inch, 0.6 * inch,
                           0.8 * inch, 0.7 * inch, 0.8 * inch],
                repeatRows=1,
            )
            fb_style = list(BASE_TABLE_STYLE)
            for r_idx, gm in enumerate(grouped_fb[:20], start=1):
                level = gm.get('overall_confidence', gm.get('similarity_level', 'LOW'))
                bg    = CONF_COLORS.get(level, WHITE)
                fb_style.append(('BACKGROUND', (0, r_idx), (-1, r_idx), bg))
            fb_tbl.setStyle(TableStyle(fb_style))
            elements.append(fb_tbl)
            elements.append(Spacer(1, 6))
            elements.append(_para(
                '<i>Note: Full AI-enriched comparison matrix was not available. '
                'Table above is based on embedding similarity matches.</i>',
                small_style,
            ))
        else:
            elements.append(_para(
                'Comparison matrix data not available. Run the full analysis to populate this section.',
                body_style,
            ))
        elements.append(PageBreak())

        # section 5 — match detail pages

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
        sorted_matches = sorted(matches, key=_sort_key)

        # Limit per-feature: max 3 patents per feature_label to avoid
        # the same feature dominating the match details section.
        # Also dedup by (feature_label, patent_number).
        _sec5_seen_fp: set = set()
        _sec5_feat_count: dict = {}
        _sec5_unique: list = []
        for m in sorted_matches:
            fl = m.get('feature_label', '')
            pn = m.get('patent_number', '')
            fp_key = (fl, pn)
            if fp_key in _sec5_seen_fp:
                continue  # exact duplicate
            _sec5_seen_fp.add(fp_key)
            _sec5_feat_count[fl] = _sec5_feat_count.get(fl, 0) + 1
            if _sec5_feat_count[fl] > 3:
                continue  # too many patents for this feature
            _sec5_unique.append(m)
        sorted_matches = _sec5_unique[:20]

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
            pat_date = format_patent_date(m.get('publication_date', ''))
            url = format_google_patent_url(m.get('patent_number', ''))
            block.append(Paragraph(
                f'<b>Patent:</b> {_safe(pat_num)} - '
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

            # Highlighted snippet — localised to the most relevant window
            from modules.element_mapper import ElementMapper
            localised = ElementMapper.localize_snippet(
                elem_text,
                safe_text_for_pdf(m.get('feature_label', ''), ''),
                safe_text_for_pdf(m.get('feature_description', ''), ''),
                max_len=200,
            )
            snippet = highlight_snippet(localised, _kw_strings[:10], max_len=200)
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

            # Gemini explanation & assessment (Fix 4: always show headers with fallback)
            gem_exp = safe_text_for_pdf(m.get('gemini_explanation', ''))
            gem_ass = safe_text_for_pdf(m.get('gemini_assessment', ''))
            block.append(_para('<b>What This Claim Requires:</b>', small_bold))
            if gem_exp and gem_exp != 'N/A':
                block.append(_para(_safe(gem_exp, 600), body_style))
            else:
                block.append(_para('Automated AI analysis was not available for this match. Manual review recommended.', body_style))
            block.append(Spacer(1, 4))
            block.append(_para('<b>Comparison Analysis:</b>', small_bold))
            if gem_ass and gem_ass != 'N/A':
                block.append(_para(_safe(gem_ass, 600), body_style))
            else:
                block.append(_para('Automated AI analysis was not available for this match. Manual review recommended.', body_style))
            block.append(Spacer(1, 4))

            # Key distinctions
            distinctions = m.get('key_distinctions', [])
            if distinctions:
                block.append(_para('<b>Key Technical Distinctions:</b>', small_bold))
                for d in distinctions[:6]:
                    block.append(_para(f'- {_safe(d, 200)}', bullet_style))
                block.append(Spacer(1, 4))

            # Cannot determine
            cd = safe_text_for_pdf(m.get('cannot_determine', ''))
            if cd and cd != 'N/A':
                block.append(_para(f'<b>Limitations of This Analysis:</b> {_safe(cd, 300)}', body_style))
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
                rec = "Potential overlap \u2014 manual claim construction review recommended."
            elif m.get('divergence_flag'):
                rec = "Divergence between embedding and LLM analysis \u2014 manual claim construction review recommended."
            elif conf == 'MODERATE':
                rec = "Partial overlap identified \u2014 further investigation recommended to assess scope of similarity."
            elif gem_exp and gem_exp != 'N/A':
                rec = "Review the contextual analysis above for specific technical comparison details."
            else:
                rec = "No strong overlap detected; consider further targeted searching."
            block.append(Paragraph(
                f'<b>Recommendation:</b> {rec}',
                recommendation_style,
            ))

            block.append(Spacer(1, 10))
            elements.append(KeepTogether(block))

        elements.append(PageBreak())

        # section 6 — patent landscape

        elements.append(_para('6. Patent Landscape', heading_style))

        chart_images = session_data.get('chart_images') or {}

        # If chart_images is empty but we have landscape data, generate
        # charts on-the-fly using matplotlib (no kaleido dependency).
        if not chart_images:
            _ls_raw = session_data.get('landscape_patents')
            if _ls_raw is not None:
                try:
                    import pandas as pd
                    from modules.landscape_analyzer import LandscapeAnalyzer
                    if isinstance(_ls_raw, pd.DataFrame):
                        _ls_df = _ls_raw
                    elif isinstance(_ls_raw, list) and _ls_raw:
                        _ls_df = pd.DataFrame(_ls_raw)
                    else:
                        _ls_df = pd.DataFrame()

                    if not _ls_df.empty and len(_ls_df) >= 3:
                        _analyzer = LandscapeAnalyzer(_ls_df)
                        chart_images = _analyzer.export_charts_as_images()
                        logger.info(
                            "Generated %d chart images on-the-fly for PDF",
                            len(chart_images),
                        )
                except Exception as _chart_exc:
                    logger.warning("On-the-fly chart generation failed: %s", _chart_exc)

        _chart_display_names = {
            "filing_trends": "Filing Trends",
            "top_assignees": "Top Patent Holders",
            "cpc_distribution": "Classification Distribution",
        }
        charts_embedded = 0
        if chart_images:
            for name, img_bytes in chart_images.items():
                if img_bytes and len(img_bytes) > 100:
                    try:
                        img = Image(BytesIO(img_bytes), width=6 * inch, height=3.5 * inch)
                        elements.append(img)
                        caption = _chart_display_names.get(name, _safe(name).replace('_', ' ').title())
                        elements.append(_para(caption, small_style))
                        elements.append(Spacer(1, 12))
                        charts_embedded += 1
                    except Exception as _img_exc:
                        logger.warning("Chart '%s' embed failed: %s", name, _img_exc)
                        elements.append(_para(
                            f'[Chart "{_safe(name)}" could not be embedded]', small_style
                        ))

        if charts_embedded == 0:
            elements.append(_para(
                'Landscape visualisations are available in the interactive dashboard.',
                body_style,
            ))
        elements.append(PageBreak())

        # section 7 — white space analysis

        elements.append(_para('7. White Space Analysis', heading_style))
        elements.append(_para(
            '<i>Note: White space findings are based on automated pattern detection '
            'and do not constitute a freedom-to-operate opinion.</i>',
            small_style,
        ))
        elements.append(Spacer(1, 8))

        if white_spaces:
            # Show corpus size from landscape data if available
            _lp_raw = session_data.get('landscape_patents')
            _corpus_n = 0
            if isinstance(_lp_raw, list):
                _corpus_n = len(_lp_raw)
            elif hasattr(_lp_raw, '__len__'):
                _corpus_n = len(_lp_raw)
            if _corpus_n:
                elements.append(_para(
                    f'White space analysis evaluated across <b>{_corpus_n}</b> patents '
                    f'from the broader landscape search, with <b>{n_patents}</b> patents '
                    f'receiving detailed claim-level analysis.',
                    body_style,
                ))
                elements.append(Spacer(1, 6))

            for i, ws in enumerate(white_spaces, start=1):
                ws_type    = _safe(ws.get('type', ''))
                area       = _safe(ws.get('title', ws.get('area', ws.get('opportunity', ''))), 120)
                conf_obj   = ws.get('confidence', {})
                if isinstance(conf_obj, dict):
                    conf_level = _safe(conf_obj.get('level', 'LOW'))
                    conf_rationale = _safe(conf_obj.get('rationale', ''), 300)
                else:
                    conf_level = _safe(conf_obj)
                    conf_rationale = ''
                description = _safe(ws.get('description', ws.get('rationale', '')), 300)
                data_note   = _safe(ws.get('data_completeness', ''), 200)

                ws_block = [
                    _para(f'<b>{i}. [{ws_type}] {area}</b>', heading2_style),
                    _para(f'Confidence: <b>{conf_level}</b>', small_style),
                ]
                if description:
                    ws_block.append(_para(description, body_style))
                if conf_rationale:
                    ws_block.append(_para(f'<i>{conf_rationale}</i>', small_style))
                if data_note:
                    ws_block.append(_para(f'<i>Data: {data_note}</i>', small_style))
                ws_block.append(Spacer(1, 4))
                elements.append(KeepTogether(ws_block))
        else:
            elements.append(_para('No white space opportunities were identified in this run.', body_style))
        elements.append(PageBreak())

        # section 8 — recommended next steps

        elements.append(_para('8. Recommended Next Steps', heading_style))

        cpc_str = ', '.join(
            (c['code'] if isinstance(c, dict) else str(c))
            for c in cpc_codes[:6]
        ) or 'see methodology section'

        # --- Analysis-driven recommendations ---
        specific_recs: list[str] = []

        # Recommendations based on high matches
        high_match_pats = [
            m.get('patent_number', '')
            for m in matches
            if m.get('overall_confidence', m.get('similarity_level', '')) == 'HIGH'
        ]
        if high_match_pats:
            unique_pats = list(dict.fromkeys(high_match_pats))[:5]
            specific_recs.append(
                f'<b>Design-Around Analysis:</b> {len(unique_pats)} patent(s) showed '
                f'high similarity ({", ".join(unique_pats[:3])}). Engage patent counsel '
                f'to perform claim construction and identify design-around opportunities.'
            )

        # Recommendations based on divergences
        div_matches = [m for m in matches if m.get('divergence_flag')]
        if div_matches:
            specific_recs.append(
                f'<b>Manual Review Required:</b> {len(div_matches)} match(es) show '
                f'divergence between embedding and AI analysis layers. These require '
                f'human expert review to resolve conflicting signals.'
            )

        # Recommendations based on white spaces
        if white_spaces:
            ws_titles = [ws.get('title', ws.get('area', ''))[:60] for ws in white_spaces[:3]]
            specific_recs.append(
                f'<b>White Space Investigation:</b> {len(white_spaces)} potential '
                f'innovation area(s) identified. Investigate through expanded manual '
                f'search before drawing conclusions about patentability: '
                f'{"; ".join(ws_titles)}.'
            )

        # Standard professional recommendations (research-oriented only, no legal advice)
        std_recs = [
            f'<b>Professional Prior Art Search:</b> Commission a comprehensive search '
            f'focusing on CPC codes: {cpc_str}.',
            '<b>International Search:</b> This analysis primarily covers US patents. '
            'Extend to EP, WO, CN, JP publications for global coverage.',
            '<b>Non-Patent Literature:</b> Search academic papers, standards documents, '
            'and product literature for additional prior art.',
            '<b>Attorney Review:</b> Request patent counsel review of the highest-scoring '
            'matches for detailed claim construction analysis.',
        ]

        all_recs = specific_recs + std_recs
        for i, rec in enumerate(all_recs, 1):
            elements.append(_para(f'{i}. {rec}', body_style))
            elements.append(Spacer(1, 4))
        elements.append(PageBreak())

        # section 9 — references

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
                    f'[{idx}] <b>{pub_num}</b> - {title}. '
                    f'{asgn}. '
                    f'<font size="8" color="#3498DB"><link href="{url}">{_safe(url, 50)}</link></font>'
                )
                elements.append(_para(ref_text, small_style))
                elements.append(Spacer(1, 2))
        else:
            elements.append(_para('No patent references to list.', body_style))

        # build pdf with page numbers

        doc.build(elements, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
        buffer.seek(0)
        return buffer.getvalue()
