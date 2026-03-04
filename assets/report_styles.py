"""
PatentScout — Report Styles

ReportLab colour palette, paragraph styles and table-cell styles used by
ReportGenerator when building the downloadable PDF analysis reports.
"""

from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY


# Colour palette (spec-required names)
PRIMARY    = HexColor('#2C3E50')
SECONDARY  = HexColor('#34495E')
ACCENT     = HexColor('#3498DB')
WARNING    = HexColor('#E74C3C')
SUCCESS    = HexColor('#27AE60')
LIGHT_GRAY = HexColor('#ECF0F1')
WHITE      = HexColor('#FFFFFF')

# Internal helpers
_ROW_ALT = HexColor('#F8FBFD')


# Paragraph styles
_base = getSampleStyleSheet()

title_style = ParagraphStyle(
    'PSTitle',
    parent=_base['Title'],
    textColor=PRIMARY,
    fontSize=26,
    leading=32,
    spaceAfter=6,
    alignment=TA_CENTER,
    fontName='Helvetica-Bold',
)

subtitle_style = ParagraphStyle(
    'PSSubtitle',
    parent=_base['Normal'],
    textColor=SECONDARY,
    fontSize=14,
    leading=20,
    spaceAfter=4,
    alignment=TA_CENTER,
    fontName='Helvetica',
)

date_style = ParagraphStyle(
    'PSDate',
    parent=_base['Normal'],
    textColor=SECONDARY,
    fontSize=10,
    leading=14,
    spaceAfter=0,
    alignment=TA_CENTER,
    fontName='Helvetica-Oblique',
)

heading_style = ParagraphStyle(
    'PSHeading',
    parent=_base['Heading1'],
    textColor=PRIMARY,
    fontSize=14,
    leading=18,
    spaceBefore=12,
    spaceAfter=8,
    fontName='Helvetica-Bold',
)

heading2_style = ParagraphStyle(
    'PSHeading2',
    parent=_base['Heading2'],
    textColor=SECONDARY,
    fontSize=12,
    leading=16,
    spaceBefore=8,
    spaceAfter=4,
    fontName='Helvetica-Bold',
)

body_style = ParagraphStyle(
    'PSBody',
    parent=_base['Normal'],
    textColor=PRIMARY,
    fontSize=10,
    leading=14,
    spaceAfter=6,
    alignment=TA_JUSTIFY,
    fontName='Helvetica',
)

bullet_style = ParagraphStyle(
    'PSBullet',
    parent=_base['Normal'],
    textColor=PRIMARY,
    fontSize=10,
    leading=14,
    spaceAfter=4,
    leftIndent=16,
    fontName='Helvetica',
)

small_style = ParagraphStyle(
    'PSSmall',
    parent=_base['Normal'],
    textColor=SECONDARY,
    fontSize=8,
    leading=11,
    spaceAfter=2,
    fontName='Helvetica',
)

disclaimer_style = ParagraphStyle(
    'PSDisclaimer',
    parent=_base['Normal'],
    textColor=PRIMARY,
    fontSize=10,
    leading=15,
    spaceAfter=0,
    alignment=TA_JUSTIFY,
    fontName='Helvetica',
)


# Table style helpers
TABLE_HEADER_STYLE = [
    ('BACKGROUND',    (0, 0), (-1, 0), PRIMARY),
    ('TEXTCOLOR',     (0, 0), (-1, 0), WHITE),
    ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE',      (0, 0), (-1, 0), 9),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 7),
    ('TOPPADDING',    (0, 0), (-1, 0), 7),
    ('ALIGN',         (0, 0), (-1, 0), 'CENTER'),
]

TABLE_BODY_STYLE = [
    ('FONTNAME',      (0, 1), (-1, -1), 'Helvetica'),
    ('FONTSIZE',      (0, 1), (-1, -1), 8),
    ('TOPPADDING',    (0, 1), (-1, -1), 5),
    ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
    ('LEFTPADDING',   (0, 0), (-1, -1), 6),
    ('RIGHTPADDING',  (0, 0), (-1, -1), 6),
    ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
    ('ROWBACKGROUNDS',(0, 1), (-1, -1), [WHITE, _ROW_ALT]),
    ('GRID',          (0, 0), (-1, -1), 0.4, LIGHT_GRAY),
    ('LINEBELOW',     (0, 0), (-1, 0),  1,   PRIMARY),
]

BASE_TABLE_STYLE = TABLE_HEADER_STYLE + TABLE_BODY_STYLE


# Additional paragraph styles for match-detail pages
mono_style = ParagraphStyle(
    'PSMono',
    parent=_base['Normal'],
    textColor=SECONDARY,
    fontSize=9,
    leading=12,
    spaceAfter=4,
    fontName='Courier',
)

small_bold = ParagraphStyle(
    'PSSmallBold',
    parent=_base['Normal'],
    textColor=PRIMARY,
    fontSize=9,
    leading=12,
    spaceAfter=2,
    fontName='Helvetica-Bold',
)

feature_header_style = ParagraphStyle(
    'PSFeatureHeader',
    parent=_base['Heading2'],
    textColor=ACCENT,
    fontSize=13,
    leading=17,
    spaceBefore=10,
    spaceAfter=4,
    fontName='Helvetica-Bold',
)

recommendation_style = ParagraphStyle(
    'PSRecommendation',
    parent=_base['Normal'],
    textColor=PRIMARY,
    fontSize=10,
    leading=14,
    spaceBefore=6,
    spaceAfter=4,
    fontName='Helvetica-BoldOblique',
    leftIndent=8,
)

# Row background colours keyed by similarity/confidence level
CONF_COLORS = {
    'HIGH':     HexColor('#FDECEA'),
    'MODERATE': HexColor('#FEF9E7'),
    'LOW':      HexColor('#EAF9EA'),
}

# Bar / badge colours for score visualisation
SCORE_BAR_FG = ACCENT
SCORE_BAR_BG = HexColor('#E0E0E0')
