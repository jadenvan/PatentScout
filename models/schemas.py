"""
PatentScout Data Schemas

Dataclass definitions used throughout the application to represent
structured data passed between modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class InventionInput:
    """
    Captures the user's raw invention description and optional supporting
    materials uploaded via the Streamlit UI.
    """
    description: str = ""
    sketch_path: Optional[str] = None
    extracted_features: List[str] = field(default_factory=list)
    predicted_cpc_codes: List[str] = field(default_factory=list)


@dataclass
class SearchStrategy:
    """
    Holds the search parameters that will be translated into BigQuery SQL,
    including CPC classification codes and keyword terms.
    """
    cpc_codes: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    country_codes: List[str] = field(default_factory=lambda: ["US"])
    date_range_start: Optional[int] = None   # YYYYMMDD integer format
    date_range_end: Optional[int] = None     # YYYYMMDD integer format
    limit: int = 100


@dataclass
class PatentRecord:
    """
    Represents a single patent retrieved from BigQuery, normalised from
    the nested BigQuery schema into a flat Python object.
    """
    publication_number: str = ""
    title: str = ""
    abstract: str = ""
    claims_text: str = ""
    cpc_codes: List[str] = field(default_factory=list)
    assignees: List[str] = field(default_factory=list)
    inventors: List[str] = field(default_factory=list)
    filing_date: Optional[int] = None    # YYYYMMDD
    grant_date: Optional[int] = None     # YYYYMMDD
    publication_date: Optional[int] = None  # YYYYMMDD
    family_id: str = ""
    country_code: str = ""
    similarity_score: float = 0.0


@dataclass
class ClaimStructure:
    """
    Structured representation of a parsed patent claim, broken into its
    preamble, transitional phrase, and body elements.
    """
    claim_number: int = 1
    claim_type: str = "independent"    # "independent" or "dependent"
    preamble: str = ""
    transition: str = ""               # e.g. "comprising", "consisting of"
    elements: List[str] = field(default_factory=list)
    plain_english: str = ""
    raw_text: str = ""


@dataclass
class ComparisonResult:
    """
    Holds the result of comparing a user's invention elements against a
    single retrieved patent, including per-element similarity scores.
    """
    patent: PatentRecord = field(default_factory=PatentRecord)
    overall_similarity: float = 0.0
    element_scores: dict = field(default_factory=dict)   # element -> score
    risk_level: str = "LOW"                              # LOW | MODERATE | HIGH
    key_overlapping_claims: List[str] = field(default_factory=list)
    distinguishing_features: List[str] = field(default_factory=list)


@dataclass
class WhiteSpaceResult:
    """
    Captures identified IP whitespace — areas where the user's invention
    has low overlap with existing patents.
    """
    whitespace_areas: List[str] = field(default_factory=list)
    coverage_map: dict = field(default_factory=dict)     # feature -> coverage %
    opportunity_score: float = 0.0                       # 0–1, higher = more room
    gemini_analysis: str = ""
    recommended_cpc_gaps: List[str] = field(default_factory=list)
