"""
PatentScout — Landscape Analyzer Module

Analyses a corpus of retrieved patents to produce landscape-level insights:
filing trends over time, top assignee concentration, and CPC technology
classification distribution.  All methods return Plotly figures ready to be
passed directly to st.plotly_chart().
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class LandscapeAnalyzer:
    """
    Produces aggregate statistics and interactive Plotly visualisations from a
    DataFrame of retrieved landscape patents.

    Parameters
    ----------
    landscape_df:
        DataFrame produced by PatentRetriever containing at minimum the
        columns ``filing_date``, ``assignee_name``, and ``cpc_code``.
    """

    def __init__(self, landscape_df: pd.DataFrame) -> None:
        self.df = landscape_df.copy()
        self._preprocess()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _preprocess(self) -> None:
        """Clean and enrich the raw DataFrame for charting."""

        # Convert filing_date from integer (YYYYMMDD) to year
        # Handle 0 and null dates gracefully
        self.df["filing_year"] = self.df["filing_date"].apply(
            lambda x: int(str(x)[:4]) if x and x > 0 else None
        )
        # Drop rows with no valid year
        self.df = self.df.dropna(subset=["filing_year"])
        self.df["filing_year"] = self.df["filing_year"].astype(int)

        # Extract primary assignee name
        # assignee_name column is a list (from our aggregation in retriever)
        # Take first assignee, clean it
        self.df["primary_assignee"] = self.df["assignee_name"].apply(
            lambda x: (
                x[0].strip().rstrip(".")
                if isinstance(x, list) and len(x) > 0
                else "Individual/Unknown"
            )
        )

        # Extract CPC section (first character of first CPC code)
        self.df["cpc_section"] = self.df["cpc_code"].apply(
            lambda x: x[0][:4] if isinstance(x, list) and len(x) > 0 else "Unknown"
        )

    # ------------------------------------------------------------------
    # Public chart methods
    # ------------------------------------------------------------------

    def filing_trends(self) -> go.Figure:
        """
        Line chart: patent count per filing year (1990 → present).
        """
        yearly = (
            self.df[self.df["filing_year"] >= 1990]
            .groupby("filing_year")
            .size()
            .reset_index(name="count")
        )

        fig = px.line(
            yearly,
            x="filing_year",
            y="count",
            title="Patent Filing Trends",
            labels={"filing_year": "Year", "count": "Patents Filed"},
            template="plotly_white",
        )
        fig.update_traces(line=dict(color="#2C3E50", width=2.5))
        fig.update_layout(font=dict(size=12), title_font_size=16)
        return fig

    def top_assignees(self) -> go.Figure:
        """
        Horizontal bar chart: top 12 patent holders by count.
        """
        top = (
            self.df.groupby("primary_assignee")
            .size()
            .nlargest(12)
            .sort_values()
            .reset_index(name="count")
        )

        fig = px.bar(
            top,
            x="count",
            y="primary_assignee",
            orientation="h",
            title="Top Patent Holders",
            labels={"count": "Patent Count", "primary_assignee": ""},
            template="plotly_white",
        )
        fig.update_traces(marker_color="#2C3E50")
        fig.update_layout(
            font=dict(size=12),
            title_font_size=16,
            yaxis=dict(autorange="reversed"),
        )
        return fig

    def cpc_distribution(self) -> go.Figure:
        """
        Horizontal bar chart: top 15 CPC 4-character groups, colour-coded
        by technology section.
        """
        # Work only on rows that have CPC data
        cpc_df = self.df.dropna(subset=["cpc_code"])
        cpc_exploded = cpc_df.explode("cpc_code")
        cpc_exploded["cpc_group"] = cpc_exploded["cpc_code"].apply(
            lambda x: x[:4] if isinstance(x, str) else "Unknown"
        )

        cpc_counts = (
            cpc_exploded.groupby("cpc_group")
            .size()
            .nlargest(15)
            .sort_values()
            .reset_index(name="count")
        )

        section_names = {
            "A": "Human Necessities",
            "B": "Operations/Transport",
            "C": "Chemistry/Metallurgy",
            "D": "Textiles/Paper",
            "E": "Fixed Constructions",
            "F": "Mechanical Engineering",
            "G": "Physics",
            "H": "Electricity",
        }
        cpc_counts["section"] = (
            cpc_counts["cpc_group"].str[0].map(section_names).fillna("Other")
        )

        fig = px.bar(
            cpc_counts,
            x="count",
            y="cpc_group",
            orientation="h",
            color="section",
            title="Technology Classification Distribution",
            labels={"count": "Patent Count", "cpc_group": "CPC Code"},
            template="plotly_white",
        )
        fig.update_layout(
            font=dict(size=12),
            title_font_size=16,
            legend_title="Technology Section",
        )
        return fig

    def export_charts_as_images(self, figures: dict) -> dict:
        """
        Export each Plotly figure as PNG bytes for PDF embedding.

        Returns
        -------
        dict
            ``{name: bytes}`` for each successfully exported figure.
            If kaleido is not available the figure is skipped and a
            warning is printed.
        """
        images: dict = {}
        for name, fig in figures.items():
            try:
                img_bytes = fig.to_image(
                    format="png", width=800, height=400, scale=2
                )
                images[name] = img_bytes
            except Exception as exc:
                print(f"Chart export failed for {name}: {exc}")
        return images
