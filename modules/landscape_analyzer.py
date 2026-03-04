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

    # Private helpers

    def _preprocess(self) -> None:
        """Clean and enrich the raw DataFrame for charting."""

        # Convert filing_date from integer (YYYYMMDD) or string to year
        # Handle 0, null, and string dates gracefully
        def _parse_filing_year(x):
            try:
                val = int(str(x)[:4])
                return val if val > 0 else None
            except (ValueError, TypeError):
                return None
        self.df["filing_year"] = self.df["filing_date"].apply(_parse_filing_year)
        # Drop rows with no valid year
        self.df = self.df.dropna(subset=["filing_year"])
        self.df["filing_year"] = self.df["filing_year"].astype(int)

        # Extract primary assignee name
        # assignee_name column may be a list, string, or JSON-encoded string
        def _extract_assignee(x):
            if isinstance(x, list) and len(x) > 0:
                name = str(x[0]).strip().rstrip(".")
                return name if name else "Individual/Unknown"
            elif isinstance(x, str) and x.strip():
                cleaned = x.strip()
                if cleaned.startswith("["):
                    import ast
                    try:
                        parsed = ast.literal_eval(cleaned)
                        if isinstance(parsed, list) and len(parsed) > 0:
                            return str(parsed[0]).strip().rstrip(".")
                    except (ValueError, SyntaxError):
                        pass
                return cleaned.rstrip(".")
            return "Individual/Unknown"

        self.df["primary_assignee"] = self.df["assignee_name"].apply(_extract_assignee)

        # Extract CPC section (first character of first CPC code)
        self.df["cpc_section"] = self.df["cpc_code"].apply(
            lambda x: x[0][:4] if isinstance(x, list) and len(x) > 0 else "Unknown"
        )

    # Public chart methods

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

    def export_charts_as_images(self, figures: dict = None) -> dict:
        """
        Generate all landscape charts as PNG bytes using matplotlib.
        Always works — no kaleido dependency required.

        The *figures* parameter (dict of Plotly figures) is accepted for
        backward compatibility but is NOT used; all charts are rendered
        from the underlying DataFrame via matplotlib.

        Returns
        -------
        dict
            ``{name: bytes}`` for each successfully exported chart.
        """
        import io
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed — cannot render charts for PDF")
            return {}

        images: dict = {}

        # Chart 1: Filing Trends
        try:
            yearly = (
                self.df[self.df["filing_year"] >= 1990]
                .groupby("filing_year")
                .size()
            )
            if not yearly.empty:
                fig_mpl, ax = plt.subplots(figsize=(10, 5))
                ax.plot(
                    yearly.index, yearly.values,
                    color="#2C3E50", linewidth=2.5,
                    marker="o", markersize=4,
                )
                ax.set_title("Patent Filing Trends", fontsize=14, fontweight="bold")
                ax.set_xlabel("Year", fontsize=12)
                ax.set_ylabel("Patents Filed", fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                plt.tight_layout()
                buf = io.BytesIO()
                fig_mpl.savefig(
                    buf, format="png", dpi=150,
                    bbox_inches="tight", facecolor="white",
                )
                plt.close(fig_mpl)
                buf.seek(0)
                images["filing_trends"] = buf.getvalue()
                print(f"  Filing trends chart: {len(images['filing_trends'])} bytes")
        except Exception as exc:
            print(f"  Filing trends chart FAILED: {exc}")

        # Chart 2: Top Assignees
        try:
            top = (
                self.df.groupby("primary_assignee")
                .size()
                .nlargest(12)
                .sort_values()
            )
            if not top.empty:
                fig_mpl, ax = plt.subplots(figsize=(10, 6))
                ax.barh(range(len(top)), top.values, color="#2C3E50")
                ax.set_yticks(range(len(top)))
                ax.set_yticklabels(top.index, fontsize=10)
                ax.set_title("Top Patent Holders", fontsize=14, fontweight="bold")
                ax.set_xlabel("Patent Count", fontsize=12)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                plt.tight_layout()
                buf = io.BytesIO()
                fig_mpl.savefig(
                    buf, format="png", dpi=150,
                    bbox_inches="tight", facecolor="white",
                )
                plt.close(fig_mpl)
                buf.seek(0)
                images["top_assignees"] = buf.getvalue()
                print(f"  Top assignees chart: {len(images['top_assignees'])} bytes")
        except Exception as exc:
            print(f"  Top assignees chart FAILED: {exc}")

        # Chart 3: CPC Distribution
        try:
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
            )
            if not cpc_counts.empty:
                section_colors = {
                    "A": "#E74C3C", "B": "#E67E22",
                    "C": "#F1C40F", "D": "#2ECC71",
                    "E": "#1ABC9C", "F": "#3498DB",
                    "G": "#9B59B6", "H": "#2C3E50",
                }
                colors = [
                    section_colors.get(code[0], "#95A5A6")
                    for code in cpc_counts.index
                ]
                fig_mpl, ax = plt.subplots(figsize=(10, 6))
                ax.barh(
                    range(len(cpc_counts)), cpc_counts.values,
                    color=colors,
                )
                ax.set_yticks(range(len(cpc_counts)))
                ax.set_yticklabels(cpc_counts.index, fontsize=10)
                ax.set_title(
                    "Technology Classification Distribution",
                    fontsize=14, fontweight="bold",
                )
                ax.set_xlabel("Patent Count", fontsize=12)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                plt.tight_layout()
                buf = io.BytesIO()
                fig_mpl.savefig(
                    buf, format="png", dpi=150,
                    bbox_inches="tight", facecolor="white",
                )
                plt.close(fig_mpl)
                buf.seek(0)
                images["cpc_distribution"] = buf.getvalue()
                print(f"  CPC distribution chart: {len(images['cpc_distribution'])} bytes")
        except Exception as exc:
            print(f"  CPC distribution chart FAILED: {exc}")

        return images
