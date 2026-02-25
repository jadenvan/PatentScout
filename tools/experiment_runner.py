"""
PatentScout — Brute-Force Experiment Runner

Orchestrates combinatorial trials of retrieval, matching, and postprocessing
strategies, scores them by a weighted objective, selects the best config,
and writes reports.

Usage (standalone)::

    venv/bin/python -m tools.experiment_runner

Or imported::

    from tools.experiment_runner import run_all_trials, choose_best_trial
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_TRIALS = 48

# Objective weights (higher is better, except cost and runtime)
W_TITLE_FRAC   = 0.30
W_HIGH_MATCHES  = 0.25
W_MOD_MATCHES   = 0.15
W_GB_SCANNED    = 0.20    # penalty
W_RUNTIME       = 0.10    # penalty

# Threshold presets
THRESHOLD_SETS = {
    "aggressive": {"high": 0.60, "moderate": 0.40, "low": 0.20},
    "medium":     {"high": 0.55, "moderate": 0.35, "low": 0.20},
    "permissive": {"high": 0.50, "moderate": 0.30, "low": 0.15},
}

# Default search space
DEFAULT_CONFIG_SPACE: dict[str, list] = {
    "query_scope":     ["title_only", "title_abstract"],
    "pattern_type":    ["regex", "substring"],
    "cpc_required":    [True, False],
    "cpc_prefix_len":  [3, 4],
    "date_filter":     [">20000101", ">20100101", None],
    "K_for_claim_fetch": [50, 100, 200],
    "max_bytes_gb":    [5, 10],
    "reformulation":   [True, False],
    "threshold_set":   ["aggressive", "medium", "permissive"],
    "dedupe_policy":   ["patent_element", "patent_feature", "none"],
    "snippet_window":  [30, 50, 80],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TrialResult:
    trial_id: str = ""
    config: dict = field(default_factory=dict)
    status: str = "pending"  # pending | success | failed
    error: str = ""
    # Metrics
    top20_title_fraction: float = 0.0
    high_matches_count: int = 0
    moderate_matches_count: int = 0
    unique_patents_count: int = 0
    gb_scanned_total: float = 0.0
    runtime_seconds: float = 0.0
    localized_snippet_fraction: float = 0.0
    # The actual match data (for the winning trial to regenerate PDF)
    matches: list = field(default_factory=list)
    detail_patents: list = field(default_factory=list)
    similarity_results: dict = field(default_factory=dict)
    score: float = 0.0
    logs: list = field(default_factory=list)

    def to_summary(self) -> dict:
        """Return a JSON-safe summary (without heavy data)."""
        return {
            "trial_id": self.trial_id,
            "config": self.config,
            "status": self.status,
            "error": self.error[:500] if self.error else "",
            "top20_title_fraction": self.top20_title_fraction,
            "high_matches_count": self.high_matches_count,
            "moderate_matches_count": self.moderate_matches_count,
            "unique_patents_count": self.unique_patents_count,
            "gb_scanned_total": round(self.gb_scanned_total, 4),
            "runtime_seconds": round(self.runtime_seconds, 2),
            "localized_snippet_fraction": round(self.localized_snippet_fraction, 4),
            "score": round(self.score, 6),
        }


# ---------------------------------------------------------------------------
# Configuration generation with pruning
# ---------------------------------------------------------------------------
def generate_config_space(
    overrides: Optional[dict] = None,
    max_trials: int = MAX_TRIALS,
) -> list[dict]:
    """
    Generate a list of trial configs from the cartesian product of the
    search space, pruned and prioritised to fit within *max_trials*.
    """
    space = dict(DEFAULT_CONFIG_SPACE)
    if overrides:
        space.update(overrides)

    # Build full cartesian product
    keys = list(space.keys())
    values = [space[k] for k in keys]
    all_combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    # Prune nonsensical combinations
    pruned = []
    for cfg in all_combos:
        # Skip claims-heavy scope with large K (too expensive)
        if cfg.get("query_scope") == "title_abstract_claims" and cfg.get("K_for_claim_fetch", 50) > 50:
            continue
        # Skip no-CPC with prefix_len variations (irrelevant)
        if not cfg.get("cpc_required") and cfg.get("cpc_prefix_len", 3) != 3:
            continue
        pruned.append(cfg)

    if len(pruned) <= max_trials:
        return pruned

    # Prioritise: reformulation=True, title_only, smaller K first
    def _priority(cfg):
        score = 0
        if cfg.get("reformulation"):
            score += 100
        if cfg.get("query_scope") == "title_only":
            score += 50
        if cfg.get("K_for_claim_fetch", 200) <= 50:
            score += 20
        if cfg.get("threshold_set") == "medium":
            score += 10
        return -score  # lower = higher priority for sorting

    pruned.sort(key=_priority)
    return pruned[:max_trials]


def _config_hash(config: dict) -> str:
    """Short hash of a config dict for caching / naming."""
    raw = json.dumps(config, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Core trial execution (uses cached session data)
# ---------------------------------------------------------------------------
def run_trial(
    config: dict,
    session_data: dict,
    search_strategy: dict,
    trial_id: str = "",
) -> TrialResult:
    """
    Execute a single experiment trial using cached session data.

    This re-runs the matching & postprocessing pipeline with different
    parameters, using the already-retrieved patents from the cached session.
    Actual BigQuery calls are NOT made per-trial — we reuse the cached
    detail_patents and parsed_claims.

    Args:
        config: Trial configuration dict.
        session_data: Loaded solar_charger_session.json dict.
        search_strategy: The search strategy from the session.
        trial_id: Unique identifier for this trial.

    Returns:
        A populated TrialResult.
    """
    if not trial_id:
        trial_id = f"trial_{_config_hash(config)}"

    result = TrialResult(trial_id=trial_id, config=config)
    t0 = time.time()

    try:
        # --- Feature reformulation ---
        features = copy.deepcopy(search_strategy.get("features", []))
        use_reformulation = config.get("reformulation", True)

        if use_reformulation:
            # Ensure patent_language is populated (from cached session)
            for f in features:
                if not f.get("patent_language"):
                    f["patent_language"] = f.get("description", "")
        else:
            # Wipe patent_language to force original-only matching
            for f in features:
                f.pop("patent_language", None)

        # --- Get thresholds ---
        ts_name = config.get("threshold_set", "medium")
        thresholds = THRESHOLD_SETS.get(ts_name, THRESHOLD_SETS["medium"])

        # --- Get parsed claims ---
        parsed_claims = session_data.get("parsed_claims", [])
        if not parsed_claims:
            result.status = "failed"
            result.error = "No parsed_claims in session data"
            return result

        # --- Run embedding similarity (recompute with feature variants) ---
        matches = _compute_matches(features, parsed_claims, thresholds, use_reformulation)

        # --- Apply dedupe policy ---
        dedupe = config.get("dedupe_policy", "patent_element")
        deduped = _apply_dedupe(matches, dedupe)

        # --- Snippet window ---
        snippet_window = config.get("snippet_window", 50)
        _apply_snippets(deduped, snippet_window, search_strategy)

        # --- Compute metrics ---
        detail_patents = session_data.get("detail_patents", [])
        search_terms = search_strategy.get("search_terms", [])
        primary_terms = [t.get("primary", "").lower() for t in search_terms if t.get("primary")]
        all_terms = list(primary_terms)
        for t in search_terms:
            all_terms.extend(syn.lower() for syn in t.get("synonyms", []))

        # top20 title fraction
        top20 = detail_patents[:20] if detail_patents else []
        title_hits = 0
        # Use both full terms and individual words for matching
        match_words = set(all_terms)
        for term in all_terms:
            for word in term.split():
                if len(word) >= 4:
                    match_words.add(word.lower())
        # Also include feature keywords
        for f in features:
            for kw in f.get("keywords", []):
                if kw and len(kw) >= 4:
                    match_words.add(kw.lower())
        for pat in top20:
            title_lower = str(pat.get("title", "")).lower()
            if any(term in title_lower for term in match_words if term):
                title_hits += 1
        result.top20_title_fraction = title_hits / max(len(top20), 1)

        # Match counts
        result.high_matches_count = sum(1 for m in deduped if m.get("similarity_level") == "HIGH")
        result.moderate_matches_count = sum(1 for m in deduped if m.get("similarity_level") == "MODERATE")

        # Unique patents
        patent_nums = set(m.get("patent_number", "") for m in deduped if m.get("patent_number"))
        result.unique_patents_count = len(patent_nums)

        # GB scanned (from session, adjusted by config)
        base_gb = session_data.get("total_gb_scanned", 0.0) or 0.0
        max_bytes_gb = config.get("max_bytes_gb", 10)
        result.gb_scanned_total = min(base_gb, max_bytes_gb)

        # Snippet localization
        localized = sum(1 for m in deduped if m.get("snippet_localized", False))
        result.localized_snippet_fraction = localized / max(len(deduped), 1)

        result.matches = deduped
        result.detail_patents = detail_patents
        result.similarity_results = {
            "matches": deduped,
            "stats": {
                "high_matches": result.high_matches_count,
                "moderate_matches": result.moderate_matches_count,
                "low_matches": sum(1 for m in deduped if m.get("similarity_level") == "LOW"),
                "total_comparisons": len(features) * sum(
                    len(p.get("independent_claims", []))
                    for p in parsed_claims
                ),
            },
        }
        result.status = "success"

    except Exception as exc:
        result.status = "failed"
        result.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        logger.error("Trial %s failed: %s", trial_id, exc)

    result.runtime_seconds = time.time() - t0
    return result


def _compute_matches(
    features: list[dict],
    parsed_claims: list[dict],
    thresholds: dict,
    use_reformulation: bool,
) -> list[dict]:
    """
    Compute similarity matches using sentence-transformers.
    Replicates EmbeddingEngine logic without Streamlit dependency.
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from config import settings

    model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)

    # Feature texts
    feature_texts_orig = [f.get("description", "") for f in features]
    feature_texts_patent = [
        f.get("patent_language") or f.get("description", "")
        for f in features
    ]
    feature_labels = [f.get("label", f"feature_{i}") for i, f in enumerate(features)]

    has_reform = use_reformulation and any(
        f.get("patent_language") and f["patent_language"] != f.get("description", "")
        for f in features
    )

    # Collect claim elements
    element_texts = []
    element_refs = []
    for patent in parsed_claims:
        for claim in patent.get("independent_claims", []):
            for element in claim.get("elements", []):
                element_texts.append(element["text"])
                element_refs.append({
                    "patent_number": patent["patent_number"],
                    "claim_number": claim["claim_number"],
                    "element_id": element["id"],
                    "full_claim_text": claim.get("full_text", ""),
                })

    if not element_texts:
        return []

    # Encode
    element_emb = model.encode(element_texts, show_progress_bar=False)
    orig_emb = model.encode(feature_texts_orig, show_progress_bar=False)

    sim_matrix = cosine_similarity(orig_emb, element_emb)

    if has_reform:
        patent_emb = model.encode(feature_texts_patent, show_progress_bar=False)
        sim_patent = cosine_similarity(patent_emb, element_emb)
        sim_matrix = np.maximum(sim_matrix, sim_patent)

    # Build matches
    th_high = thresholds.get("high", 0.55)
    th_mod = thresholds.get("moderate", 0.35)
    th_low = thresholds.get("low", 0.20)

    matches = []
    for i, feature in enumerate(features):
        feature_matches = []
        for j, ref in enumerate(element_refs):
            score = float(sim_matrix[i][j])
            if score >= th_low:
                if score >= th_high:
                    level = "HIGH"
                elif score >= th_mod:
                    level = "MODERATE"
                else:
                    level = "LOW"
                feature_matches.append({
                    "feature_label": feature_labels[i],
                    "feature_description": feature.get("description", ""),
                    "element_text": element_texts[j],
                    "patent_number": ref["patent_number"],
                    "claim_number": ref["claim_number"],
                    "element_id": ref["element_id"],
                    "similarity_score": round(score, 3),
                    "similarity_level": level,
                })
        feature_matches.sort(key=lambda x: x["similarity_score"], reverse=True)
        matches.extend(feature_matches[:10])

    return matches


def _apply_dedupe(matches: list[dict], policy: str) -> list[dict]:
    """Deduplicate matches according to policy."""
    if policy == "none":
        return matches

    seen = set()
    deduped = []
    for m in sorted(matches, key=lambda x: x.get("similarity_score", 0), reverse=True):
        if policy == "patent_element":
            key = (m.get("patent_number", ""), m.get("element_id", ""))
        elif policy == "patent_feature":
            key = (m.get("patent_number", ""), m.get("feature_label", ""))
        else:
            key = id(m)

        if key not in seen:
            seen.add(key)
            deduped.append(m)
    return deduped


def _apply_snippets(
    matches: list[dict],
    window: int,
    search_strategy: dict,
) -> None:
    """Add snippet and snippet_localized fields to matches in-place."""
    primary_terms = []
    for t in search_strategy.get("search_terms", []):
        p = t.get("primary", "")
        if p:
            primary_terms.append(p.lower())
        for syn in t.get("synonyms", []):
            if syn:
                primary_terms.append(syn.lower())

    for m in matches:
        text = m.get("element_text", "")
        text_clean = " ".join(text.replace("\n", " ").split())

        # Try to find a matched token and centre snippet around it
        localized = False
        snippet = ""
        text_lower = text_clean.lower()
        for term in primary_terms:
            idx = text_lower.find(term)
            if idx >= 0:
                start = max(0, idx - window // 2)
                end = min(len(text_clean), idx + len(term) + window // 2)
                snippet = text_clean[start:end]
                localized = True
                break

        if not snippet:
            snippet = text_clean[:window * 2]

        if len(snippet) < len(text_clean):
            snippet += "..."

        m["snippet"] = snippet
        m["snippet_localized"] = localized


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_trial(result: TrialResult) -> float:
    """Compute a single scalar score for a trial result (higher = better)."""
    if result.status != "success":
        return -999.0
    return result.score  # set by normalize_and_score_trials


def normalize_and_score_trials(results: list[TrialResult]) -> None:
    """Normalize metrics across all trials and compute final scores in-place."""
    successful = [r for r in results if r.status == "success"]
    if not successful:
        return

    def _minmax(values):
        lo, hi = min(values), max(values)
        rng = hi - lo if hi > lo else 1.0
        return [(v - lo) / rng for v in values]

    title_fracs = _minmax([r.top20_title_fraction for r in successful])
    high_matches = _minmax([r.high_matches_count for r in successful])
    mod_matches = _minmax([r.moderate_matches_count for r in successful])
    gb_scans = _minmax([r.gb_scanned_total for r in successful])
    runtimes = _minmax([r.runtime_seconds for r in successful])

    for i, r in enumerate(successful):
        r.score = (
            W_TITLE_FRAC  * title_fracs[i]
            + W_HIGH_MATCHES * high_matches[i]
            + W_MOD_MATCHES * mod_matches[i]
            - W_GB_SCANNED  * gb_scans[i]
            - W_RUNTIME     * runtimes[i]
        )


def choose_best_trial(results: list[TrialResult]) -> Optional[TrialResult]:
    """Return the trial with the highest score, or None."""
    successful = [r for r in results if r.status == "success"]
    if not successful:
        return None
    return max(successful, key=lambda r: r.score)


# ---------------------------------------------------------------------------
# Run all trials
# ---------------------------------------------------------------------------
def run_all_trials(
    config_space: list[dict],
    session_data: dict,
    search_strategy: dict,
    max_workers: int = 2,
) -> list[TrialResult]:
    """
    Run all trial configs.  Uses ThreadPoolExecutor with limited parallelism.
    """
    results: list[TrialResult] = []

    # For efficiency, pre-load the sentence-transformers model once
    # (each trial will load from cache)
    print(f"[ExperimentRunner] Running {len(config_space)} trials (max_workers={max_workers})...")

    def _run_one(i_cfg):
        i, cfg = i_cfg
        tid = f"trial_{i:03d}_{_config_hash(cfg)}"
        return run_trial(cfg, session_data, search_strategy, trial_id=tid)

    # Run sequentially to avoid model-loading race conditions
    for i, cfg in enumerate(config_space):
        tid = f"trial_{i:03d}_{_config_hash(cfg)}"
        r = run_trial(cfg, session_data, search_strategy, trial_id=tid)
        results.append(r)
        status_char = "+" if r.status == "success" else "X"
        print(f"  [{status_char}] {tid}: score_raw H={r.high_matches_count} M={r.moderate_matches_count} "
              f"T20={r.top20_title_fraction:.2f} GB={r.gb_scanned_total:.1f} "
              f"t={r.runtime_seconds:.1f}s")

    normalize_and_score_trials(results)
    return results


# ---------------------------------------------------------------------------
# Apply best trial as default config
# ---------------------------------------------------------------------------
def apply_best_trial_as_default(trial: TrialResult) -> str:
    """
    Write chosen trial config to config/active_config.json.
    Returns the path written.
    """
    config_path = os.path.join("config", "active_config.json")
    payload = {
        "chosen_trial_id": trial.trial_id,
        "chosen_at": datetime.now(timezone.utc).isoformat(),
        "config": trial.config,
        "thresholds": THRESHOLD_SETS.get(
            trial.config.get("threshold_set", "medium"),
            THRESHOLD_SETS["medium"],
        ),
        "metrics": {
            "top20_title_fraction": trial.top20_title_fraction,
            "high_matches_count": trial.high_matches_count,
            "moderate_matches_count": trial.moderate_matches_count,
            "unique_patents_count": trial.unique_patents_count,
            "gb_scanned_total": round(trial.gb_scanned_total, 4),
            "runtime_seconds": round(trial.runtime_seconds, 2),
            "score": round(trial.score, 6),
        },
    }
    os.makedirs("config", exist_ok=True)
    with open(config_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[ExperimentRunner] Best config written to {config_path}")
    return config_path


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def write_report(
    results: list[TrialResult],
    chosen: TrialResult,
    timestamp: Optional[str] = None,
) -> tuple[str, str]:
    """
    Write bruteforce_run_<timestamp>.json and .md reports.
    Returns (json_path, md_path).
    """
    if not timestamp:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    os.makedirs("reports", exist_ok=True)
    base = f"bruteforce_run_{timestamp}"
    json_path = os.path.join("reports", f"{base}.json")
    md_path = os.path.join("reports", f"{base}.md")

    # JSON report
    report = {
        "timestamp": timestamp,
        "total_trials": len(results),
        "successful_trials": sum(1 for r in results if r.status == "success"),
        "failed_trials": sum(1 for r in results if r.status == "failed"),
        "chosen_trial_id": chosen.trial_id if chosen else None,
        "chosen_config": chosen.config if chosen else None,
        "chosen_score": round(chosen.score, 6) if chosen else None,
        "chosen_metrics": chosen.to_summary() if chosen else None,
        "trials": [r.to_summary() for r in results],
    }
    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=2)

    # Markdown report
    md_lines = [
        f"# PatentScout Brute-Force Experiment Report",
        f"",
        f"**Generated:** {timestamp}",
        f"",
        f"## Summary",
        f"",
        f"- **Total trials:** {len(results)}",
        f"- **Successful:** {sum(1 for r in results if r.status == 'success')}",
        f"- **Failed:** {sum(1 for r in results if r.status == 'failed')}",
        f"- **Chosen trial:** `{chosen.trial_id}`" if chosen else "- **Chosen trial:** None",
        f"- **Best score:** {chosen.score:.6f}" if chosen else "- **Best score:** N/A",
        f"",
    ]

    if chosen:
        md_lines.extend([
            f"## Chosen Configuration",
            f"",
            f"```json",
            json.dumps(chosen.config, indent=2),
            f"```",
            f"",
            f"### Chosen Metrics",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Top-20 Title Fraction | {chosen.top20_title_fraction:.3f} |",
            f"| High Matches | {chosen.high_matches_count} |",
            f"| Moderate Matches | {chosen.moderate_matches_count} |",
            f"| Unique Patents | {chosen.unique_patents_count} |",
            f"| GB Scanned | {chosen.gb_scanned_total:.2f} |",
            f"| Runtime (s) | {chosen.runtime_seconds:.2f} |",
            f"| Score | {chosen.score:.6f} |",
            f"",
        ])

    # Top 10 trials table
    sorted_results = sorted(
        [r for r in results if r.status == "success"],
        key=lambda r: r.score, reverse=True,
    )[:10]
    if sorted_results:
        md_lines.extend([
            f"## Top 10 Trials",
            f"",
            f"| Rank | Trial | Score | High | Mod | T20 Frac | GB | Runtime |",
            f"|------|-------|-------|------|-----|----------|----|---------|",
        ])
        for rank, r in enumerate(sorted_results, 1):
            md_lines.append(
                f"| {rank} | `{r.trial_id}` | {r.score:.4f} | "
                f"{r.high_matches_count} | {r.moderate_matches_count} | "
                f"{r.top20_title_fraction:.2f} | {r.gb_scanned_total:.1f} | "
                f"{r.runtime_seconds:.1f}s |"
            )
        md_lines.append("")

    # Failed trials
    failed = [r for r in results if r.status == "failed"]
    if failed:
        md_lines.extend([
            f"## Failed Trials ({len(failed)})",
            f"",
        ])
        for r in failed[:5]:
            md_lines.append(f"- `{r.trial_id}`: {r.error[:200]}")
        md_lines.append("")

    with open(md_path, "w") as fh:
        fh.write("\n".join(md_lines))

    print(f"[ExperimentRunner] Reports written: {json_path}, {md_path}")
    return json_path, md_path


# ---------------------------------------------------------------------------
# PDF generation with chosen config
# ---------------------------------------------------------------------------
def generate_final_pdf(
    session_data: dict,
    chosen: TrialResult,
    output_path: str = "tests/patentscout_report.pdf",
) -> str:
    """
    Generate the final PDF using the chosen trial's match data.
    Backs up existing PDF if present.
    Returns the output path.
    """
    # Backup existing
    if os.path.exists(output_path):
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = output_path.replace(".pdf", f"_{ts}.pdf")
        os.rename(output_path, backup)
        print(f"[ExperimentRunner] Backed up existing PDF to {backup}")

    # Build session data with chosen trial's results
    pdf_session = dict(session_data)
    if chosen and chosen.similarity_results:
        pdf_session["similarity_results"] = chosen.similarity_results
    if chosen and chosen.matches:
        # Rebuild comparison_matrix from matches (HIGH + MODERATE only)
        cmat = [
            m for m in chosen.matches
            if m.get("similarity_level") in ("HIGH", "MODERATE")
        ]
        pdf_session["comparison_matrix"] = cmat

    from modules.report_generator import ReportGenerator
    rg = ReportGenerator()
    pdf_bytes = rg.generate(pdf_session)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as fh:
        fh.write(pdf_bytes)

    size = os.path.getsize(output_path)
    print(f"[ExperimentRunner] PDF written: {output_path} ({size:,} bytes)")
    return output_path


# ---------------------------------------------------------------------------
# STOP report
# ---------------------------------------------------------------------------
def write_stop_report(reason: str, logs: list[str]) -> str:
    """Write a STOP report and return its path."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join("reports", f"STOP_bruteforce_{ts}.md")
    os.makedirs("reports", exist_ok=True)
    content = [
        "# PatentScout Brute-Force STOP Report",
        "",
        f"**Timestamp:** {ts}",
        f"**Reason:** {reason}",
        "",
        "## Logs",
        "",
    ]
    content.extend(f"- {line}" for line in logs)
    with open(path, "w") as fh:
        fh.write("\n".join(content))
    return path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    """
    Run the full brute-force experiment.
    """
    print("=" * 72)
    print("  PatentScout Brute-Force Experiment Runner")
    print("=" * 72)

    # Load cached session
    session_path = os.path.join("examples", "solar_charger_session.json")
    if not os.path.exists(session_path):
        path = write_stop_report(
            "Cached session not found",
            [f"Expected: {session_path}"],
        )
        print(f"STOP: {path}")
        return

    print(f"Loading cached session from {session_path}...")
    with open(session_path) as fh:
        session_data = json.load(fh)

    search_strategy = session_data.get("search_strategy", {})
    if not search_strategy:
        path = write_stop_report(
            "No search_strategy in session data",
            ["Session file may be corrupt"],
        )
        print(f"STOP: {path}")
        return

    # Generate config space
    configs = generate_config_space(max_trials=MAX_TRIALS)
    print(f"Config space: {len(configs)} trials")

    # Run all trials
    results = run_all_trials(configs, session_data, search_strategy)

    # Choose best
    best = choose_best_trial(results)
    if not best:
        path = write_stop_report(
            "All trials failed",
            [r.error[:200] for r in results if r.status == "failed"][:10],
        )
        print(f"STOP: {path}")
        return

    print(f"\n{'=' * 72}")
    print(f"  BEST TRIAL: {best.trial_id}")
    print(f"  Score: {best.score:.6f}")
    print(f"  Config: {json.dumps(best.config, indent=2)}")
    print(f"{'=' * 72}\n")

    # Apply config
    config_path = apply_best_trial_as_default(best)

    # Write reports
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path, md_path = write_report(results, best, timestamp=ts)

    # Generate final PDF
    try:
        pdf_path = generate_final_pdf(session_data, best)
    except Exception as exc:
        print(f"[ExperimentRunner] PDF generation failed: {exc}")
        # Retry up to 3 times
        for attempt in range(2):
            try:
                time.sleep(1)
                pdf_path = generate_final_pdf(session_data, best)
                break
            except Exception:
                pass
        else:
            path = write_stop_report(
                "PDF generation failed 3 times",
                [str(exc), traceback.format_exc()],
            )
            print(f"STOP: {path}")
            return

    print(f"\n{'=' * 72}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"  Config:  {config_path}")
    print(f"  Report:  {json_path}")
    print(f"  Summary: {md_path}")
    print(f"  PDF:     {pdf_path}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    # Allow running as: python -m tools.experiment_runner
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
