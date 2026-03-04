#!/usr/bin/env python3
"""Render full text content of all 4 tabs for both PatentScout demos."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo_data import build_solar_demo_data, build_doorbell_demo_data
from collections import defaultdict, Counter
import pandas as pd

SEP = "=" * 72
THICK = "#" * 80

def fmt_demo(name, data):
    out = []
    def p(*args): out.append(" ".join(str(a) for a in args))

    p(THICK)
    p(f"# DEMO: {name}")
    p(THICK)

    # ── Invention description ──
    p("\n## INVENTION DESCRIPTION")
    p(data["invention_text"])

    # ── Extracted features ──
    strat = data["search_strategy"]
    p("\n## EXTRACTED FEATURES")
    for f in strat.get("features", []):
        src = f.get("source", "text")
        tag = "  [from sketch]" if src == "sketch" else ""
        p(f"  • {f['label']}{tag}")
        p(f"    Description: {f['description']}")
        kw = f.get("keywords", [])
        if kw:
            p(f"    Keywords: {', '.join(kw)}")
        pl = f.get("patent_language", "")
        if pl:
            p(f"    Patent language: {pl[:160]}")

    p("\n## PREDICTED CPC CODES")
    for c in strat.get("cpc_codes", []):
        p(f"  • {c['code']} — {c.get('rationale', '')}")

    # TAB 1 — Prior Art Results
    p(f"\n{SEP}")
    p("TAB 1: PRIOR ART RESULTS")
    p(SEP)

    dp_list = data["detail_patents"]
    df = pd.DataFrame(dp_list)

    unique_asgn = set()
    for a in df["assignee_name"]:
        if isinstance(a, list):
            unique_asgn.update(x for x in a if x)
        elif isinstance(a, str) and a:
            unique_asgn.add(a)

    dates_n = pd.to_numeric(df["publication_date"], errors="coerce").dropna()
    dates_n = dates_n[dates_n > 0]
    oldest = str(int(dates_n.min()))[:4] if not dates_n.empty else "N/A"
    newest = str(int(dates_n.max()))[:4] if not dates_n.empty else "N/A"

    p(f"\n[METRICS]")
    p(f"  Patents Found      : {len(df)}")
    p(f"  Unique Assignees   : {len(unique_asgn)}")
    p(f"  Publication Date Range : {oldest}–{newest}")

    p(f"\n[TOP 20 PATENTS TABLE]")
    p(f"  {'#':<3} {'Publication #':<22} {'Title':<50} {'Assignee':<28} {'Pub Date':<12} Relevance")
    p("  " + "-" * 130)
    for i, row in df.head(20).iterrows():
        asgn = row["assignee_name"]
        if isinstance(asgn, list):
            asgn = "; ".join(str(a) for a in asgn[:2])
        pub_date = row.get("publication_date_str", str(row.get("publication_date", "")))
        score = row.get("relevance_score", "")
        score_str = f"{score:.3f}" if isinstance(score, float) else str(score)
        p(f"  {i+1:<3} {row['publication_number']:<22} {str(row['title'])[:49]:<50} {str(asgn)[:27]:<28} {pub_date:<12} {score_str}")

    p(f"\n[ABSTRACT DETAIL — TOP 10 PATENTS]")
    for _, row in df.head(10).iterrows():
        p(f"\n  [{row['publication_number']}]  {row['title']}")
        p(f"  URL   : {row.get('patent_url', '#')}")
        p(f"  Abstract: {str(row.get('abstract', ''))[:280]}")

    p(f"\n  ✅ Retrieved {len(df)} patents from Google BigQuery Patents Database (100M+ publications)")

    # TAB 2 — Claim Analysis
    p(f"\n{SEP}")
    p("TAB 2: CLAIM ANALYSIS — Feature vs. Claim Comparison")
    p(SEP)

    pc_list = data["parsed_claims"]
    n = len(pc_list)
    p(f"\n  Parsed claims for {n} out of {n} patents")

    sim = data["similarity_results"]
    stats = sim["stats"]
    matches = sim["matches"]

    p(f"\n[SIMILARITY ANALYSIS — STATS]")
    p(f"  Total Comparisons  : {stats['total_comparisons']}")
    p(f"  🔴 High Similarity  : {stats['high_matches']}")
    p(f"  🟠 Moderate         : {stats['moderate_matches']}")
    p(f"  ⚪ Low              : {stats['low_matches']}")

    icon = {"HIGH": "🔴", "MODERATE": "🟠", "LOW": "⚪"}

    cmat = data.get("comparison_matrix", [])
    if cmat:
        p(f"\n[CONTEXTUAL ANALYSIS — TOP MATCHES  (AI-enriched, grouped by patent)]")
        groups = defaultdict(list)
        for entry in cmat:
            groups[entry.get("patent_number", "")].append(entry)

        for pat_num, entries in groups.items():
            best = max(entries, key=lambda x: x.get("best_score", x.get("similarity_score", 0)))
            overall = best.get("overall_confidence", "LOW")
            best_score = best.get("best_score", best.get("similarity_score", 0))
            feat_names = [e.get("feature_label", "") for e in entries]
            n_feat = len(feat_names)
            div_flag = best.get("divergence_flag", False)
            div_prefix = "⚠️  " if div_flag else ""
            ic = icon.get(overall, "⚪")

            p(f"\n  {div_prefix}{ic} {pat_num}  —  Best: {best_score:.3f} ({overall})  —  "
              f"{n_feat} feature(s): {', '.join(feat_names[:3])}"
              + (f" (+{n_feat-3} more)" if n_feat > 3 else ""))

            if div_flag:
                p(f"     ⚠️  DIVERGENCE: {best.get('divergence_note','')[:180]}")

            for e in entries:
                conf = e.get("overall_confidence", "LOW")
                p(f"    {icon.get(conf,'⚪')}  {e.get('feature_label','')}  —  "
                  f"Claim {e.get('claim_number','?')}  —  "
                  f"Score: {e.get('similarity_score',0):.3f}  ({conf})")
                elem = e.get("element_text", "")
                if elem:
                    p(f"       Element: {elem[:160]}")

            gem_expl = best.get("gemini_explanation", "")
            gem_asmt = best.get("gemini_assessment", "")
            if gem_expl:
                p(f"\n     📋 What this claim legally requires:")
                p(f"     {gem_expl[:300]}")
            if gem_asmt:
                p(f"\n     🔬 Technical Comparison:")
                p(f"     {gem_asmt[:300]}")
            dists = best.get("key_distinctions", [])
            if dists:
                p(f"\n     Key Distinctions:")
                for d in dists[:5]:
                    p(f"       - {d}")
            cannot = best.get("cannot_determine", "")
            if cannot:
                p(f"\n     Cannot Determine Without Expert Review: {cannot[:200]}")

    low_only = [m for m in matches if m["similarity_level"] == "LOW"]
    if low_only:
        p(f"\n  ⚪ {len(low_only)} low-similarity match(es) (embedding-only, not analysed by Gemini)")
        for m in low_only[:10]:
            p(f"     {m['feature_label']}  |  {m['patent_number']}  |  score {m['similarity_score']:.3f}")

    unmatched = sim.get("unmatched_features", [])
    if unmatched:
        p(f"\n[FEATURES WITH NO STRONG PRIOR ART MATCH]")
        for uf in unmatched:
            p(f"  ✅ {uf['label']} — {uf.get('description','')}")
    else:
        p(f"\n  ✅ All features have at least one HIGH or MODERATE match in the retrieved patents.")

    p(f"\n[PARSED CLAIM DETAIL  — all 20 patents]")
    for pc in pc_list:
        pub = pc.get("patent_number", "")
        ind = pc.get("independent_claims", [])
        total = pc.get("total_claims_found", 0)
        conf = pc.get("parsing_confidence", "")
        cc = {"HIGH": "🟢", "MODERATE": "🟡", "LOW": "🔴"}.get(conf, "")
        p(f"\n  {cc} {pub}  —  {len(ind)} independent claim(s) of {total} total  [{conf}]")
        for cl in ind:
            preamble = cl.get("preamble", "")
            transition = cl.get("transitional_phrase", "")
            elements = cl.get("elements", [])
            plain = cl.get("plain_english", "")
            p(f"    Claim {cl.get('claim_number','?')}: {preamble} {transition}")
            for el in elements:
                p(f"      [{el['id']}] {el['text']}")
            if plain:
                p(f"      Plain English: {plain}")

    p(f"\n[WHITE SPACE ANALYSIS]")
    ws = data.get("white_spaces", [])
    if not ws:
        p("  (No white spaces identified — all features showed moderate/high similarity to existing patents.)")
    else:
        p(f"  Found {len(ws)} potential white-space area(s). Research signals, not legal findings.")
        ws_icon = {"HIGH": "🟢", "MODERATE": "🟡", "LOW": "🔴", "INSUFFICIENT": "⚫"}
        type_icon = {"Feature Gap": "🔍", "Classification Gap": "📂",
                     "Classification Density": "📂", "Combination Novelty": "✨"}
        for w in ws:
            conf_d = w.get("confidence", {})
            cl = conf_d.get("level", "LOW")
            ci = ws_icon.get(cl, "⚫")
            ti = type_icon.get(w["type"], "🔹")
            p(f"\n  {ti} {w['type']}  {ci} {cl}  —  {w['title']}")
            p(f"     {w['description'][:300]}")
            bp = w.get("boundary_patents", [])
            if bp:
                names = [str(b.get("patent", "") if isinstance(b, dict) else b) for b in bp[:4]]
                p(f"     Nearest Prior Art: {', '.join(names)}")
            p(f"     Confidence: {cl} — {conf_d.get('rationale','')[:160]}")
            p(f"     Data completeness: {w.get('data_completeness','')[:120]}")
            p(f"     ⚠️  {w.get('disclaimer','')[:160]}")

    # TAB 3 — Landscape
    p(f"\n{SEP}")
    p("TAB 3: PATENT LANDSCAPE")
    p(SEP)

    lp_list = data["landscape_patents"]
    ldf = pd.DataFrame(lp_list)
    n_lp = len(ldf)
    p(f"\n  Analysis based on {n_lp} patents from the USPTO database.")

    ldf["filing_year"] = ldf["filing_date"].apply(
        lambda x: int(str(x)[:4]) if x else None)
    trends = ldf.groupby("filing_year").size().reset_index(name="count")
    max_count = int(trends["count"].max())

    p(f"\n[CHART: Filing Trends by Year]")
    for _, r in trends.iterrows():
        bar = "█" * int(r["count"] / max_count * 40)
        p(f"  {int(r['filing_year'])}: {bar} {int(r['count'])}")

    ldf["primary_assignee"] = ldf["assignee_name"].apply(
        lambda x: x[0] if isinstance(x, list) and x else str(x) if x else "Unknown")
    top_a = ldf["primary_assignee"].value_counts().head(15)
    p(f"\n[CHART: Top Patent Holders (ranked by # patents)]")
    for asgn, cnt in top_a.items():
        bar = "█" * int(cnt / top_a.iloc[0] * 30)
        p(f"  {asgn:<35} {bar} {cnt}")

    cpc_flat = []
    for codes in ldf["cpc_code"]:
        if isinstance(codes, list) and codes:
            cpc_flat.append(codes[0])
    cpc_counts = Counter(cpc_flat).most_common(10)
    p(f"\n[CHART: Technology Classification Distribution (CPC sections)]")
    total_cpc = sum(c for _, c in cpc_counts)
    for code, cnt in cpc_counts:
        pct = cnt / total_cpc * 100
        bar = "█" * int(pct / 100 * 30)
        p(f"  {code:<12} {bar} {cnt} ({pct:.1f}%)")

    cpc_available = sum(1 for codes in ldf["cpc_code"]
                        if isinstance(codes, list) and codes)
    p(f"\n  CPC data available for {cpc_available} of {n_lp} patents.")

    # TAB 4 — Report
    p(f"\n{SEP}")
    p("TAB 4: REPORT")
    p(SEP)

    n_pat = len(dp_list)
    high = stats["high_matches"]
    mod = stats["moderate_matches"]
    ws_n = len(data.get("white_spaces", []))

    p(f"\n[Phase Status]")
    p(f"  ✅ Patent Retrieval")
    p(f"  ✅ Claim Parsing")
    p(f"  ✅ Similarity Analysis")
    p(f"  ✅ Landscape Analysis")
    p(f"  ✅ White-Space Finder")

    p(f"\n[Executive Summary Preview]")
    p(f"  PatentScout retrieved and analysed {n_pat} patents from the Google BigQuery")
    p(f"  Patents database. Similarity scoring found {high} high-overlap and")
    p(f"  {mod} moderate-overlap match(es). White-space analysis identified")
    p(f"  {ws_n} potential opportunity area(s).")

    p(f"\n  [PDF report available for download — covers all sections above]")

    return "\n".join(out)


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for demo_name, builder in [
        ("SOLAR CHARGER", build_solar_demo_data),
        ("SMART DOORBELL (Text + Sketch)", build_doorbell_demo_data),
    ]:
        print(fmt_demo(demo_name, builder()))
        print()
