#!/usr/bin/env python3
"""Phase-2 probe: test IN-list queries on pub_numbers from CPC scout."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import settings
from google.cloud import bigquery

bq = bigquery.Client(project=settings.BIGQUERY_PROJECT)

def probe(label, sql, cap_gb=300):
    cfg = bigquery.QueryJobConfig(maximum_bytes_billed=int(cap_gb * 1e9))
    t0 = time.time()
    try:
        job = bq.query(sql, job_config=cfg)
        df = job.to_dataframe(create_bqstorage_client=False)
        bp = job.total_bytes_processed or 0
        bb = job.total_bytes_billed or 0
        elapsed = time.time() - t0
        print(f"  {label}: rows={len(df)}  processed={bp/1e9:.2f} GB  billed={bb/1e9:.2f} GB  elapsed={elapsed:.1f}s")
        return bp, bb, df
    except Exception as exc:
        print(f"  {label}: FAILED — {exc}")
        return None, None, None

# Phase 1: CPC scout (already tested at ~19GB)
print("Phase 1: CPC scout...")
_, _, scout_df = probe("cpc-scout", """
SELECT publication_number, filing_date, grant_date, publication_date, country_code, family_id
FROM `patents-public-data.patents.publications`
WHERE country_code = 'US'
  AND grant_date > 0
  AND filing_date > 20100101
  AND EXISTS(SELECT 1 FROM UNNEST(cpc) AS c
             WHERE c.code LIKE 'H02S40%' OR c.code LIKE 'H02J7%' OR c.code LIKE 'H01L31%')
LIMIT 500
""")

if scout_df is None or scout_df.empty:
    print("CPC scout failed, exiting")
    sys.exit(1)

pub_numbers = scout_df["publication_number"].tolist()
print(f"  Got {len(pub_numbers)} pub_numbers from CPC scout")
print(f"  Sample: {pub_numbers[:5]}")

# Phase 2: IN-list query for title only (no abstract)
in_list = ", ".join(f"'{p}'" for p in pub_numbers[:200])
print(f"\nPhase 2: IN-list title query ({min(200, len(pub_numbers))} pubs)...")
_, _, title_df = probe("in-list-title", f"""
SELECT
    publication_number,
    (SELECT t.text FROM UNNEST(title_localized) t WHERE t.language='en' LIMIT 1) AS pub_title
FROM `patents-public-data.patents.publications`
WHERE publication_number IN ({in_list})
""")

if title_df is not None and not title_df.empty:
    print(f"  Got {len(title_df)} titles")
    for i, (_, row) in enumerate(title_df.head(10).iterrows()):
        print(f"    {i+1}. {row.get('pub_title', '')[:80]}")

# Phase 2b: IN-list for title + abstract
print(f"\nPhase 2b: IN-list title+abstract query...")
_, _, detail_df = probe("in-list-title-abstract", f"""
SELECT
    publication_number,
    (SELECT t.text FROM UNNEST(title_localized) t WHERE t.language='en' LIMIT 1) AS pub_title,
    (SELECT a.text FROM UNNEST(abstract_localized) a WHERE a.language='en' LIMIT 1) AS pub_abstract
FROM `patents-public-data.patents.publications`
WHERE publication_number IN ({in_list})
""")

if detail_df is not None and not detail_df.empty:
    import re
    SOLAR_KW = ["solar", "photovoltaic", "charger", "battery", "usb", "power bank", "portable", "foldable"]
    hits = sum(
        1 for _, r in detail_df.head(20).iterrows()
        if any(kw in (str(r.get("pub_title","")) + " " + str(r.get("pub_abstract",""))).lower() for kw in SOLAR_KW)
    )
    print(f"\n  Solar keyword fraction in top 20: {hits}/20 = {hits/20:.0%}")

# Phase 2c: Claims via IN-list
print(f"\nPhase 2c: IN-list claims query...")
in_list_10 = ", ".join(f"'{p}'" for p in pub_numbers[:10])
probe("in-list-claims-10", f"""
SELECT
    publication_number,
    STRING_AGG(claims.text, ' | ' ORDER BY claims.text) AS claims_text
FROM `patents-public-data.patents.publications`,
     UNNEST(claims_localized) AS claims
WHERE publication_number IN ({in_list_10})
  AND claims.language = 'en'
GROUP BY publication_number
""")

print("\nDone.")
