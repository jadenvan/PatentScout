#!/usr/bin/env python3
"""Probe actual BigQuery column costs for the patents table."""
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
        return bp, bb, len(df)
    except Exception as exc:
        print(f"  {label}: FAILED — {exc}")
        return None, None, 0

# 1) Count only (free)
probe("count-only", """
SELECT COUNT(*) AS cnt
FROM `patents-public-data.patents.publications`
WHERE country_code = 'US'
""")

# 2) CPC-only scout: publication_number + CPC filter (no text)
probe("cpc-scout", """
SELECT publication_number
FROM `patents-public-data.patents.publications`
WHERE country_code = 'US'
  AND grant_date > 0
  AND filing_date > 20100101
  AND EXISTS(SELECT 1 FROM UNNEST(cpc) AS c
             WHERE c.code LIKE 'H02S40%' OR c.code LIKE 'H02J7%' OR c.code LIKE 'H01L31%')
LIMIT 500
""")

# 3) CPC scout + title (no abstract)
probe("cpc+title", """
WITH base AS (
  SELECT
    pub.publication_number,
    (SELECT t.text FROM UNNEST(pub.title_localized) t WHERE t.language='en' LIMIT 1) AS pub_title,
    pub.filing_date, pub.grant_date, pub.publication_date, pub.country_code, pub.family_id
  FROM `patents-public-data.patents.publications` AS pub
  WHERE pub.country_code = 'US'
    AND pub.grant_date > 0
    AND pub.filing_date > 20100101
    AND EXISTS(SELECT 1 FROM UNNEST(cpc) AS c
               WHERE c.code LIKE 'H02S40%' OR c.code LIKE 'H02J7%' OR c.code LIKE 'H01L31%')
)
SELECT * FROM base WHERE pub_title IS NOT NULL
ORDER BY publication_date DESC LIMIT 200
""")

# 4) Full CPC + title + abstract
probe("cpc+title+abstract", """
WITH base AS (
  SELECT
    pub.publication_number,
    (SELECT t.text FROM UNNEST(pub.title_localized) t WHERE t.language='en' LIMIT 1) AS pub_title,
    (SELECT a.text FROM UNNEST(pub.abstract_localized) a WHERE a.language='en' LIMIT 1) AS pub_abstract,
    pub.filing_date, pub.grant_date, pub.publication_date, pub.country_code, pub.family_id
  FROM `patents-public-data.patents.publications` AS pub
  WHERE pub.country_code = 'US'
    AND pub.grant_date > 0
    AND pub.filing_date > 20100101
    AND EXISTS(SELECT 1 FROM UNNEST(cpc) AS c
               WHERE c.code LIKE 'H02S40%' OR c.code LIKE 'H02J7%' OR c.code LIKE 'H01L31%')
)
SELECT * FROM base WHERE pub_title IS NOT NULL
ORDER BY publication_date DESC LIMIT 200
""")

# 5) No CPC, text regex on abstract only (how expensive is the regex?)
probe("regex-only-no-cpc", """
WITH base AS (
  SELECT
    pub.publication_number,
    (SELECT t.text FROM UNNEST(pub.title_localized) t WHERE t.language='en' LIMIT 1) AS pub_title,
    (SELECT a.text FROM UNNEST(pub.abstract_localized) a WHERE a.language='en' LIMIT 1) AS pub_abstract,
    pub.filing_date, pub.grant_date, pub.publication_date, pub.country_code, pub.family_id
  FROM `patents-public-data.patents.publications` AS pub
  WHERE pub.country_code = 'US'
    AND pub.grant_date > 0
    AND pub.filing_date > 20150101
)
SELECT * FROM base
WHERE pub_title IS NOT NULL
  AND REGEXP_CONTAINS(pub_abstract, r'(?i)(solar|photovoltaic)')
ORDER BY publication_date DESC LIMIT 200
""")

print("\nDone.")
