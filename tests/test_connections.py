"""
PatentScout — Connection Tests

Verifies connectivity to all external services used by the application:
  1. Environment variables (.env)
  2. Gemini API
  3. BigQuery (patents-public-data public dataset)
  4. Sentence-transformers embedding model

Run with:
    python tests/test_connections.py
"""

from __future__ import annotations

import os
import sys
import time


# Ensure project root is on the path when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def _result(label: str, passed: bool, elapsed: float, detail: str = "") -> None:
    status = PASS if passed else FAIL
    msg = f"[{status}] {label} ({elapsed:.2f}s)"
    if detail:
        msg += f"\n       {detail}"
    print(msg)


# ---------------------------------------------------------------------------
# Test 1 — Environment variables
# ---------------------------------------------------------------------------
def test_env_vars() -> bool:
    t0 = time.time()
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    gcp_project = os.getenv("GOOGLE_CLOUD_PROJECT", "")

    missing = []
    if not gemini_key or gemini_key.startswith("<"):
        missing.append("GEMINI_API_KEY")
    if not gcp_project or gcp_project.startswith("<"):
        missing.append("GOOGLE_CLOUD_PROJECT")

    passed = len(missing) == 0
    detail = f"Missing / placeholder values: {missing}" if missing else \
             f"GEMINI_API_KEY=...{gemini_key[-4:]}  |  GOOGLE_CLOUD_PROJECT={gcp_project}"
    _result("ENV VARS", passed, time.time() - t0, detail)
    return passed


# ---------------------------------------------------------------------------
# Test 2 — Gemini API
# ---------------------------------------------------------------------------
def test_gemini() -> bool:
    t0 = time.time()
    try:
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key or api_key.startswith("<"):
            raise ValueError("GEMINI_API_KEY not set")

        client = genai.Client(api_key=api_key)
        models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
        response = None
        used_model = None
        last_exc = None
        for m in models_to_try:
            try:
                response = client.models.generate_content(
                    model=m,
                    contents="Reply with the single word: CONNECTED",
                )
                used_model = m
                break
            except Exception as e:
                last_exc = e
                continue
        if response is None:
            raise last_exc
        text = response.text.strip()
        passed = bool(text)
        _result("GEMINI API", passed, time.time() - t0, f"Model: {used_model}  |  Response: {text[:80]}")
        return passed
    except Exception as exc:
        _result("GEMINI API", False, time.time() - t0, str(exc))
        return False


# ---------------------------------------------------------------------------
# Test 3 — BigQuery (patents-public-data)
# ---------------------------------------------------------------------------
def test_bigquery() -> bool:
    t0 = time.time()
    try:
        from google.cloud import bigquery

        project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
        if not project or project.startswith("<"):
            raise ValueError("GOOGLE_CLOUD_PROJECT not set")

        credentials_path = os.getenv(
            "GOOGLE_APPLICATION_CREDENTIALS", "credentials/service_account.json"
        )
        # Resolve relative to project root
        root = os.path.join(os.path.dirname(__file__), "..")
        abs_creds = os.path.abspath(os.path.join(root, credentials_path))
        if os.path.exists(abs_creds):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = abs_creds

        client = bigquery.Client(project=project)

        sql = """
        SELECT
          publication_number,
          title_localized.text AS title
        FROM
          `patents-public-data.patents.publications`,
          UNNEST(title_localized) AS title_localized
        WHERE
          title_localized.language = 'en'
          AND country_code = 'US'
          AND publication_number = 'US-7479949-B2'
        LIMIT 1
        """

        rows = list(client.query(sql).result())
        passed = len(rows) > 0
        detail = f"publication_number={rows[0].publication_number}, title={rows[0].title[:60]}" \
                 if passed else "No rows returned"
        _result("BIGQUERY", passed, time.time() - t0, detail)
        return passed
    except Exception as exc:
        _result("BIGQUERY", False, time.time() - t0, str(exc))
        return False


# ---------------------------------------------------------------------------
# Test 4 — Sentence-Transformers
# ---------------------------------------------------------------------------
def test_embeddings() -> bool:
    t0 = time.time()
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        vector = model.encode(["This is a test sentence for PatentScout."])
        passed = vector is not None and len(vector[0]) > 0
        detail = f"Embedding shape: {vector.shape}  |  dim={vector.shape[1]}"
        _result("SENTENCE-TRANSFORMERS", passed, time.time() - t0, detail)
        return passed
    except Exception as exc:
        _result("SENTENCE-TRANSFORMERS", False, time.time() - t0, str(exc))
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n===== PatentScout — Connection Tests =====\n")

    results = {
        "ENV VARS":              test_env_vars(),
        "GEMINI API":            test_gemini(),
        "BIGQUERY":              test_bigquery(),
        "SENTENCE-TRANSFORMERS": test_embeddings(),
    }

    total = len(results)
    passed = sum(results.values())

    print(f"\n{'='*42}")
    print(f"Results: {passed}/{total} tests passed")
    if passed < total:
        failed = [k for k, v in results.items() if not v]
        print(f"Failed:  {', '.join(failed)}")
    print("=" * 42)
    sys.exit(0 if passed == total else 1)
