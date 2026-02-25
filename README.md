# PatentScout

Automated patent landscape analysis — from invention description to downloadable PDF report in under 3 minutes.

**[Live Demo](https://patentscout.streamlit.app)**

---

## What It Does

PatentScout helps inventors and product teams understand the patent landscape around a new idea before investing in development. Describe your invention in plain English, and PatentScout queries **100 million+ patents** from Google's BigQuery public dataset, maps your invention's features against existing claim language using sentence-transformer embeddings and Gemini AI analysis, visualises the technology landscape, and generates a professional 8-section PDF report — all in a single Streamlit session.

## Screenshot

> *(Add a screenshot after deployment: drag a PNG into this section or use `![screenshot](assets/screenshot.png)`)*

---

## Technology Stack

| Layer | Technology |
|---|---|
| UI | Streamlit 1.54 |
| Feature Extraction | Google Gemini 1.5 Flash (via google-genai) |
| **Patent Database** | **Google BigQuery — patents-public-data (100M+ patents)** |
| Semantic Similarity | sentence-transformers / all-MiniLM-L6-v2 |
| Contextual AI Analysis | Google Gemini 1.5 Flash |
| Visualisation | Plotly |
| PDF Report | ReportLab |
| Infrastructure | Streamlit Community Cloud |

---

## Architecture

```
User Input (text + optional image)
        │
        ▼
  QueryBuilder ──► Gemini ──► features[], cpc_codes[], search_terms[]
        │
        ▼
  PatentRetriever ──► BigQuery (patents-public-data) ──► detail_df, landscape_df
        │                                                      │
        ▼                                                      ▼
  ClaimParser ──► Gemini                             LandscapeAnalyzer ──► Plotly charts
        │
        ▼
  EmbeddingEngine ──► cosine similarity matrix
        │
        ▼
  ElementMapper ──► Gemini ──► enriched comparison_matrix[]
        │
        ▼
  WhiteSpaceFinder ──► Gemini ──► white_spaces[]
        │
        ▼
  ReportGenerator ──► ReportLab ──► PDF bytes ──► st.download_button
```

---

## How To Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/patentscout.git
cd patentscout
```

### 2. Set up a Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set up Google Cloud and BigQuery

PatentScout queries the freely available `patents-public-data.patents.publications` BigQuery dataset. No data upload is required.

1. Create a [Google Cloud project](https://console.cloud.google.com)
2. Enable the **BigQuery API**
3. Create a **Service Account** with roles `BigQuery Job User` + `BigQuery Data Viewer`
4. Download the JSON key and save as `credentials/service-account.json`
5. Set up a billing alert at $0 to protect against unexpected charges
   *(Free tier: 1 TB of queries/month — PatentScout uses ~0.5–2 GB per query)*

### 4. Configure environment variables

Create a `.env` file:

```
GEMINI_API_KEY=your-gemini-api-key
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_APPLICATION_CREDENTIALS=credentials/service-account.json
```

Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com).

### 5. Run connectivity tests and launch

```bash
python tests/test_connections.py
streamlit run app.py
```

---

## Deploying to Streamlit Community Cloud

1. **Push this repository** to a public GitHub repo (credentials are in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select your repo
3. Set the **Main file path** to `app.py`
4. Open **Advanced settings → Secrets** and paste:

```toml
[general]
GEMINI_API_KEY = "your-key-here"

[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "...@....iam.gserviceaccount.com"
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "..."
universe_domain = "googleapis.com"
```

> See [`.streamlit/secrets.toml.template`](.streamlit/secrets.toml.template) for the full template.

5. Click **Deploy**. First load may take 1–2 minutes while the embedding model downloads.

---

## Project Structure

```
patentscout/
├── app.py                      # Streamlit entry point
├── requirements.txt            # Pinned Python dependencies
├── packages.txt                # System packages for Streamlit Cloud
├── .env                        # Local env vars (NOT in git)
├── .gitignore
├── README.md
├── .streamlit/
│   ├── config.toml             # Theme and server settings
│   └── secrets.toml.template   # Secrets template (NOT in git)
├── config/
│   ├── settings.py             # Constants and configuration
│   └── prompts.py              # Gemini prompt templates
├── models/
│   └── schemas.py              # Dataclass schemas
├── modules/
│   ├── query_builder.py        # Gemini feature extraction + BigQuery SQL
│   ├── patent_retriever.py     # BigQuery data retrieval
│   ├── claim_parser.py         # Patent claim parsing via Gemini
│   ├── embedding_engine.py     # Sentence-transformer embeddings
│   ├── element_mapper.py       # Claim element mapping via Gemini
│   ├── landscape_analyzer.py   # Patent landscape visualisation
│   ├── whitespace_finder.py    # IP white-space detection
│   └── report_generator.py     # PDF report generation
├── assets/
│   └── report_styles.py        # ReportLab styles and colour palette
├── tests/
│   └── test_connections.py     # BigQuery + Gemini connectivity test
├── examples/
│   ├── sample_description.txt  # Test Case 1 invention description
│   └── sample_report.pdf       # Sample output PDF
└── credentials/                # GCP service account key (NOT in git)
    └── service-account.json
```

---

## Key Design Decisions

**Two-layer similarity scoring:** Embedding cosine similarity alone produces noisy results on patent language (which differs significantly from natural-language descriptions). Adding a Gemini contextual analysis pass on the top-N matches dramatically reduces false positives and surfaces genuine technical distinctions.

**BigQuery over PatentsView/USPTO APIs:** The Google Patents BigQuery public dataset contains 100M+ publications with structured CPC codes, full claim text, and assignee data accessible via standard SQL. No rate limiting, no API key required for the dataset itself, and queries return in 10–30 seconds.

**Capped query cost:** Every BigQuery job is capped at 5 GB billed (`maximum_bytes_billed`). This ensures a single bad query cannot exceed the free-tier monthly allowance.

**Graceful degradation:** Each pipeline phase has independent error handling. If one phase fails (e.g. Gemini rate limit), the app shows what succeeded and continues with remaining phases. PDF generation works with whatever data is available.

---

## Sample Test Cases

| # | Description | Expected CPC Areas |
|---|---|---|
| 1 | Foldable solar charger with USB-C and battery pack | H02J7/35, H01L31/042, H02S10/30 |
| 2 | Smart doorbell with facial recognition and auto-unlock | H04N7/18, G06V40/16, E05B47/00 |
| 3 | Bicycle helmet with bone conduction audio + air quality + gesture turn signals | A42B3/04, H04R1/10, G01N33/00 |

See [examples/sample_description.txt](examples/sample_description.txt) for the full Test Case 1 input and [examples/sample_report.pdf](examples/sample_report.pdf) for a sample output.

---

## Limitations

- **Preliminary research tool, not legal advice.** All findings require review by a qualified patent attorney before any business or legal decisions are made.
- **US patent focus.** This analysis primarily covers US grants and publications in the `patents-public-data` dataset.
- **Embedding similarity is approximate.** Semantic similarity between natural language and patent claim language is inherently imperfect. Results should be validated manually.
- **LLM analysis may contain errors.** Gemini outputs are heuristic and should not be relied upon for claim construction or infringement analysis.
- **Query cost varies.** Complex descriptions may generate broader WHERE clauses that scan more data. The 5 GB cap prevents runaway costs but may limit recall on niche technologies.

---

## Future Roadmap

- Citation network analysis (forward/backward citations)
- International patent search expansion (EP, WO, CN)
- Interactive claim highlighting with element-level colour coding
- Watchlist / alerts for new filings in a technology area
- Prosecution history retrieval via USPTO PAIR API

---

## Disclaimer

This tool is an automated preliminary research summary. It is **NOT** a legal opinion, freedom-to-operate analysis, patentability opinion, or infringement analysis. All findings should be reviewed and validated by a qualified patent professional before any business or legal decisions are made.

