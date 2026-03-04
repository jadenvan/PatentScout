# PatentScout

AI-powered patent landscape analysis and prior art research tool.

Takes an invention description (text and/or design sketch), queries Google's patent database, and generates a prior art research report with claim-level analysis.

## Features

- Multi-modal input (text descriptions + design sketches via Gemini Vision)
- Patent retrieval from 100M+ publications via Google BigQuery
- Two-layer analysis: embedding similarity + LLM contextual comparison
- Claim element parsing and feature mapping
- Landscape visualization (filing trends, top assignees, CPC distribution)
- White space / innovation gap analysis
- 30+ page PDF report generation

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Gemini 2.0 Flash |
| Database | Google BigQuery |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Visualization | Plotly, Matplotlib |
| PDF | ReportLab |

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/patentscout.git
cd patentscout
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit secrets.toml with your keys
streamlit run app.py
```

## Demo Mode

Click "Explore Example Analyses" in the sidebar to try pre-built demos without API keys.

## Disclaimer

Research tool for preliminary analysis. Not legal advice. Consult a patent attorney for professional guidance.
