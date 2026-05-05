# BigQuery NL2SQL Chatbot

A natural language interface for querying BigQuery datasets using Gemini Pro and LangChain. Ask a question in plain English — the system generates BigQuery SQL, executes it, and returns a human-readable answer.

Built as a POC for the LLM chatbot shipped to production at Walmart Labs, where it serves ~100 business queries/day over a 550M-SKU inventory dataset.

---

## What it does

Users type questions like *"Which product categories had the most returns last quarter?"* or *"Show me the top 10 SKUs by revenue in the electronics department"* — no SQL required. The system inspects the dataset schema, generates a valid BigQuery SQL query via Gemini Pro, executes it, and synthesizes a plain-English answer.

---

## Architecture

```
User Question
     │
     ▼
┌──────────────────────────┐
│  Streamlit UI  (app.py)  │  ← chat interface, sidebar config
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  GenAIChain              │  ← LangChain + Gemini Pro
│  (src/llm_logic.py)     │     NL → SQL generation
│                          │     SQL result → NL answer
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  BigQueryClient          │  ← google-cloud-bigquery
│  (src/bq_client.py)     │     SQLAlchemy connection URI
│                          │     query execution
└──────────────────────────┘
```

**Pipeline for each query:**
1. User submits a natural language question via the Streamlit chat input
2. `GenAIChain` uses `create_sql_query_chain` (LangChain) to inspect the BigQuery schema and generate SQL via Gemini Pro
3. SQL is cleaned and executed against BigQuery via `sqlalchemy-bigquery`
4. Raw results are passed back to Gemini Pro, which synthesizes a plain-English answer
5. The question, generated SQL, raw result, and answer are all surfaced in the UI

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Gemini Pro (`gemini-pro`) via `langchain-google-genai` |
| Orchestration | LangChain (`create_sql_query_chain`) |
| Database | Google BigQuery |
| ORM / Connection | SQLAlchemy + `sqlalchemy-bigquery` |
| UI | Streamlit |
| Auth | GCP Service Account JSON |

---

## Prerequisites

- Python 3.11+
- Google Cloud project with BigQuery enabled
- Service account with BigQuery Data Viewer + Job User roles
- Google AI Studio API key (Gemini Pro)

---

## Setup

```bash
git clone https://github.com/shubhambakre/bq-genai-poc.git
cd bq-genai-poc
pip install -r requirements.txt
```

Configure via environment variables or enter values in the Streamlit sidebar at runtime:

```bash
export BQ_PROJECT_ID=your-gcp-project-id
export BQ_DATASET_ID=your-dataset-id
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
export GOOGLE_API_KEY=your-gemini-api-key
```

---

## Run

```bash
streamlit run app.py
```

Open `http://localhost:8501`, fill in the sidebar config, click **Connect**, and start querying.

---

## Project Structure

```
bq-genai-poc/
├── app.py              # Streamlit UI: chat interface, sidebar, session state
├── requirements.txt
└── src/
    ├── bq_client.py    # BigQueryClient: connection management, schema inspection, query execution
    └── llm_logic.py    # GenAIChain: SQL generation, execution, answer synthesis via Gemini Pro
```

---

## POC Notes

This is a proof-of-concept extracted from a production system. The production version at Walmart adds:

- Schema caching and domain-specific prompt tuning over a 550M-SKU inventory dataset
- Query validation and BigQuery cost guardrails before execution
- Role-based access control via GCP IAM
- Query audit logging, monitoring, and alerting

---

*Stack: Python · LangChain · Gemini Pro · Google BigQuery · Streamlit · GCP*
