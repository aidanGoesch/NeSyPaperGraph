# Backend - Paper Graph API

FastAPI backend for the paper graph analysis system using file-based storage.

## Project Structure

```
backend/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── api/                   # API route handlers
├── services/              # Business logic layer
├── models/                # Data models and schemas
├── data/                  # File-based data storage
├── utils/                 # Helper functions and utilities
├── tests/                 # Test suite
└── storage/               # File storage
    ├── uploads/           # Uploaded PDFs
    ├── processed/         # Processed data
    ├── graphs/            # Graph visualizations
    └── cache/             # Cached LLM results
```

## Data Storage
- `data/papers.json` - Paper metadata and topics
- `data/topics.json` - Topic relationships
- `data/graph.pkl` - NetworkX graph object

## Core Features
- PDF upload and text extraction
- GROBID-first metadata extraction (title/authors/date) with heuristic fallback
- LLM-based topic extraction
- Graph generation and analysis
- Paper recommendation system (stretch goal)

## Setup
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## GROBID Metadata Extraction
- Ingest metadata now uses GROBID when available, then falls back to a local heuristic parser.
- Relevant env vars:
  - `GROBID_ENABLED=true`
  - `GROBID_URL=http://localhost:8070`
  - `GROBID_TIMEOUT_SECONDS=8.0`
- When using Docker Compose, backend is configured to call `http://grobid:8070`.
