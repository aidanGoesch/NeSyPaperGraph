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
- Docling-first PDF parsing + metadata extraction (title/authors/date) with heuristic fallback
- LLM-based topic extraction
- Graph generation and analysis
- Paper recommendation system (stretch goal)

## Setup
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## Docling Ingest Parsing
- Ingest parsing now uses Docling first for text + metadata extraction, then falls back to legacy parsing and local heuristic metadata if Docling fails.
- Relevant env vars:
  - `DOCLING_ENABLED=true`
  - `DOCLING_MAX_PAGES=2`
  - `DOCLING_MAX_TEXT_CHARS=8000`
- Rollback toggle: set `DOCLING_ENABLED=false` to use legacy parsing path only.
