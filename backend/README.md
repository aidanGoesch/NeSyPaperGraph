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
- LLM-based topic extraction
- Graph generation and analysis
- Paper recommendation system (stretch goal)

## Setup
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```
