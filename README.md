# NeSy Paper Graph

An intelligent research paper analysis system that automatically extracts topics, builds knowledge graphs, and enables natural language querying using LLM-powered agents.

![Application Screenshot](./screenshot.png)
*Interactive graph visualization with AI-powered search capabilities*

## Overview

NeSy Paper Graph processes academic PDFs to create an interactive bipartite graph connecting papers to their core topics. The system uses LangGraph agents to answer complex questions about the research corpus, automatically traversing the knowledge graph to find relevant information.

## Technical Achievements

### 🧠 LangGraph Agent Architecture
- **Multi-node reasoning pipeline** using LangGraph's StateGraph for structured question answering
- **Dynamic graph traversal** that intelligently navigates paper-topic relationships
- **Conversational memory** maintaining context across multiple queries
- **Automated citation generation** linking answers back to source papers

### 🔗 Intelligent Graph Construction
- **Bipartite graph validation** ensuring structural integrity between papers and topics
- **Semantic topic merging** using sentence-transformers (all-MiniLM-L6-v2) to identify and consolidate similar topics
- **Incremental graph building** allowing papers to be added without rebuilding the entire graph
- **Topic synonym detection** preventing duplicate topics across uploads

### 🎨 Interactive Visualization
- **Force-directed graph layout** using D3.js for intuitive exploration
- **Real-time path highlighting** showing agent reasoning chains through the graph
- **Responsive node interactions** with zoom, pan, and focus capabilities
- **Dark mode support** for extended research sessions

### ⚡ Modern Full-Stack Architecture
- **FastAPI backend** with async request handling and CORS middleware
- **React frontend** with hooks-based state management
- **RESTful API design** for graph operations and agent queries
- **Persistent graph storage** using pickle serialization

## Tech Stack

**Backend:**
- FastAPI + Uvicorn
- LangGraph + LangChain (OpenAI integration)
- NetworkX (graph data structures)
- Sentence-Transformers (semantic similarity)
- PyPDF (document parsing)

**Frontend:**
- React 18
- D3.js (graph visualization)
- Mermaid (agent architecture diagrams)
- React Markdown (formatted responses)

## Setup

### Prerequisites
- Python 3.10+
- Node.js 16+
- OpenAI API key

### Backend Setup
```bash
cd backend
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add required variables to .env:
# OPENAI_API_KEY
# AWS_ACCESS_KEY_ID
# AWS_SECRET_ACCESS_KEY
# AWS_REGION
# S3_BUCKET_NAME
# FRONTEND_URL

# Start server
uvicorn main:app --reload
```

Backend runs on `http://localhost:8000`

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

Frontend runs on `http://localhost:3000`

### Docker Compose (Local Cloud-Parity)
```bash
docker compose up --build
```

Backend runs on `http://localhost:8000`, frontend on `http://localhost:3000`.

## Usage

1. **Upload Papers**: Upload PDF research papers into the interface
2. **Explore Graph**: Interact with the bipartite graph to see paper-topic relationships
3. **Ask Questions**: Use natural language to query the research corpus
4. **Follow Citations**: Click paper references in answers to navigate to source papers

Uploads are now asynchronous. The frontend submits files, receives a `job_id`, and polls `/api/jobs/{job_id}` until processing completes.

## Key Features

- **Semantic Search**: Ask questions like "What papers discuss neural-symbolic integration?"
- **Topic Clustering**: Automatically groups papers by shared research themes
- **Agent Transparency**: View the reasoning path the agent took to answer your question
- **Incremental Updates**: Add new papers without losing existing graph structure

## Architecture

The system uses a three-layer architecture:
1. **Data Layer**: NetworkX graph with persistent storage
2. **Agent Layer**: LangGraph state machine for question answering
3. **Presentation Layer**: React + D3.js for interactive visualization

The LangGraph agent implements a multi-step reasoning process:
- Query classification and intent detection
- Graph traversal and information retrieval
- Context synthesis and answer generation
- Citation extraction and path visualization

## Cloud Deployment

### Required Environment Variables
Define these in Railway (backend) and Vercel (frontend):

```bash
# Backend
OPENAI_API_KEY=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1
S3_BUCKET_NAME=
FRONTEND_URL=https://your-vercel-app.vercel.app

# Frontend (Vercel)
REACT_APP_API_URL=https://your-railway-service.railway.app
```

### Deploy Backend to Railway
1. Push repository to GitHub.
2. Create Railway project from GitHub repo.
3. Set Railway root directory to `backend/`.
4. Railway auto-detects `backend/Dockerfile`.
5. Add backend environment variables listed above.
6. Deploy and note Railway URL.

### Deploy Frontend to Vercel
1. Create Vercel project from GitHub repo.
2. Set Vercel root directory to `frontend/`.
3. Set `REACT_APP_API_URL` to Railway backend URL.
4. Deploy.

### S3 Bucket Seeding
1. Create private S3 bucket (for example `nesy-paper-graph`).
2. Create IAM credentials with bucket access.
3. Upload initial graph object:

```bash
aws s3 cp backend/storage/saved_graph.pkl s3://nesy-paper-graph/saved_graph.pkl
```

## Verification Checklist

- [ ] `GET /health` on Railway returns 200.
- [ ] CORS allows requests from Vercel frontend domain.
- [ ] PDF upload returns `job_id` immediately.
- [ ] Job status transitions `pending -> processing -> done` through `/api/jobs/{job_id}`.
- [ ] Frontend refreshes graph after job completes.
- [ ] Graph persists across backend restart/redeploy (loaded from S3).
- [ ] Sentence-transformer model loads on cold start without runtime download errors.

## Future Enhancements

- Support for additional document formats (arXiv, HTML)
- Multi-modal embeddings for figures and equations
- Collaborative graph editing and annotations
- Export capabilities (GraphML, JSON, CSV)

## License

MIT

---

*Built for exploring neurosymbolic AI research papers*
