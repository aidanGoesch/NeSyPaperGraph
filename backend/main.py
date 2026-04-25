import signal
import sys
from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from api.graph import router as graph_router
from services.question_agent import QuestionAgent
from services.storage_service import load_graph
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def signal_handler(sig, frame):
    print('\nShutting down gracefully...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

class SearchRequest(BaseModel):
    query: str

def get_current_graph_data(app: FastAPI):
    """Get the most recent graph data"""
    graph_obj = getattr(app.state, "graph", None)
    try:
        if graph_obj is None:
            graph_obj = load_graph()
            app.state.graph = graph_obj
        if graph_obj is not None:
            from api.graph import graph_to_dict
            return graph_to_dict(graph_obj), graph_obj
    except Exception as e:
        print(f"Could not load saved graph: {e}")
    
    try:
        # Fallback to dummy graph
        from api.graph import get_dummy_graph
        return get_dummy_graph(), None
    except Exception as e:
        print(f"Could not load dummy graph: {e}")
        return {
            "papers": [
                {"title": "Paper A", "topics": ["Topic 1", "Topic 2", "Topic 3"]},
                {"title": "Paper B", "topics": ["Topic 1", "Topic 2", "Topic 3"]}
            ],
            "topics": ["Topic 1", "Topic 2", "Topic 3"]
        }, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.jobs = {}
    app.state.graph = None
    app.state.graph_lock = asyncio.Lock()
    app.state.agent = None
    app.state.agent_graph_identity = None

    try:
        app.state.graph = load_graph()
    except Exception as exc:
        print(f"Could not load graph from S3 on startup: {exc}")

    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("FRONTEND_URL", "http://localhost:3000")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(graph_router, prefix="/api")

@app.get("/api/agent/architecture")
def get_agent_architecture():
    # Always return the static agent architecture diagram - no need for agent instance
    mermaid_diagram = QuestionAgent.get_agent_architecture_diagram()
    return {"mermaid": mermaid_diagram, "status": "success"}

@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    job = app.state.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/search")
async def search(request: SearchRequest):
    current_graph_data, current_graph_obj = get_current_graph_data(app)
    current_graph_identity = id(current_graph_obj) if current_graph_obj is not None else None
    if app.state.agent is None or app.state.agent_graph_identity != current_graph_identity:
        app.state.agent = QuestionAgent(current_graph_data, current_graph_obj)
        app.state.agent_graph_identity = current_graph_identity
    
    print(f"Received search query: {request.query}")
    
    try:
        answer = await app.state.agent.answer_question(request.query)
        
        # Get mermaid diagram for chat (may be None if no path)
        chat_mermaid = app.state.agent.get_mermaid_diagram()
        
        # Get sources used
        sources_used = []
        if hasattr(app.state.agent, '_last_state') and app.state.agent._last_state.get('sources_used'):
            sources_used = app.state.agent._last_state['sources_used']
        
        # Check if this is a search results response
        if answer == "SEARCH_RESULTS" and hasattr(app.state.agent, '_last_state') and app.state.agent._last_state.get('search_results'):
            return {
                "query": request.query,
                "search_results": app.state.agent._last_state['search_results'],
                "mermaid": chat_mermaid,
                "path": getattr(app.state.agent, '_last_path', None),
                "sources_used": sources_used,
                "status": "search_results"
            }
        
        return {
            "query": request.query, 
            "answer": answer, 
            "mermaid": chat_mermaid,
            "path": getattr(app.state.agent, '_last_path', None),
            "sources_used": sources_used,
            "status": "success"
        }
    except Exception as e:
        print(f"Error processing question: {e}")
        return {"query": request.query, "error": str(e), "status": "error"}
