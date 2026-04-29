import signal
import sys
from contextlib import asynccontextmanager
import asyncio
import secrets
from urllib.parse import urlsplit
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from api.graph import router as graph_router, start_queue_worker, stop_queue_worker
from api.workspace import router as workspace_router
from services.storage_service import load_graph
from services.observability import log_memory, timed_block
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def signal_handler(sig, frame):
    print('\nShutting down gracefully...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

class SearchRequest(BaseModel):
    query: str


def prune_jobs(jobs: dict) -> None:
    max_jobs = int(os.getenv("MAX_JOB_HISTORY", "200") or "200")
    ttl_seconds = int(os.getenv("JOB_TTL_SECONDS", "3600") or "3600")
    if not jobs:
        return

    import time

    now = time.time()

    expired = []
    for job_id, job in jobs.items():
        finished_at = job.get("finished_at")
        if finished_at and now - finished_at > ttl_seconds:
            expired.append(job_id)
    for job_id in expired:
        jobs.pop(job_id, None)

    if max_jobs > 0 and len(jobs) > max_jobs:
        sorted_jobs = sorted(
            jobs.items(),
            key=lambda item: item[1].get("started_at", 0),
        )
        for job_id, _ in sorted_jobs[: len(jobs) - max_jobs]:
            jobs.pop(job_id, None)

def get_current_graph(app: FastAPI):
    """Get the most recent graph object."""
    graph_obj = getattr(app.state, "graph", None)
    try:
        if graph_obj is None:
            with timed_block("load_graph_for_search"):
                graph_obj = load_graph()
            app.state.graph = graph_obj
        if graph_obj is not None:
            return graph_obj
    except Exception as e:
        print(f"Could not load saved graph: {e}")
    
    try:
        # Fallback to dummy graph
        from services.graph_builder import create_dummy_graph

        return create_dummy_graph()
    except Exception as e:
        print(f"Could not load dummy graph: {e}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.jobs = {}
    app.state.graph = None
    app.state.graph_lock = asyncio.Lock()
    app.state.upload_queue = asyncio.Queue()
    app.state.queue_worker_task = None
    app.state.event_subscribers = set()
    app.state.agent = None
    app.state.agent_graph_identity = None
    start_queue_worker(app)
    log_memory("startup_initialized_state")

    yield

    await stop_queue_worker(app)


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def enforce_access_key(request: Request, call_next):
    """
    Optional shared-key guard for API routes.
    Set APP_ACCESS_KEY in env to require clients to send X-Access-Key.
    """
    required_key = os.environ.get("APP_ACCESS_KEY", "").strip()
    path = request.url.path

    # Leave health checks and CORS preflight unauthenticated.
    if (
        not required_key
        or request.method == "OPTIONS"
        or path == "/health"
        or not path.startswith("/api/")
    ):
        return await call_next(request)

    provided_key = request.headers.get("x-access-key", "").strip()
    if not provided_key and path == "/api/graph/stream":
        provided_key = request.query_params.get("access_key", "").strip()
    if not provided_key:
        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            provided_key = auth_header[7:].strip()

    if not secrets.compare_digest(provided_key, required_key):
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    return await call_next(request)

configured_origins = [
    origin.strip()
    for origin in os.environ.get("FRONTEND_URL", "http://localhost:3000").split(",")
    if origin.strip()
]


def normalize_origin(origin: str) -> str:
    cleaned = origin.strip().strip('"').strip("'").rstrip("/")
    parts = urlsplit(cleaned)
    if parts.scheme and parts.netloc:
        return f"{parts.scheme}://{parts.netloc}"
    return cleaned


configured_origins = [normalize_origin(origin) for origin in configured_origins]
configured_origins = [origin for origin in configured_origins if origin]
configured_origins = list(dict.fromkeys(configured_origins))
configured_origin_regex = os.environ.get("FRONTEND_URL_REGEX", "").strip() or None

app.add_middleware(
    CORSMiddleware,
    allow_origins=configured_origins,
    allow_origin_regex=configured_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(graph_router, prefix="/api")
app.include_router(workspace_router, prefix="/api")

@app.get("/api/agent/architecture")
def get_agent_architecture():
    # Import lazily to avoid loading LangGraph stack at startup.
    from services.question_agent import QuestionAgent

    mermaid_diagram = QuestionAgent.get_agent_architecture_diagram()
    return {"mermaid": mermaid_diagram, "status": "success"}

@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    prune_jobs(app.state.jobs)
    job = app.state.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("queued"):
        queue_items = list(getattr(app.state.upload_queue, "_queue", []))
        queued_job_ids = [item[0] for item in queue_items if item]
        if job_id in queued_job_ids:
            job["queue_position"] = queued_job_ids.index(job_id) + 1
        elif job.get("status") == "pending":
            job["queue_position"] = 0
    return job

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/api/search")
async def search(request: SearchRequest):
    # Import lazily to avoid loading LangGraph stack at startup.
    from services.question_agent import QuestionAgent

    with timed_block("search_graph_fetch"):
        current_graph_obj = get_current_graph(app)
    current_graph_identity = id(current_graph_obj) if current_graph_obj is not None else None
    if app.state.agent is None or app.state.agent_graph_identity != current_graph_identity:
        app.state.agent = QuestionAgent(None, current_graph_obj)
        app.state.agent_graph_identity = current_graph_identity
        log_memory("search_agent_rebuilt")
    
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
