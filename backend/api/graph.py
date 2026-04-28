from uuid import uuid4
import time
import json
import asyncio
import hashlib

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from services.graph_builder import create_dummy_graph, GraphBuilder, get_all_topics_seen
from services.storage_service import load_graph, save_graph
from services.observability import log_memory, timed_block
import traceback
import logging
import os
from typing import List, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
SSE_HEARTBEAT_SECONDS = 15


def prune_jobs(jobs: dict) -> None:
    max_jobs = int(os.getenv("MAX_JOB_HISTORY", "200") or "200")
    ttl_seconds = int(os.getenv("JOB_TTL_SECONDS", "3600") or "3600")
    if not jobs:
        return

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

def graph_to_dict(graph):
    """Convert PaperGraph to dictionary format for frontend"""
    papers = []
    edges = []
    
    for node, data in graph.graph.nodes(data=True):
        if data.get('type') == 'paper':
            paper_data = data['data']
            papers.append({
                "title": paper_data.title,
                "topics": paper_data.topics,
                "authors": paper_data.authors,
                "publication_date": paper_data.publication_date,
                "abstract": paper_data.summary,  # Use summary as abstract for frontend
                "file_path": paper_data.file_path
            })
    
    # Extract all edges and collect connected topics
    connected_topics = set()
    for source, target, edge_data in graph.graph.edges(data=True):
        edges.append({
            "source": source,
            "target": target,
            "type": edge_data.get('type', 'topic'),
            "weight": edge_data.get('weight', 1.0)
        })
        # Track which topics are actually connected
        if graph.graph.nodes[source].get('type') == 'topic':
            connected_topics.add(source)
        if graph.graph.nodes[target].get('type') == 'topic':
            connected_topics.add(target)
    
    return {
        "papers": papers,
        "topics": list(connected_topics),  # Only include topics with edges
        "edges": edges
    }


def build_graph_payload(graph) -> dict:
    graph_data = graph_to_dict(graph)
    graph_data["all_topics_seen"] = list(get_all_topics_seen())
    return graph_data


def _serialize_sse_event(event_name: str, payload: Any) -> str:
    json_payload = json.dumps(payload, default=str)
    return f"event: {event_name}\ndata: {json_payload}\n\n"


def broadcast_event(app, event_name: str, payload: dict) -> None:
    subscribers = getattr(app.state, "event_subscribers", set())
    stale_queues = []
    for subscriber_queue in list(subscribers):
        try:
            subscriber_queue.put_nowait((event_name, payload))
        except asyncio.QueueFull:
            stale_queues.append(subscriber_queue)
        except Exception:
            stale_queues.append(subscriber_queue)

    for stale_queue in stale_queues:
        subscribers.discard(stale_queue)

@router.get("/graph/dummy")
def get_dummy_graph():
    """Get dummy graph data for frontend visualization"""
    graph = create_dummy_graph()
    return graph_to_dict(graph)

async def process_pdf_job(app, job_id: str, files_data: list[tuple[str, bytes, str]]):
    """Run one queued PDF processing job and persist updates incrementally."""
    jobs = app.state.jobs
    jobs[job_id]["status"] = "processing"
    jobs[job_id]["queued"] = False
    jobs[job_id]["processing"] = True
    jobs[job_id]["error"] = None
    jobs[job_id]["started_at"] = time.time()
    jobs[job_id]["paper_total"] = len(files_data)
    jobs[job_id]["paper_index"] = 0
    jobs[job_id]["current_paper"] = None
    log_memory(f"job_{job_id}_started")
    broadcast_event(
        app,
        "job_started",
        {"job_id": job_id, "status": "processing", "paper_total": len(files_data)},
    )
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY environment variable not set")

        async with app.state.graph_lock:
            existing_graph = app.state.graph
            if existing_graph is None:
                with timed_block("load_graph_for_job"):
                    existing_graph = load_graph()

            existing_hashes = getattr(existing_graph, "paper_content_hashes", set()) if existing_graph else set()
            filtered_files = []
            progress_count = 0
            for filename, file_bytes, content_hash in files_data:
                if content_hash and content_hash in existing_hashes:
                    progress_count += 1
                    jobs[job_id]["paper_index"] = progress_count
                    jobs[job_id]["current_paper"] = filename
                    broadcast_event(
                        app,
                        "paper_processed",
                        {
                            "job_id": job_id,
                            "paper_title": filename,
                            "paper_index": progress_count,
                            "paper_total": len(files_data),
                            "status": "skipped",
                            "reason": "duplicate_hash",
                        },
                    )
                    continue
                filtered_files.append((filename, file_bytes, content_hash))

            if not filtered_files:
                if existing_graph:
                    save_graph(existing_graph)
                    app.state.graph = existing_graph
                jobs[job_id]["status"] = "done"
                jobs[job_id]["processing"] = False
                jobs[job_id]["paper_index"] = len(files_data)
                jobs[job_id]["finished_at"] = time.time()
                broadcast_event(
                    app,
                    "job_done",
                    {
                        "job_id": job_id,
                        "status": "done",
                        "paper_total": len(files_data),
                        "paper_index": len(files_data),
                    },
                )
                return

            builder = GraphBuilder()

            def on_paper_processed(payload: dict) -> None:
                nonlocal progress_count
                progress_count += 1
                paper_title = payload.get("paper_title")
                jobs[job_id]["paper_index"] = progress_count
                jobs[job_id]["current_paper"] = paper_title

                if payload.get("status") == "processed":
                    graph_snapshot = payload.get("graph")
                    if graph_snapshot is not None:
                        save_graph(graph_snapshot)
                        app.state.graph = graph_snapshot
                        graph_payload = build_graph_payload(graph_snapshot)
                    else:
                        graph_payload = None
                else:
                    graph_payload = None

                event_payload = {
                    "job_id": job_id,
                    "paper_title": paper_title,
                    "paper_index": progress_count,
                    "paper_total": len(files_data),
                    "status": payload.get("status"),
                    "reason": payload.get("reason"),
                }
                if graph_payload is not None:
                    event_payload["graph"] = graph_payload
                broadcast_event(app, "paper_processed", event_payload)

            with timed_block("build_graph_job"):
                from services.verification import verify_bipartite

                if existing_graph:
                    updated_graph = builder.build_graph(
                        files_data=filtered_files,
                        existing_graph=existing_graph,
                        on_paper_processed=on_paper_processed,
                    )
                    logger.info(
                        "Verifying %s new nodes and %s new edges...",
                        len(updated_graph.new_nodes),
                        len(updated_graph.new_edges),
                    )
                    verify_bipartite(updated_graph, updated_graph.new_nodes, updated_graph.new_edges)
                    updated_graph.clear_incremental_tracking()
                else:
                    updated_graph = builder.build_graph(
                        files_data=filtered_files,
                        on_paper_processed=on_paper_processed,
                    )
                    logger.info("Verifying entire graph...")
                    verify_bipartite(updated_graph)

            with timed_block("save_graph_job"):
                save_graph(updated_graph)
            app.state.graph = updated_graph
            app.state.agent = None
            app.state.agent_graph_identity = None

        jobs[job_id]["status"] = "done"
        jobs[job_id]["processing"] = False
        jobs[job_id]["finished_at"] = time.time()
        jobs[job_id]["paper_index"] = len(files_data)
        jobs[job_id]["current_paper"] = None
        broadcast_event(
            app,
            "job_done",
            {
                "job_id": job_id,
                "status": "done",
                "paper_total": len(files_data),
                "paper_index": len(files_data),
                "graph": build_graph_payload(updated_graph),
            },
        )
        log_memory(f"job_{job_id}_done")
    except Exception as exc:
        logger.error("Error in background PDF job %s: %s\n%s", job_id, str(exc), traceback.format_exc())
        jobs[job_id]["status"] = "error"
        jobs[job_id]["processing"] = False
        jobs[job_id]["error"] = str(exc)
        jobs[job_id]["finished_at"] = time.time()
        broadcast_event(
            app,
            "job_error",
            {
                "job_id": job_id,
                "status": "error",
                "error": str(exc),
                "paper_total": jobs[job_id].get("paper_total", 0),
                "paper_index": jobs[job_id].get("paper_index", 0),
            },
        )
        log_memory(f"job_{job_id}_error")


async def process_upload_queue(app) -> None:
    """Single-consumer FIFO worker for upload jobs."""
    while True:
        queued_item = await app.state.upload_queue.get()
        if queued_item is None:
            app.state.upload_queue.task_done()
            break
        job_id, files_data = queued_item
        try:
            await process_pdf_job(app, job_id, files_data)
        finally:
            app.state.upload_queue.task_done()


def start_queue_worker(app) -> None:
    if getattr(app.state, "queue_worker_task", None) is not None:
        return
    app.state.upload_queue = asyncio.Queue()
    app.state.event_subscribers = set()
    app.state.queue_worker_task = asyncio.create_task(process_upload_queue(app))


async def stop_queue_worker(app) -> None:
    worker_task = getattr(app.state, "queue_worker_task", None)
    if worker_task is None:
        return
    await app.state.upload_queue.put(None)
    await worker_task
    app.state.queue_worker_task = None


@router.post("/graph/upload")
async def upload_papers(
    request: Request,
    files: List[UploadFile] = File(...),
):
    """
    Upload PDF files, process them with OpenAI, and return graph data.
    Files are processed in memory and not stored on disk.
    
    Args:
        files: List of PDF files to upload and process
        
    Returns:
        Graph data in the same format as /graph/dummy endpoint
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Read all files into memory (don't save to disk)
    files_data = []
    filenames = []
    for file in files:
        # Skip hidden files and non-PDF files
        if not file.filename or file.filename.startswith(".") or not file.filename.endswith(".pdf"):
            continue
        contents = await file.read()
        file_hash = hashlib.sha256(contents).hexdigest()
        files_data.append((file.filename, contents, file_hash))
        filenames.append(file.filename)

    if not files_data:
        raise HTTPException(status_code=400, detail="No valid PDF files found")

    jobs = request.app.state.jobs
    prune_jobs(jobs)
    job_id = str(uuid4())
    queue_position = request.app.state.upload_queue.qsize() + 1
    jobs[job_id] = {
        "status": "pending",
        "queued": True,
        "processing": False,
        "queued_at": time.time(),
        "filename": ", ".join(filenames),
        "paper_total": len(files_data),
        "paper_index": 0,
        "current_paper": None,
        "queue_position": queue_position,
        "error": None,
    }

    request.app.state.upload_queue.put_nowait((job_id, files_data))
    broadcast_event(
        request.app,
        "job_queued",
        {
            "job_id": job_id,
            "status": "pending",
            "queue_position": queue_position,
            "paper_total": len(files_data),
            "filenames": filenames,
        },
    )
    return {"job_id": job_id, "status": "pending", "queue_position": queue_position}


@router.get("/graph/stream")
async def graph_stream(request: Request):
    subscriber_queue: asyncio.Queue = asyncio.Queue(maxsize=200)
    request.app.state.event_subscribers.add(subscriber_queue)

    async def event_generator():
        try:
            graph = request.app.state.graph
            if graph is not None:
                yield _serialize_sse_event(
                    "graph_snapshot",
                    {
                        "graph": build_graph_payload(graph),
                    },
                )
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event_name, payload = await asyncio.wait_for(
                        subscriber_queue.get(), timeout=SSE_HEARTBEAT_SECONDS
                    )
                    yield _serialize_sse_event(event_name, payload)
                except asyncio.TimeoutError:
                    yield _serialize_sse_event("heartbeat", {"ts": time.time()})
        finally:
            request.app.state.event_subscribers.discard(subscriber_queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/graph/load")
async def load_saved_graph(request: Request):
    """Load previously saved graph data"""
    try:
        async with request.app.state.graph_lock:
            graph = request.app.state.graph
            if graph is None:
                with timed_block("load_graph_endpoint_s3"):
                    logger.info("Loading graph from S3...")
                    graph = load_graph()
                request.app.state.graph = graph

        if graph is None:
            raise HTTPException(status_code=404, detail="No saved graph found")

        with timed_block("graph_to_dict_endpoint"):
            graph_data = build_graph_payload(graph)
        
        log_memory("graph_load_endpoint_return")
        return JSONResponse(content=graph_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading saved graph: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading graph: {str(e)}")
