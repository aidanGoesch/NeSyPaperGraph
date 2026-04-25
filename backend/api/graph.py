from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from services.graph_builder import create_dummy_graph, GraphBuilder, get_all_topics_seen
from services.verification import verify_bipartite
from services.storage_service import load_graph, save_graph
import traceback
import logging
import os
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

def graph_to_dict(graph):
    """Convert PaperGraph to dictionary format for frontend"""
    papers = []
    topics = set()
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
            topics.update(paper_data.topics)
    
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

@router.get("/graph/dummy")
def get_dummy_graph():
    """Get dummy graph data for frontend visualization"""
    graph = create_dummy_graph()
    return graph_to_dict(graph)

async def process_pdf_job(app, job_id: str, files_data: list[tuple[str, bytes]]):
    """Run PDF processing pipeline in background and persist updated graph to S3."""
    jobs = app.state.jobs
    jobs[job_id]["status"] = "processing"
    jobs[job_id]["error"] = None
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY environment variable not set")

        async with app.state.graph_lock:
            existing_graph = app.state.graph
            if existing_graph is None:
                existing_graph = load_graph()

            builder = GraphBuilder()
            if existing_graph:
                updated_graph = builder.build_graph(
                    files_data=files_data,
                    existing_graph=existing_graph,
                )
                logger.info(
                    "Verifying %s new nodes and %s new edges...",
                    len(updated_graph.new_nodes),
                    len(updated_graph.new_edges),
                )
                verify_bipartite(updated_graph, updated_graph.new_nodes, updated_graph.new_edges)
                updated_graph.clear_incremental_tracking()
            else:
                updated_graph = builder.build_graph(files_data=files_data)
                logger.info("Verifying entire graph...")
                verify_bipartite(updated_graph)

            save_graph(updated_graph)
            app.state.graph = updated_graph
            app.state.agent = None
            app.state.agent_graph_identity = None

        jobs[job_id]["status"] = "done"
    except Exception as exc:
        logger.error("Error in background PDF job %s: %s\n%s", job_id, str(exc), traceback.format_exc())
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(exc)


@router.post("/graph/upload")
async def upload_papers(
    request: Request,
    background_tasks: BackgroundTasks,
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
        files_data.append((file.filename, contents))
        filenames.append(file.filename)

    if not files_data:
        raise HTTPException(status_code=400, detail="No valid PDF files found")

    jobs = request.app.state.jobs
    job_id = str(uuid4())
    jobs[job_id] = {
        "status": "pending",
        "filename": ", ".join(filenames),
        "error": None,
    }

    background_tasks.add_task(process_pdf_job, request.app, job_id, files_data)
    return {"job_id": job_id, "status": "pending"}


@router.get("/graph/load")
def load_saved_graph(request: Request):
    """Load previously saved graph data"""
    try:
        graph = request.app.state.graph
        if graph is None:
            logger.info("Loading graph from S3...")
            graph = load_graph()
            request.app.state.graph = graph

        if graph is None:
            raise HTTPException(status_code=404, detail="No saved graph found")

        logger.info("Graph loaded successfully")
        
        logger.info("Verifying bipartiteness...")
        verify_bipartite(graph)
        logger.info("Bipartite verification complete")
        
        # Load synonyms into cache if they exist
        from services.graph_builder import _topic_synonyms_cache
        has_synonyms = hasattr(graph, 'topic_synonyms') and graph.topic_synonyms and len(graph.topic_synonyms) > 0
        
        if has_synonyms:
            _topic_synonyms_cache.update(graph.topic_synonyms)
            logger.info(f"Loaded {len(graph.topic_synonyms)} cached synonyms from graph")
        
        # If no synonyms exist, compute them
        if not has_synonyms:
            from services.llm_service import OpenAILLMClient
            from services.verification import find_optimal_topic_merge
            
            all_topics = [node for node, data in graph.graph.nodes(data=True) if data.get('type') == 'topic']
            logger.info(f"Found {len(all_topics)} topics without synonyms")
            
            if all_topics:
                logger.info(f"Computing synonyms for {len(all_topics)} topics in batches...")
                client = OpenAILLMClient()
                
                # Process in batches of 50 topics at a time
                batch_size = 50
                topic_synonyms = {}
                for i in range(0, len(all_topics), batch_size):
                    batch = all_topics[i:i+batch_size]
                    logger.info(f"Processing batch {i//batch_size + 1}/{(len(all_topics)-1)//batch_size + 1} ({len(batch)} topics)")
                    batch_synonyms = client.generate_topic_synonyms(batch)
                    topic_synonyms.update(batch_synonyms)
                
                graph.topic_synonyms = topic_synonyms
                _topic_synonyms_cache.update(topic_synonyms)
                logger.info(f"Computed synonyms for {len(topic_synonyms)} topics")
                
                # Find optimal topic merges
                logger.info("Finding optimal topic merges...")
                from services.verification import find_optimal_topic_merge
                merge_groups = find_optimal_topic_merge(all_topics, topic_synonyms)
                graph.topic_merge_groups = merge_groups
                logger.info(f"Found {len(merge_groups)} merge groups")
                
                # Apply the merges to the graph
                logger.info("Applying topic merges to graph...")
                graph.merge_topics(merge_groups)
                logger.info("Topic merges applied")
                
                # Save updated graph with synonyms and merged topics
                save_graph(graph)
                logger.info("Saved graph with computed synonyms and merged topics")
        
        # Convert to frontend format
        logger.info("Converting graph to frontend format...")
        graph_data = graph_to_dict(graph)
        
        # Add all topics seen
        all_topics = get_all_topics_seen()
        graph_data["all_topics_seen"] = list(all_topics)
        
        logger.info("Loaded saved graph successfully")
        return JSONResponse(content=graph_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading saved graph: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading graph: {str(e)}")
