from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from services.graph_builder import create_dummy_graph, GraphBuilder
from pathlib import Path
import os
import shutil
import traceback
import logging
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Storage directory for uploads (use absolute path from backend directory)
BACKEND_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BACKEND_DIR / "storage" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


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
                "topics": paper_data.topics
            })
            topics.update(paper_data.topics)
        elif data.get('type') == 'topic':
            topics.add(node)
    
    # Extract all edges
    for source, target, edge_data in graph.graph.edges(data=True):
        edges.append({
            "source": source,
            "target": target,
            "type": edge_data.get('type', 'topic'),
            "weight": edge_data.get('weight', 1.0)
        })
    
    return {
        "papers": papers,
        "topics": list(topics),
        "edges": edges
    }

@router.get("/graph/dummy")
def get_dummy_graph():
    """Get dummy graph data for frontend visualization"""
    graph = create_dummy_graph()
    return graph_to_dict(graph)


@router.post("/graph/upload")
async def upload_papers(files: List[UploadFile] = File(...)):
    """
    Upload PDF files, process them with OpenAI, and return graph data.
    
    Args:
        files: List of PDF files to upload and process
        
    Returns:
        Graph data in the same format as /graph/dummy endpoint
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Create a temporary directory for this upload session
    import uuid
    session_id = str(uuid.uuid4())
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save all uploaded files
        saved_files = []
        for file in files:
            if not file.filename or not file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
            
            # Handle nested directory structures in filename (from directory uploads)
            # Create the full path including any subdirectories
            file_path = session_dir / file.filename
            
            # Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Read file content asynchronously
            contents = await file.read()
            with open(file_path, "wb") as buffer:
                buffer.write(contents)
            saved_files.append(file_path)
            logger.info(f"Saved file: {file_path}")
        
        # Process the PDFs using GraphBuilder (which uses OpenAI)
        logger.info(f"Processing {len(saved_files)} PDF files...")
        builder = GraphBuilder()
        graph = builder.build_graph(str(session_dir))
        logger.info(f"Graph built successfully with {len(graph.graph.nodes())} nodes")
        
        # Convert graph to frontend format
        graph_data = graph_to_dict(graph)
        
        # Clean up uploaded files (optional - you might want to keep them)
        # shutil.rmtree(session_dir)
        
        return JSONResponse(content=graph_data)
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        if session_dir.exists():
            shutil.rmtree(session_dir)
        raise
    except Exception as e:
        # Log the full error for debugging
        error_trace = traceback.format_exc()
        logger.error(f"Error processing files: {str(e)}\n{error_trace}")
        
        # Clean up on error
        if session_dir.exists():
            shutil.rmtree(session_dir)
        
        # Return detailed error message
        error_msg = f"Error processing files: {str(e)}"
        if "OPENAI" in str(e).upper() or "assistant" in str(e).lower():
            error_msg += " (Check your OPENAI_API_KEY and OPENAI_ASSISTANT_ID environment variables)"
        raise HTTPException(status_code=500, detail=error_msg)
