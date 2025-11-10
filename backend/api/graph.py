from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from services.graph_builder import create_dummy_graph, GraphBuilder, get_all_topics_seen
import traceback
import logging
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
    Files are processed in memory and not stored on disk.
    
    Args:
        files: List of PDF files to upload and process
        
    Returns:
        Graph data in the same format as /graph/dummy endpoint
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    try:
        # Read all files into memory (don't save to disk)
        files_data = []
        for file in files:
            if not file.filename or not file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
            
            # Read file content asynchronously into memory
            contents = await file.read()
            files_data.append((file.filename, contents))
        
        # Process the PDFs using GraphBuilder (which uses OpenAI)
        builder = GraphBuilder()
        graph = builder.build_graph(files_data=files_data)
        
        # Convert graph to frontend format
        graph_data = graph_to_dict(graph)
        
        # Add all topics seen across all uploads to the response
        all_topics = get_all_topics_seen()
        graph_data["all_topics_seen"] = list(all_topics)
        
        return JSONResponse(content=graph_data)
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log the full error for debugging
        error_trace = traceback.format_exc()
        logger.error(f"Error processing files: {str(e)}\n{error_trace}")
        
        # Return detailed error message
        error_msg = f"Error processing files: {str(e)}"
        if "OPENAI" in str(e).upper() or "assistant" in str(e).lower():
            error_msg += " (Check your OPENAI_API_KEY and OPENAI_ASSISTANT_ID environment variables)"
        raise HTTPException(status_code=500, detail=error_msg)
