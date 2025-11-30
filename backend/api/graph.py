from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from services.graph_builder import create_dummy_graph, GraphBuilder, get_all_topics_seen
from services.verification import verify_bipartite
import traceback
import logging
import pickle
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
            # Skip hidden files and non-PDF files
            if not file.filename or file.filename.startswith('.') or not file.filename.endswith('.pdf'):
                continue
            
            # Read file content asynchronously into memory
            contents = await file.read()
            files_data.append((file.filename, contents))
        
        if not files_data:
            raise HTTPException(status_code=400, detail="No valid PDF files found")
        
        logger.info(f"Processing {len(files_data)} PDF files")
        
        # Check if OpenAI API key is set
        if not os.getenv('OPENAI_API_KEY'):
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY environment variable not set")
        
        # Try to load existing graph, or create new one
        save_path = "storage/saved_graph.pkl"
        try:
            if os.path.exists(save_path):
                with open(save_path, 'rb') as f:
                    graph = pickle.load(f)
                logger.info("Loaded existing graph to update")
            else:
                graph = None
                logger.info("No existing graph found, creating new one")
        except Exception as e:
            logger.warning(f"Could not load existing graph: {e}, creating new one")
            graph = None
        
        # Process the PDFs using GraphBuilder
        builder = GraphBuilder()
        if graph:
            # Update existing graph with new papers
            updated_graph = builder.build_graph(files_data=files_data, existing_graph=graph)
            # Incremental verification - only check new nodes/edges
            logger.info(f"Verifying {len(updated_graph.new_nodes)} new nodes and {len(updated_graph.new_edges)} new edges...")
            verify_bipartite(updated_graph, updated_graph.new_nodes, updated_graph.new_edges)
            updated_graph.clear_incremental_tracking()
        else:
            # Create new graph
            updated_graph = builder.build_graph(files_data=files_data)
            # Full verification for new graph
            logger.info("Verifying entire graph...")
            verify_bipartite(updated_graph)
        
        graph = updated_graph
        
        # Convert graph to frontend format
        graph_data = graph_to_dict(graph)
        
        # Save the graph for future loading
        save_path = "storage/saved_graph.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(graph, f)
        logger.info(f"Graph saved to {save_path}")
        
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


@router.get("/graph/load")
def load_saved_graph():
    """Load previously saved graph data"""
    save_path = "storage/saved_graph.pkl"
    
    if not os.path.exists(save_path):
        raise HTTPException(status_code=404, detail="No saved graph found")
    
    try:
        logger.info("Loading graph from pickle file...")
        with open(save_path, 'rb') as f:
            graph = pickle.load(f)
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
                with open(save_path, 'wb') as f:
                    pickle.dump(graph, f)
                logger.info("Saved graph with computed synonyms and merged topics")
        
        # Convert to frontend format
        logger.info("Converting graph to frontend format...")
        graph_data = graph_to_dict(graph)
        
        # Add all topics seen
        all_topics = get_all_topics_seen()
        graph_data["all_topics_seen"] = list(all_topics)
        
        logger.info("Loaded saved graph successfully")
        return JSONResponse(content=graph_data)
        
    except Exception as e:
        logger.error(f"Error loading saved graph: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading graph: {str(e)}")
