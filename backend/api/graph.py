from fastapi import APIRouter
from pydantic import BaseModel
from services.graph_builder import create_dummy_graph, GraphBuilder

router = APIRouter()

class FileList(BaseModel):
    files: list[str]

from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel
from services.graph_builder import create_dummy_graph, GraphBuilder
from typing import List
import tempfile
import os

router = APIRouter()

class FileList(BaseModel):
    files: list[str]

@router.post("/graph/build")
async def build_graph_from_files(files: List[UploadFile] = File(...)):
    """Build graph from uploaded PDF files"""
    try:
        print(f"Received {len(files)} files")
        
        from models.paper import Paper
        from services.pdf_preprocessor import extract_text_from_pdf
        from services.llm_service import TopicExtractor, HuggingFaceLLMClient
        
        # Initialize LLM for topic extraction
        client = HuggingFaceLLMClient()
        extractor = TopicExtractor(client)
        
        papers = []
        
        # Process each uploaded PDF file
        for uploaded_file in files[:5]:  # Limit to 5 files
            if not uploaded_file.filename.endswith('.pdf'):
                continue
                
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    content = await uploaded_file.read()
                    temp_file.write(content)
                    temp_path = temp_file.name
                
                # Extract text from PDF
                text = extract_text_from_pdf(temp_path)
                file_name = uploaded_file.filename.replace('.pdf', '')
                
                # Create paper with extracted text
                paper = Paper(
                    title=file_name,
                    file_path=uploaded_file.filename,
                    text=text[:2000]  # Limit text for LLM processing
                )
                
                # Extract topics using LLM
                topics = extractor.extract_topics(paper.text)
                paper.topics = topics if topics else ["General"]
                
                papers.append(paper)
                print(f"Processed {file_name}: {len(topics)} topics extracted")
                
                # Clean up temp file
                os.unlink(temp_path)
                
            except Exception as e:
                print(f"Error processing {uploaded_file.filename}: {e}")
                continue
        
        # Build graph from processed papers
        from models.graph import PaperGraph
        graph_obj = PaperGraph()
        for paper in papers:
            graph_obj.add_paper(paper)
        
        graph_obj.add_semantic_edges()
        
        # Format response
        papers_data = []
        topics = set()
        edges = []
        
        for node, data in graph_obj.graph.nodes(data=True):
            if data.get('type') == 'paper':
                paper_data = data['data']
                papers_data.append({
                    "title": paper_data.title,
                    "topics": paper_data.topics
                })
                topics.update(paper_data.topics)
            elif data.get('type') == 'topic':
                topics.add(node)
        
        for source, target, edge_data in graph_obj.graph.edges(data=True):
            edges.append({
                "source": source,
                "target": target,
                "type": edge_data.get('type', 'topic'),
                "weight": edge_data.get('weight', 1.0)
            })
        
        print(f"Returning {len(papers_data)} papers, {len(topics)} topics, {len(edges)} edges")
        
        return {
            "papers": papers_data,
            "topics": list(topics),
            "edges": edges
        }
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

@router.get("/graph/dummy")
def get_dummy_graph():
    """Get dummy graph data for frontend visualization"""
    graph = create_dummy_graph()
    
    # Extract papers and topics from NetworkX graph
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
