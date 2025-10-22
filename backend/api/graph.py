from fastapi import APIRouter
from services.graph_builder import create_dummy_graph

router = APIRouter()

@router.get("/graph/dummy")
def get_dummy_graph():
    """Get dummy graph data for frontend visualization"""
    graph = create_dummy_graph()
    
    # Extract papers and topics from NetworkX graph
    papers = []
    topics = set()
    
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
    
    return {
        "papers": papers,
        "topics": list(topics)
    }
