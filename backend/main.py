import signal
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from api.graph import router as graph_router
from services.question_agent import QuestionAgent
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

def get_current_graph_data():
    """Get the most recent graph data"""
    try:
        # Try to load saved graph first
        import pickle
        save_path = "storage/saved_graph.pkl"
        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                from api.graph import graph_to_dict
                graph = pickle.load(f)
                return graph_to_dict(graph), graph  # Return both dict and object
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

app = FastAPI()
agent = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(graph_router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI"}

@app.get("/api/data")
def get_data():
    return {"data": "Sample data from backend"}

@app.get("/api/agent/architecture")
def get_agent_architecture():
    global agent
    if not agent:
        current_graph_data, current_graph_obj = get_current_graph_data()
        agent = QuestionAgent(current_graph_data, current_graph_obj)
    mermaid_diagram = agent.get_mermaid_diagram()
    return {"mermaid": mermaid_diagram, "status": "success"}

@app.post("/api/search")
async def search(request: SearchRequest):
    global agent
    if not agent:
        current_graph_data, current_graph_obj = get_current_graph_data()
        agent = QuestionAgent(current_graph_data, current_graph_obj)
    
    print(f"Received search query: {request.query}")
    
    try:
        answer = await agent.answer_question(request.query)
        
        # Check if this is a search results response
        if answer == "SEARCH_RESULTS" and hasattr(agent, '_last_state') and agent._last_state.get('search_results'):
            return {
                "query": request.query,
                "search_results": agent._last_state['search_results'],
                "mermaid": agent.get_mermaid_diagram(),
                "status": "search_results"
            }
        
        return {
            "query": request.query, 
            "answer": answer, 
            "mermaid": agent.get_mermaid_diagram(),
            "status": "success"
        }
    except Exception as e:
        print(f"Error processing question: {e}")
        return {"query": request.query, "error": str(e), "status": "error"}
