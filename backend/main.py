from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from api.graph import router as graph_router
from services.question_agent import QuestionAgent
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class SearchRequest(BaseModel):
    query: str

def get_current_graph_data():
    """Get the most recent graph data"""
    try:
        from api.graph import get_dummy_graph
        return get_dummy_graph()
    except Exception as e:
        print(f"Could not load current graph data: {e}")
        return {
            "papers": [
                {"title": "Paper A", "topics": ["Topic 1", "Topic 2", "Topic 3"]},
                {"title": "Paper B", "topics": ["Topic 1", "Topic 2", "Topic 3"]}
            ],
            "topics": ["Topic 1", "Topic 2", "Topic 3"]
        }

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
        current_graph = get_current_graph_data()
        agent = QuestionAgent(current_graph)
    mermaid_diagram = agent.get_mermaid_diagram()
    return {"mermaid": mermaid_diagram, "status": "success"}

@app.post("/api/search")
async def search(request: SearchRequest):
    global agent
    if not agent:
        current_graph = get_current_graph_data()
        agent = QuestionAgent(current_graph)
    
    print(f"Received search query: {request.query}")
    
    try:
        answer = await agent.answer_question(request.query)
        mermaid_diagram = agent.get_mermaid_diagram()
        return {
            "query": request.query, 
            "answer": answer, 
            "mermaid": mermaid_diagram,
            "status": "success"
        }
    except Exception as e:
        print(f"Error processing question: {e}")
        return {"query": request.query, "error": str(e), "status": "error"}
