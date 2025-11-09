from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.graph import router as graph_router
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

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
