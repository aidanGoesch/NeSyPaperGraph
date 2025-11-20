from pydantic import BaseModel
from typing import List, Optional

class Paper(BaseModel):
    title: str
    file_path: str
    text: str
    summary: Optional[str] = None
    topics: List[str] = []
    embedding: Optional[List[float]] = None
    authors: Optional[List[str]] = None
    publication_date: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True
