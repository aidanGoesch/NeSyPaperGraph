from pydantic import BaseModel
from typing import List, Optional

class Paper(BaseModel):
    title: str
    file_path: str
    topics: List[str] = []
    authors: Optional[str] = None
    abstract: Optional[str] = None
