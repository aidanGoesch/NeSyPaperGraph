from pydantic import BaseModel
from typing import List, Optional

class Paper(BaseModel):
    title: str
    file_path: str
    text: str
    topics: List[str] = []
    authors: Optional[str] = None 
