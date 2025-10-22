from pydantic import BaseModel
from typing import List, Optional
import numpy as np

class Paper(BaseModel):
    title: str
    file_path: str
    text: str
    topics: List[str] = []
    embedding: Optional[List[float]] = None  # Changed from np.array to List[float] for Pydantic compatibility
    authors: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True
