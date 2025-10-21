from pydantic import BaseModel
from typing import List

class Topic(BaseModel):
    name: str
    papers: List[str] = []  # paper titles
    frequency: int = 0
