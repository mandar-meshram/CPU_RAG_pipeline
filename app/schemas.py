from typing import List
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query : str

class Source(BaseModel):
    text : str
    source : str
    score : float

class QueryResponse(BaseModel):
    answer : str
    sources : List[Source]
    latency_ms : float