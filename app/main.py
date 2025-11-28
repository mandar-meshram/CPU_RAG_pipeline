import time
from fastapi import FastAPI
from app.config import settings
from app.schemas import QueryRequest, QueryResponse, Source
from app.retriever import Retriever
from app.llm_client import LLMClient
from app.pipeline import RAGPipeline

app = FastAPI()

@app.on_event("startup")
def startup_event():
    retriever = Retriever()
    llm_client = LLMClient()
    pipeline = RAGPipeline(retriever, llm_client, settings.TOP_K)
    app.state.pipeline = pipeline

@app.post("/query", response_model=QueryResponse)
def query_rag(payload: QueryRequest):
    t0 = time.time()
    answer, contexts = app.state.pipeline.answer_query(payload.query)
    latency_ms = (time.time() - t0) * 1000.0
    sources = [
        Source(text=c["text"], source=c["source"], score=c["score"]) for c in contexts
    ]
    return QueryResponse(answer=answer, sources=sources, latency_ms=latency_ms)
