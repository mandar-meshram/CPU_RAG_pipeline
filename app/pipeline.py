from typing import List, Dict, Tuple
from app.config import settings
from app.retriever import Retriever
from app.llm_client import LLMClient

def build_prompt(query: str, contexts: List[Dict]) -> str:
    context_texts = "\n\n".join(
        f"Source: {c['source']}\n{c['text']}" for c in contexts
    )
    prompt = (
        f"Context:\n{context_texts}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    return prompt

class RAGPipeline:
    def __init__(self, retriever: Retriever, llm_client: LLMClient, top_k: int):
        self.retriever = retriever
        self.llm_client = llm_client
        self.top_k = top_k

    def answer_query(self, query: str) -> Tuple[str, List[Dict]]:
        contexts = self.retriever.retriever(query, self.top_k)
        prompt = build_prompt(query, contexts)
        answer = self.llm_client.generate(prompt)
        return answer, contexts
