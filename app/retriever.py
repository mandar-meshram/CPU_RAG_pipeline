import json
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from app.config import settings


class Retriever:

    def __init__(self):
        self.index = faiss.read_index(settings.INDEX_PATH)
        with open(settings.METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME, device="cpu")

    
    def retriever(self, query : str, top_k : int) -> List[Dict]:
        q_vec = self.model.encode(query).astype("float32")
        q_vec = q_vec.reshape(1, -1)
        faiss.normalize_L2(q_vec)
        scores, indices = self.index.search(q_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            meta = self.metadata[int(idx)]
            if score >= 0.5:
                results.append({
                    "text": meta["text"],
                        "source": meta["source"],
                        "score": float(score),
                })
        return results