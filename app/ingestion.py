import os
import json
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from app.config import settings



def load_documents(docs_path : str) -> List[Dict]:
    docs = []
    for fname in os.listdir(docs_path):
        if not fname.endswith('.txt'):
            continue
        path = os.path.join(docs_path, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append({'id' : fname, 'text' : text, 'source' : fname})
    return docs


def chunk_text(text : str, chunk_size : int = 500, overlap : int = 100) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap
    return chunks


def create_embedding_model(model_name : str) -> SentenceTransformer:
    model = SentenceTransformer(model_name, device="cpu")
    return model


def build_embeddings(docs : list[Dict], model : SentenceTransformer) -> Tuple[np.ndarray, List[Dict]]:
    embeddings = []
    metadata = []
    idx = 0
    for doc in docs:
        chunks = chunk_text(doc["text"])
        for chunk in chunks:
            vec = model.encode(chunk)
            embeddings.append(vec)
            metadata.append(
                {"idx" : idx,
                 "text" : chunk,
                 "source" : doc["source"],
                 }
            )
            idx += 1
    emb_matrix = np.vstack(embeddings).astype("float32")
    return emb_matrix, metadata


def build_faiss_index(emb_matrix : np.ndarray, index_path : str):
    dim = emb_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(emb_matrix)
    index.add(emb_matrix)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)


def save_metadata(metadata : List[Dict], path : str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)