import os 
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.config import settings
from app.ingestion import (
    load_documents,
    create_embedding_model,
    build_embeddings,
    build_faiss_index,
    save_metadata
)


def main():
    docs = load_documents(settings.DOCS_PATH)
    print(f"Loaded {len(docs)} documents.")

    model = create_embedding_model(settings.EMBEDDING_MODEL_NAME)
    emb_matrix, metadata = build_embeddings(docs, model)
    print(f"Built embeddings for {len(metadata)} chunks , dim = {emb_matrix.shape[1]}")

    build_faiss_index(emb_matrix, settings.INDEX_PATH)
    save_metadata(metadata, settings.METADATA_PATH)
    print("Index and metadata saved.")



if __name__ == "__main__":
    main()