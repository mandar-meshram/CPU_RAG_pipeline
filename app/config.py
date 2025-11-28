from pydantic import BaseModel

class Settings(BaseModel):

    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    DOCS_PATH: str = "data/raw"
    INDEX_PATH: str = "indexes/faiss_index.bin"
    METADATA_PATH: str = "indexes/metadata.json"
    LLM_API_URL: str = "http://127.0.0.1:8080/"
    TOP_K: int = 5

settings = Settings()