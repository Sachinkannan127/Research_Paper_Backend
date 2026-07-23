import os
import chromadb

from app.rag.chunk import chunk_text
from app.rag.embeddings import Embeddings

# Global shared client instance to reuse and avoid Python 3.13 PyO3 persistent client panics
_client = None

class VectorStore:

    COLLECTION_NAME = "research_papers"

    def __init__(self, persist_path="./chromadb"):
        global _client
        if _client is None:
            # We use EphemeralClient (in-memory mode) to bypass the PyO3 Rust bindings crash on Python 3.13.
            # Using an EphemeralClient makes database querying and indexing extremely fast and panic-free.
            _client = chromadb.EphemeralClient()
            print("[VectorStore] Initialized global in-memory EphemeralClient to prevent Python 3.13 PyO3 panics.")
        self.client = _client

        # Load active similarity metric from RAG config
        from app.core.config import settings
        config = settings.load_rag_config()
        self.similarity_metric = config.get("similarity_metric", "cosine")
        
        # In ChromaDB, the distance metric is set via metadata {"hnsw:space": "cosine" | "l2" | "ip"}
        hnsw_space = self.similarity_metric
        if hnsw_space not in ["cosine", "l2", "ip"]:
            hnsw_space = "cosine"

        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": hnsw_space}
        )

    def recreate_collection(self, similarity_metric="cosine"):
        try:
            self.client.delete_collection(name=self.COLLECTION_NAME)
        except Exception:
            pass
        
        hnsw_space = similarity_metric
        if hnsw_space not in ["cosine", "l2", "ip"]:
            hnsw_space = "cosine"
            
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": hnsw_space}
        )

    def add_documents(
        self,
        ids,
        documents,
        embeddings,
        metadatas,
    ):

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def count(self):

        return self.collection.count()

    def get_all(self):

        return self.collection.get()

    def query(self, query_embeddings: list[float], top_k: int = 5):

        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
        )

    def delete_all(self):

        ids = self.collection.get()["ids"]

        if ids:
            self.collection.delete(ids=ids)


