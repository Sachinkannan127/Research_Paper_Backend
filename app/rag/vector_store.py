import chromadb

from app.rag.chunk import chunk_text
from app.rag.embeddings import Embeddings

class VectorStore:

    COLLECTION_NAME = "research_papers"

    def __init__(self, persist_path="./chromadb"):

        self.client = chromadb.PersistentClient(
            path=persist_path
        )

        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME
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


