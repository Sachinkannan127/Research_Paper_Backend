from app.rag.embeddings import Embeddings
from app.rag.vector_store import VectorStore


class Retriever:

    def __init__(self):
        self.embeddings = Embeddings()
        self.vector_store = VectorStore()

    def retrieve(self, question: str, top_k: int = 5):

        query_embedding = self.embeddings.embed_query(question)

        results = self.vector_store.query(
            query_embeddings=query_embedding,
            top_k=top_k
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        retrieved_chunks = []

        for doc, metadata, score in zip(documents, metadatas, distances):

            retrieved_chunks.append(
                {
                    "text": doc,
                    "page": metadata.get("page"),
                    "source": metadata.get("source"),
                    "score": score,
                }
            )

        return retrieved_chunks