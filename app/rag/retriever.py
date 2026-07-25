from app.rag.embeddings import Embeddings
from app.rag.vector_store import VectorStore


class Retriever:

    def __init__(self):
        self.embeddings = Embeddings()
        self.vector_store = VectorStore()

    async def retrieve(self, question: str, top_k: int = 5):

        query_embedding = await self.embeddings.embed_query(question)

        results = await self.vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k
        )

        retrieved_chunks = []

        for doc in results:
            retrieved_chunks.append(
                {
                    "text": doc.get("text", ""),
                    "page": doc.get("page"),
                    "source": doc.get("source", "unknown"),
                    "score": doc.get("score", 0.0),
                }
            )

        return retrieved_chunks