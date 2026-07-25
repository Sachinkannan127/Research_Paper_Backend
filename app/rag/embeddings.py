import litellm


class EmbeddingModel:
    """
    Turns text into vectors (lists of numbers) so we can compare how
    similar two pieces of text are by comparing their vectors.

    Uses Google's Gemini embedding API (hosted, no local model to download).
    """

    def __init__(self, model_name: str = "gemini/gemini-embedding-001", dimensions: int = 768):
        self.model_name = model_name
        self.dimensions = dimensions

    async def embed_texts(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """Embeds many chunks at once — used during ingestion."""
        embeddings = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            response = await litellm.aembedding(
                model=self.model_name,
                input=batch,
                dimensions=self.dimensions,
                task_type="RETRIEVAL_DOCUMENT",
            )
            embeddings.extend(item["embedding"] for item in response.data)

        return embeddings

    async def embed_query(self, text: str) -> list[float]:
        """Embeds a single piece of text — used for a user's question."""
        response = await litellm.aembedding(
            model=self.model_name,
            input=[text],
            dimensions=self.dimensions,
            task_type="RETRIEVAL_QUERY",
        )
        return response.data[0]["embedding"]


# Expose alias for backwards compatibility
Embeddings = EmbeddingModel


if __name__ == "__main__":
    import asyncio
    from app.rag.chunk import chunk_text
    pdf_path = "app\\uploads\\Research_paper.pdf"
    chunks = chunk_text(pdf_path)

    async def main():
        embeddings = await EmbeddingModel().embed_texts(chunks)
        print(embeddings)
        print("Length:", len(chunks))

    asyncio.run(main())