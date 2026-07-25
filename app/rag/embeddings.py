import asyncio
import logging
import re
import litellm
from app.rag.chunk import chunk_text

logger = logging.getLogger("uvicorn")

class EmbeddingModel:
    """
    Turns text into vectors (lists of numbers) so we can compare how
    similar two pieces of text are by comparing their vectors.

    Uses Google's Gemini embedding API (hosted, no local model to download).
    """

    def __init__(self, model_name: str = "gemini/gemini-embedding-001", dimensions: int = 768):
        self.model_name = model_name
        self.dimensions = dimensions

    async def _execute_with_retry(self, func, *args, **kwargs):
        max_attempts = 5
        base_delay = 2.0
        for attempt in range(1, max_attempts + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                err_msg = str(e).lower()
                error_name = e.__class__.__name__.lower()
                
                is_rate_limit = (
                    "rate" in err_msg 
                    or "limit" in err_msg 
                    or "429" in err_msg 
                    or "quota" in err_msg 
                    or "exhausted" in err_msg 
                    or "ratelimit" in error_name
                )
                
                if is_rate_limit and attempt < max_attempts:
                    # Attempt to parse specific retry delay from the error message
                    sleep_time = base_delay
                    match = re.search(r"retry in (\d+\.?\d*)s", err_msg)
                    if match:
                        try:
                            sleep_time = float(match.group(1)) + 0.5
                        except ValueError:
                            pass
                    
                    logger.warning(
                        f"[Embeddings] Rate limit hit. Retrying attempt {attempt}/{max_attempts} "
                        f"in {sleep_time:.2f}s... Error: {e}"
                    )
                    await asyncio.sleep(sleep_time)
                    base_delay *= 2.0
                else:
                    logger.error(f"[Embeddings] Call failed after {attempt} attempts: {e}")
                    raise e

    async def embed_texts(self, texts: list[str], batch_size: int = 25) -> list[list[float]]:
        """Embeds many chunks at once — used during ingestion."""
        embeddings = []

        for i, start in enumerate(range(0, len(texts), batch_size)):
            if i > 0:
                # Slight pacing sleep between batches (e.g. 0.8 seconds) to prevent 429 errors
                await asyncio.sleep(0.8)
                
            batch = texts[start:start + batch_size]
            
            async def _embed_batch():
                return await litellm.aembedding(
                    model=self.model_name,
                    input=batch,
                    dimensions=self.dimensions,
                    task_type="RETRIEVAL_DOCUMENT",
                )
                
            response = await self._execute_with_retry(_embed_batch)
            embeddings.extend(item["embedding"] for item in response.data)

        return embeddings

    async def embed_query(self, text: str) -> list[float]:
        """Embeds a single piece of text — used for a user's question."""
        async def _embed_single():
            return await litellm.aembedding(
                model=self.model_name,
                input=[text],
                dimensions=self.dimensions,
                task_type="RETRIEVAL_QUERY",
            )
            
        response = await self._execute_with_retry(_embed_single)
        return response.data[0]["embedding"]


# Expose alias for backwards compatibility
Embeddings = EmbeddingModel


async def main():
    pdf_path = "app\\uploads\\Research_paper.pdf"
    chunks = chunk_text(pdf_path)
    model = EmbeddingModel()
    embeddings = await model.embed_texts(chunks)

    print(embeddings[:2])  # print first two only to avoid spamming
    print("Length of chunks:", len(chunks))
    print("Embedding size:", len(embeddings[0]) if embeddings else 0)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(main())