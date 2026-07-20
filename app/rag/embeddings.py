from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[2]))


from sentence_transformers import SentenceTransformer
import numpy as np

from app.rag.chunk import chunk_text


class Embeddings:
    """Generates embeddings for text using a sentence-transformer model."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings

    def embed_query(self, query: str) -> list[float]:
        embedding = self.model.encode(query, show_progress_bar=False)
        return embedding.tolist() if isinstance(embedding, np.ndarray) else embedding


if __name__ == "__main__":
    pdf_path = "app\\uploads\\Research_paper.pdf"
    chunks = chunk_text(pdf_path)
    embeddings = Embeddings().embed_texts(chunks)

    print(embeddings)
    print("Length:", len(chunks))