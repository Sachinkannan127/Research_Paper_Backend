import logging
import os
import json
import shutil
from pymongo import UpdateOne
from app.db.mongodb import get_vector_collection
import certifi
logger = logging.getLogger("uvicorn")


class VectorStore:
    """
    Stores text chunks alongside their embeddings in MongoDB Atlas,
    and performs vector search using Atlas Vector Search ($vectorSearch).
    """

    INDEX_NAME = "vector_index"

    def _get_collection(self):
        return get_vector_collection()

    async def get_existing_ids(self) -> set[str]:
        """Returns all document chunk IDs already present in MongoDB Atlas."""
        try:
            collection = self._get_collection()
            cursor = collection.find({}, {"_id": 1, "id": 1})
            docs = await cursor.to_list(length=50000)
            existing = set()
            for doc in docs:
                if "_id" in doc:
                    existing.add(str(doc["_id"]))
                if "id" in doc:
                    existing.add(str(doc["id"]))
            return existing
        except Exception as e:
            logger.error(f"Failed to fetch existing document IDs from MongoDB: {e}")
            return set()

    async def ensure_vector_index(self):
        """Automatically checks and creates/updates the Atlas Vector Search index if missing or dimension mismatch."""
        try:
            collection = self._get_collection()
            cursor = collection.list_search_indexes()
            existing_indexes = await cursor.to_list(length=100)
            
            target_dimensions = 768
            existing_index = None
            for idx in existing_indexes:
                if idx.get("name") == self.INDEX_NAME:
                    existing_index = idx
                    break

            should_create = True
            if existing_index is not None:
                # Check dimensions
                definition = existing_index.get("latestDefinition") or existing_index.get("definition") or {}
                fields = definition.get("fields", [{}])
                current_dimensions = fields[0].get("numDimensions") if fields else None
                
                if current_dimensions == target_dimensions:
                    should_create = False
                else:
                    logger.info(f"Dimensions mismatch (found {current_dimensions}, expected {target_dimensions}). Dropping index...")
                    try:
                        await collection.drop_search_index(self.INDEX_NAME)
                        logger.info("Successfully dropped search index.")
                    except Exception as drop_err:
                        logger.warning(f"Failed to drop search index (it might be dropping already): {drop_err}")
                    # Allow some time or let Atlas handle the drop-create pipeline
                    import asyncio
                    await asyncio.sleep(2)

            if should_create:
                from pymongo.operations import SearchIndexModel
                index_model = SearchIndexModel(
                    definition={
                        "fields": [
                            {
                                "type": "vector",
                                "path": "embedding",
                                "numDimensions": target_dimensions,
                                "similarity": "cosine"
                            }
                        ]
                    },
                    name=self.INDEX_NAME,
                    type="vectorSearch"
                )
                await collection.create_search_index(model=index_model)
                logger.info(f"✅ Automatically created Atlas Vector Search index '{self.INDEX_NAME}' with {target_dimensions} dimensions on MongoDB Atlas.")
        except Exception as e:
            logger.warning(f"Atlas Search index check note: {e}")

    async def add(self, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]):
        """Stores or updates a batch of document chunks in MongoDB Atlas and local embeddings folder."""
        if not ids:
            return

        collection = self._get_collection()
        operations = []

        # Setup local embeddings folder in project root
        local_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "embeddings"))
        os.makedirs(local_dir, exist_ok=True)

        for chunk_id, embedding, doc, meta in zip(ids, embeddings, documents, metadatas):
            doc_body = {
                "_id": chunk_id,
                "id": chunk_id,
                "text": doc,
                "chunk": doc,
                "embedding": embedding,
                "source": meta.get("source", "unknown"),
                "metadata": meta,
            }
            operations.append(
                UpdateOne({"_id": chunk_id}, {"$set": doc_body}, upsert=True)
            )

            # Save locally in the local embedding folder
            local_file = os.path.join(local_dir, f"{chunk_id}.json")
            try:
                with open(local_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "id": chunk_id,
                        "chunk": doc,
                        "embedding": embedding,
                        "metadata": meta
                    }, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Failed to write local embedding file: {e}")

        if operations:
            result = await collection.bulk_write(operations)
            logger.info(f"MongoDB Vector Store updated: {result.upserted_count} inserted, {result.modified_count} modified.")
            await self.ensure_vector_index()

    async def add_documents(self, ids: list[str], documents: list[str], embeddings: list[list[float]], metadatas: list[dict]):
        """Wrapper for add() to support backward compatibility with the old interface."""
        await self.add(ids, embeddings, documents, metadatas)

    async def count(self) -> int:
        """Returns the number of document chunks in the database."""
        try:
            collection = self._get_collection()
            count_val = await collection.count_documents({})
            return count_val
        except Exception as e:
            logger.error(f"Failed to count documents in MongoDB: {e}")
            return 0

    async def delete_all(self):
        """Deletes all document chunks from the collection and clears local embeddings folder."""
        try:
            collection = self._get_collection()
            await collection.delete_many({})
            logger.info("🗑️ Cleared all document chunks from MongoDB Vector Store.")

            # Clear local embedding folder
            local_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "embeddings"))
            if os.path.exists(local_dir):
                shutil.rmtree(local_dir)
                os.makedirs(local_dir, exist_ok=True)
                logger.info("🗑️ Cleared local embeddings folder.")
        except Exception as e:
            logger.error(f"Failed to delete documents from MongoDB: {e}")

    async def recreate_collection(self, similarity_metric="cosine"):
        """Clears existing documents and ensures the vector search index exists."""
        await self.delete_all()
        await self.ensure_vector_index()

    async def get_all(self) -> dict:
        """Returns all documents, metadata and IDs from the database (for testing scripts)."""
        try:
            collection = self._get_collection()
            cursor = collection.find({})
            docs = await cursor.to_list(length=50000)
            return {
                "ids": [str(doc["_id"]) for doc in docs],
                "documents": [doc.get("text", "") for doc in docs],
                "chunks": [doc.get("chunk", "") for doc in docs],
                "embeddings": [doc.get("embedding", []) for doc in docs],
                "metadatas": [doc.get("metadata", {}) for doc in docs]
            }
        except Exception as e:
            logger.error(f"Failed to retrieve all documents from MongoDB: {e}")
            return {"ids": [], "documents": [], "chunks": [], "embeddings": [], "metadatas": []}

    async def query(self, query_embedding: list[float] = None, top_k: int = 4, query_embeddings: list[float] = None) -> list[dict]:
        """Finds the `top_k` chunks most similar to the query embedding using MongoDB Atlas Vector Search."""
        # Support both singular and plural parameter names for compatibility
        vector = query_embedding if query_embedding is not None else query_embeddings
        if vector is None:
            logger.warning("No query vector provided to VectorStore.query.")
            return []

        collection = self._get_collection()

        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.INDEX_NAME,
                    "path": "embedding",
                    "queryVector": vector,
                    "numCandidates": max(top_k * 10, 50),
                    "limit": top_k,
                }
            },
            {
                "$project": {
                    "text": 1,
                    "source": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            }
        ]

        try:
            cursor = collection.aggregate(pipeline)
            results = await cursor.to_list(length=top_k)
            matches = []
            for doc in results:
                matches.append({
                    "text": doc.get("text", ""),
                    "source": doc.get("source", "unknown"),
                    "page": doc.get("metadata", {}).get("page"),
                    "score": doc.get("score", 0.0),
                    "distance": 1.0 - doc.get("score", 0.0),
                })
            return matches
        except Exception as e:
            logger.warning(
                f"MongoDB Vector Search query failed or 'vector_index' is not configured yet in Atlas UI: {e}"
            )
            return []
