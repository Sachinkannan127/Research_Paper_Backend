from app.rag.vector_store import VectorStore

db = VectorStore()

print("Total Chunks:", db.count())

data = db.get_all()

print("\nIDs")
print(data["ids"])

print("\nDocuments")
for doc in data["documents"]:
    print(doc)

print("\nMetadata")
for meta in data["metadatas"]:
    print(meta)