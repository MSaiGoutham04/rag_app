# debug_retrieval_safe.py
import chromadb
from pathlib import Path

COLLECTION_NAME = "pdf_chunks"
TOP_K = 6

client = chromadb.Client()
try:
    col = client.get_collection(COLLECTION_NAME)
except Exception:
    print(f"Collection {COLLECTION_NAME} not found. Create it or run embed_and_upsert.py to populate it.")
    col = client.create_collection(COLLECTION_NAME)
    print("Created empty collection. Please run embeddings to populate it.")
    raise SystemExit(1)

query = "Tell me about Function Groups"
res = col.query(query_texts=[query], n_results=TOP_K, include=["documents","metadatas","distances"])

docs = res["documents"][0]
metas = res["metadatas"][0]
dists = res["distances"][0]

for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
    print(f"\n[{i}] distance={dist:.4f} source={meta.get('source')} page={meta.get('page')}")
    print(doc[:800].replace("\n"," ") + ("..." if len(doc)>800 else ""))
