# embed_and_upsert.py
"""
Embed chunks (JSONL) and upsert into a local Chroma collection.

Expected input file: out/chunks.jsonl
Each line is a JSON object with keys:
  - chunk_id (optional, but recommended)
  - source
  - page
  - chunk_index
  - text
  - token_count (optional)

Outputs:
  - Chroma DB persisted to CHROMA_DIR (default: out/chroma_db)
  - prints progress and a small semantic query example at the end
"""

import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

import numpy as np
from sentence_transformers import SentenceTransformer

import chromadb
from chromadb.config import Settings

# ---------------- CONFIG ----------------
CHUNKS_JSONL = Path(r"D:\RAG\out\chunks.jsonl")
CHROMA_DIR = Path(r"D:\RAG\out\chroma_db") # persistent chroma dir
COLLECTION_NAME = "pdf_chunks"                # collection name in chroma
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64                               # adjust for memory/CPU
# -----------------------------------------

def load_chunks(path: Path) -> List[Dict]:
    assert path.exists(), f"Chunks file not found: {path}"
    chunks = []
    with path.open("r", encoding="utf8") as f:
        for line in f:
            obj = json.loads(line)
            # ensure required fields exist
            if "text" not in obj:
                continue
            # ensure chunk_id exists (create if missing)
            if "chunk_id" not in obj:
                # deterministic-ish fallback id
                obj["chunk_id"] = f"{obj.get('source','doc')}-p{obj.get('page','?')}-c{obj.get('chunk_index',0)}"
            chunks.append(obj)
    return chunks

def embed_batch(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """
    Encode a batch of strings into embeddings (np.ndarray).
    """
    emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return emb

def main():
    print("Loading chunks from:", CHUNKS_JSONL)
    chunks = load_chunks(CHUNKS_JSONL)
    print("Loaded chunks:", len(chunks))

    if len(chunks) == 0:
        print("No chunks to embed. Exiting.")
        return
    
    # Load embedding model
    print("Loading embedding model:", EMBED_MODEL_NAME)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    # Init chroma client with persistence
    print("Initializing Chroma persistent client at:", CHROMA_DIR)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # create or get collection
    try:
        collection = client.get_collection(COLLECTION_NAME)
        print(f"Using existing collection: {COLLECTION_NAME}")
    except Exception:
        collection = client.create_collection(COLLECTION_NAME)
        print(f"Created collection: {COLLECTION_NAME}")

    # # Load embedding model
    # print("Loading embedding model:", EMBED_MODEL_NAME)
    # embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    # # Init chroma client (local) and collection
    # # client = chromadb.Client(Settings(chroma_db_impl="chroma", persist_directory=str(CHROMA_DIR)))
    # client = chromadb.Client()
    # # create or get collection
    # try:
    #     collection = client.get_collection(COLLECTION_NAME)
    #     print(f"Using existing collection: {COLLECTION_NAME}")
    #     cols = client.list_collections()
    #     print("Collections:", [c.name for c in cols])
    # except Exception:
    #     collection = client.create_collection(COLLECTION_NAME)
    #     print(f"Created collection: {COLLECTION_NAME}")

    # Upsert in batches
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        ids = [c["chunk_id"] for c in batch]
        texts = [c["text"] for c in batch]
        metadatas = [
            {
                "source": c.get("source"),
                "page": c.get("page"),
                "chunk_index": c.get("chunk_index"),
                "token_count": c.get("token_count")
            }
            for c in batch
        ]

        embeddings = embed_batch(embed_model, texts)  # np.ndarray shape (N, dim)

        # Chroma add expects lists (embeddings -> list of lists)
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings.tolist()
        )

        print(f"Upserted chunks {i} .. {i + len(batch) - 1}")

    # --- after upserting batches ---
    print("Upsert complete.")

    # NOTE: newer chromadb versions do not have client.persist().
    # If you need explicit persistence / a custom persistent location,
    # see the migration docs or use the migration/downgrade options below.
    print("Chroma client upsert finished. If you need disk persistence, follow the migration or install an older chromadb version.")


    # ---- Quick semantic query demonstration ----
    sample_query = "How to prepare for AI-102 exam?"
    print("\nRunning a quick semantic search demo for query:")
    print("  >", sample_query)
    q_emb = embed_model.encode([sample_query], convert_to_numpy=True)
    res = collection.query(query_embeddings=q_emb.tolist(), n_results=5, include=["metadatas", "documents", "distances"])
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    print("\nTop results:")
    for doc, meta, dist in zip(docs, metas, dists):
        print(f"- score: {dist:.4f} | source: {meta.get('source')} | page: {meta.get('page')}")
        print("  snippet:", doc[:180].replace("\n"," ") + "...")
    print("\nDone.")

if __name__ == "__main__":
    main()
