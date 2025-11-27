# # combined_retriever.py
# """
# High-accuracy retriever: Embedding cosine (primary) + TFIDF cosine (fallback/boost).
# - Loads out/chunks.jsonl (expects chunk_id, text, source, page)
# - Loads precomputed embeddings if available at out/chroma_backup/embeddings.npz
#   otherwise computes embeddings with sentence-transformers and saves them.
# - Given a query, returns top-K passages ranked by combined score:
#     final_score = alpha * embed_score + (1 - alpha) * tfidf_score
# - Configurable thresholds and sizes.

# Usage:
#     from retrieval.combined_retriever import CombinedRetriever
#     r = CombinedRetriever()
#     results = r.retrieve("Tell me about Function Groups", final_k=5)
#     for res in results: print(res["final_score"], res["meta"], res["text"][:200])
# """

# from pathlib import Path
# import json
# from typing import List, Dict, Optional
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # ---------- CONFIG (tweakable) ----------
# CHUNKS_JSONL = Path("out/chunks.jsonl")
# BACKUP_EMB = Path("out/chroma_backup/embeddings.npz")  # optional precomputed embeddings
# EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# TFIDF_MAX_FEATURES = 20000
# DENSE_CANDIDATES = 50   # how many candidates to consider from dense search (before merging)
# TFIDF_TOPN = 50         # how many TFIDF candidates to compute
# FINAL_TOPK = 5
# ALPHA = 0.7             # weight for embedding score (0..1). TFIDF weight = 1-ALPHA
# EMBED_THRESHOLD = 0.42  # if top embed score < threshold, we'll rely more on TFIDF (still merged)
# # ----------------------------------------

# def load_chunks(path: Path) -> List[Dict]:
#     if not path.exists():
#         raise FileNotFoundError(f"Chunks file missing: {path}")
#     chunks = []
#     with path.open(encoding="utf8") as fh:
#         for line in fh:
#             line = line.strip()
#             if not line:
#                 continue
#             obj = json.loads(line)
#             if "chunk_id" not in obj:
#                 obj["chunk_id"] = f"{obj.get('source','doc')}-p{obj.get('page','?')}-c{obj.get('chunk_index',0)}"
#             chunks.append(obj)
#     return chunks

# class CombinedRetriever:
#     def __init__(
#         self,
#         chunks_path: Path = CHUNKS_JSONL,
#         backup_emb: Path = BACKUP_EMB,
#         embed_model_name: str = EMBED_MODEL,
#         tfidf_max_features: int = TFIDF_MAX_FEATURES
#     ):
#         # load chunks
#         self.chunks = load_chunks(chunks_path)
#         self.ids = [c["chunk_id"] for c in self.chunks]
#         self.texts = [c.get("text","") or "" for c in self.chunks]
#         self.metas = [ {"source": c.get("source"), "page": c.get("page")} for c in self.chunks]

#         # TF-IDF vectorizer (fit on all texts)
#         self.vectorizer = TfidfVectorizer(stop_words="english", max_features=tfidf_max_features)
#         self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)  # (N, F)

#         # Embedding model + embeddings (load or compute)
#         self.model = SentenceTransformer(embed_model_name)
#         if backup_emb.exists():
#             try:
#                 arr = np.load(backup_emb)
#                 self.embeddings = arr["embeddings"]
#                 if self.embeddings.shape[0] != len(self.texts):
#                     # mismatch -> recompute
#                     print("Embedding count mismatch with chunks; recomputing embeddings.")
#                     self.embeddings = self.model.encode(self.texts, convert_to_numpy=True, show_progress_bar=True)
#                     np.savez_compressed(backup_emb, embeddings=self.embeddings)
#                 else:
#                     print(f"Loaded precomputed embeddings: {backup_emb} ({self.embeddings.shape})")
#             except Exception as e:
#                 print("Failed loading backup embeddings:", e)
#                 print("Computing embeddings from scratch.")
#                 self.embeddings = self.model.encode(self.texts, convert_to_numpy=True, show_progress_bar=True)
#                 np.savez_compressed(backup_emb, embeddings=self.embeddings)
#         else:
#             # compute & save
#             print("No backup embeddings found — computing embeddings (may take a while)...")
#             self.embeddings = self.model.encode(self.texts, convert_to_numpy=True, show_progress_bar=True)
#             try:
#                 backup_emb.parent.mkdir(parents=True, exist_ok=True)
#                 np.savez_compressed(backup_emb, embeddings=self.embeddings)
#                 print("Saved embeddings to:", backup_emb)
#             except Exception as e:
#                 print("Warning: failed to save embeddings:", e)

#     @staticmethod
#     def _normalize_scores(scores: np.ndarray) -> np.ndarray:
#         """
#         Normalize array to 0..1 (min-max). If all same, return ones.
#         """
#         if scores.size == 0:
#             return scores
#         mn = float(np.min(scores))
#         mx = float(np.max(scores))
#         if mx - mn < 1e-12:
#             return np.ones_like(scores)
#         return (scores - mn) / (mx - mn)

#     def _embed_scores(self, query: str, topn: int = DENSE_CANDIDATES):
#         q_emb = self.model.encode([query], convert_to_numpy=True)
#         sims = cosine_similarity(q_emb, self.embeddings).flatten()  # [-1,1] or [0,1]
#         # best indices
#         top_idx = np.argsort(-sims)[:topn]
#         top_sims = sims[top_idx]
#         return top_idx, top_sims

#     def _tfidf_scores(self, query: str, topn: int = TFIDF_TOPN):
#         q_vec = self.vectorizer.transform([query])
#         sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()  # [0,1]
#         top_idx = np.argsort(-sims)[:topn]
#         top_sims = sims[top_idx]
#         return top_idx, top_sims

#     def retrieve(
#         self,
#         query: str,
#         dense_topn: int = DENSE_CANDIDATES,
#         tfidf_topn: int = TFIDF_TOPN,
#         final_k: int = FINAL_TOPK,
#         alpha: float = ALPHA,
#         embed_threshold: float = EMBED_THRESHOLD
#     ) -> List[Dict]:
#         """
#         Returns list of dicts:
#           { chunk_id, text, meta, embed_score, tfidf_score, final_score }
#         Scores are normalized to 0..1 before combining.
#         """
#         # 1) get dense candidates
#         idx_d, sims_d = self._embed_scores(query, topn=dense_topn)
#         # normalize embedding sims to 0..1
#         sims_d_norm = self._normalize_scores(sims_d)

#         dense_map = { self.ids[i]: {"idx": i, "embed_score": float(s)} for i,s in zip(idx_d, sims_d_norm) }

#         # 2) get tfidf candidates
#         idx_t, sims_t = self._tfidf_scores(query, topn=tfidf_topn)
#         sims_t_norm = self._normalize_scores(sims_t)
#         tfidf_map = { self.ids[i]: {"idx": i, "tfidf_score": float(s)} for i,s in zip(idx_t, sims_t_norm) }

#         # 3) merge candidate IDs
#         cand_ids = list(dict.fromkeys([self.ids[i] for i in idx_d] + [self.ids[i] for i in idx_t]))  # keep order
#         candidates = []
#         for cid in cand_ids:
#             i = self.ids.index(cid)
#             embed_score = dense_map.get(cid, {}).get("embed_score", 0.0)
#             tfidf_score = tfidf_map.get(cid, {}).get("tfidf_score", 0.0)
#             # If embed top score is very low (below threshold), downweight it by moving alpha toward 0.5
#             # (simple heuristic)
#             # compute final score
#             final_score = alpha * embed_score + (1.0 - alpha) * tfidf_score
#             candidates.append({
#                 "chunk_id": cid,
#                 "text": self.texts[i],
#                 "meta": self.metas[i],
#                 "embed_score": float(embed_score),
#                 "tfidf_score": float(tfidf_score),
#                 "final_score": float(final_score)
#             })

#         # 4) sort by final_score desc and return top-k
#         candidates.sort(key=lambda x: x["final_score"], reverse=True)
#         return candidates[:final_k]


# # ---------- quick CLI test ----------
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("query", nargs="?", help="Query (wrap in quotes)", default=None)
#     parser.add_argument("--k", type=int, default=5)
#     args = parser.parse_args()
#     if args.query is None:
#         q = input("Query: ").strip()
#     else:
#         q = args.query
#     r = CombinedRetriever()
#     res = r.retrieve(q, final_k=args.k)
#     for i, r in enumerate(res, start=1):
#         print(f"\n[{i}] final={r['final_score']:.4f} embed={r['embed_score']:.4f} tfidf={r['tfidf_score']:.4f} source={r['meta'].get('source')} page={r['meta'].get('page')}")
#         print(r['text'][:400].replace("\n"," ") + "...")


import os
from datetime import datetime



# combined_retriever.py
"""
High-accuracy retriever: Embedding cosine (primary) + TFIDF cosine (fallback/boost).
Now with caching for embeddings and TF-IDF vectorizer.

Usage:
    from retrieval.combined_retriever import CombinedRetriever
    r = CombinedRetriever()                      # uses cache if present
    # or force refresh:
    r = CombinedRetriever(force_refresh=True)

CLI:
    python combined_retriever.py "Tell me about Function Groups" --k 5
    python combined_retriever.py "query" --k 5 --refresh-cache
"""

from pathlib import Path
import json
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import tempfile
import os
import sys

# ---------- CONFIG (tweakable) ----------
CHUNKS_JSONL = Path("out/chunks.jsonl")
BACKUP_DIR = Path("out/chroma_backup")
BACKUP_EMB = BACKUP_DIR / "embeddings.npz"       # optional precomputed embeddings
TFIDF_PICKLE = BACKUP_DIR / "tfidf_vectorizer.pkl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TFIDF_MAX_FEATURES = 20000
DENSE_CANDIDATES = 50   # how many candidates to consider from dense search (before merging)
TFIDF_TOPN = 50         # how many TFIDF candidates to compute
FINAL_TOPK = 5
ALPHA = 0.7             # weight for embedding score (0..1). TFIDF weight = 1-ALPHA
EMBED_THRESHOLD = 0.42  # if top embed score < threshold, we'll rely more on TFIDF (still merged)
# ----------------------------------------

# -------------------- small atomic IO helpers --------------------
def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(path.parent)) as tmp:
        tmp_name = tmp.name
        tmp.write(data)
    os.replace(tmp_name, str(path))

def _atomic_write_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(path.parent)) as tmp:
        tmp_name = tmp.name
    try:
        with open(tmp_name, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_name, str(path))
    finally:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except Exception:
                pass

def save_embeddings_npz(path: Path, embeddings: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # write to temp file and replace atomically
    with tempfile.NamedTemporaryFile(delete=False, dir=str(path.parent)) as tmp:
        tmp_name = tmp.name
    try:
        np.savez_compressed(tmp_name, embeddings=embeddings)
        os.replace(tmp_name, str(path))
    finally:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except Exception:
                pass

def load_embeddings_npz(path: Path) -> np.ndarray:
    data = np.load(str(path), allow_pickle=False)
    if "embeddings" not in data:
        raise ValueError("NPZ does not contain 'embeddings' array")
    return data["embeddings"]

def save_vectorizer(path: Path, vectorizer: TfidfVectorizer) -> None:
    _atomic_write_pickle(path, vectorizer)

def load_vectorizer(path: Path) -> TfidfVectorizer:
    with open(path, "rb") as f:
        return pickle.load(f)

def cache_exists(path: Path) -> bool:
    return path.exists()

# -------------------- end helpers --------------------


def load_chunks(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Chunks file missing: {path}")
    chunks = []
    with path.open(encoding="utf8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "chunk_id" not in obj:
                obj["chunk_id"] = f"{obj.get('source','doc')}-p{obj.get('page','?')}-c{obj.get('chunk_index',0)}"
            chunks.append(obj)
    return chunks


class CombinedRetriever:
    def __init__(
        self,
        chunks_path: Path = CHUNKS_JSONL,
        backup_emb: Path = BACKUP_EMB,
        tfidf_pickle: Path = TFIDF_PICKLE,
        embed_model_name: str = EMBED_MODEL,
        tfidf_max_features: int = TFIDF_MAX_FEATURES,
        force_refresh: bool = False,
    ):
        """
        force_refresh: if True, recompute embeddings and TF-IDF even if caches exist.
        """
        # load chunks
        self.chunks = load_chunks(chunks_path)
        self.ids = [c["chunk_id"] for c in self.chunks]
        self.texts = [c.get("text","") or "" for c in self.chunks]
        self.metas = [ {"source": c.get("source"), "page": c.get("page")} for c in self.chunks]

        # TF-IDF vectorizer (load from cache or fit and save)
        self.vectorizer = None
        self.tfidf_matrix = None
        self._init_tfidf(tfidf_pickle, tfidf_max_features, force_refresh)

        # Embedding model + embeddings (load or compute)
        self.model = SentenceTransformer(embed_model_name)
        self.embeddings = None
        self._init_embeddings(backup_emb, force_refresh)

    def _init_tfidf(self, tfidf_pickle: Path, tfidf_max_features: int, force_refresh: bool):
        if not force_refresh and cache_exists(tfidf_pickle):
            try:
                self.vectorizer = load_vectorizer(tfidf_pickle)
                self.tfidf_matrix = self.vectorizer.transform(self.texts)
                print("Loaded TF-IDF vectorizer from cache:", tfidf_pickle)
                return
            except Exception as e:
                print("Failed to load TF-IDF cache (will refit):", e)

        # fit and save
        print("Fitting TF-IDF vectorizer...")
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=tfidf_max_features)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)
        try:
            save_vectorizer(tfidf_pickle, self.vectorizer)
            print("Saved TF-IDF vectorizer to:", tfidf_pickle)
        except Exception as e:
            print("Warning: failed to save TF-IDF vectorizer cache:", e)

    def _init_embeddings(self, backup_emb: Path, force_refresh: bool):
        if not force_refresh and cache_exists(backup_emb):
            try:
                embs = load_embeddings_npz(backup_emb)
                if embs.shape[0] != len(self.texts):
                    raise ValueError("embeddings length mismatch with chunks; will recompute")
                self.embeddings = embs
                print(f"Loaded precomputed embeddings from {backup_emb} shape={self.embeddings.shape}")
                return
            except Exception as e:
                print("Failed loading embeddings cache (will recompute):", e)

        # compute & save
        print("Computing embeddings (may take a while)...")
        self.embeddings = self.model.encode(self.texts, convert_to_numpy=True, show_progress_bar=True)
        try:
            save_embeddings_npz(backup_emb, self.embeddings)
            print("Saved embeddings to:", backup_emb)
        except Exception as e:
            print("Warning: failed to save embeddings cache:", e)

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        """
        Normalize array to 0..1 (min-max). If all same, return ones.
        """
        if scores.size == 0:
            return scores
        mn = float(np.min(scores))
        mx = float(np.max(scores))
        if mx - mn < 1e-12:
            return np.ones_like(scores)
        return (scores - mn) / (mx - mn)

    def _embed_scores(self, query: str, topn: int = DENSE_CANDIDATES):
        q_emb = self.model.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(q_emb, self.embeddings).flatten()  # [-1,1] or [0,1]
        top_idx = np.argsort(-sims)[:topn]
        top_sims = sims[top_idx]
        return top_idx, top_sims

    def _tfidf_scores(self, query: str, topn: int = TFIDF_TOPN):
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()  # [0,1]
        top_idx = np.argsort(-sims)[:topn]
        top_sims = sims[top_idx]
        return top_idx, top_sims

    def retrieve(
        self,
        query: str,
        dense_topn: int = DENSE_CANDIDATES,
        tfidf_topn: int = TFIDF_TOPN,
        final_k: int = FINAL_TOPK,
        alpha: float = ALPHA,
        embed_threshold: float = EMBED_THRESHOLD
    ) -> List[Dict]:
        """
        Returns list of dicts:
          { chunk_id, text, meta, embed_score, tfidf_score, final_score }
        Scores are normalized to 0..1 before combining.
        """
        # 1) get dense candidates
        idx_d, sims_d = self._embed_scores(query, topn=dense_topn)
        sims_d_norm = self._normalize_scores(sims_d)
        dense_map = { self.ids[i]: {"idx": i, "embed_score": float(s)} for i,s in zip(idx_d, sims_d_norm) }

        # 2) get tfidf candidates
        idx_t, sims_t = self._tfidf_scores(query, topn=tfidf_topn)
        sims_t_norm = self._normalize_scores(sims_t)
        tfidf_map = { self.ids[i]: {"idx": i, "tfidf_score": float(s)} for i,s in zip(idx_t, sims_t_norm) }

        # 3) merge candidate IDs
        cand_ids = list(dict.fromkeys([self.ids[i] for i in idx_d] + [self.ids[i] for i in idx_t]))  # keep order
        candidates = []
        for cid in cand_ids:
            i = self.ids.index(cid)
            embed_score = dense_map.get(cid, {}).get("embed_score", 0.0)
            tfidf_score = tfidf_map.get(cid, {}).get("tfidf_score", 0.0)

            # combine scores
            final_score = alpha * embed_score + (1.0 - alpha) * tfidf_score

            candidates.append({
                "chunk_id": cid,
                "text": self.texts[i],
                "meta": self.metas[i],
                "embed_score": float(embed_score),
                "tfidf_score": float(tfidf_score),
                "final_score": float(final_score)
            })

        # 4) sort by final_score desc and return top-k
        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        return candidates[:final_k]

def cache_info():
    """Print basic info about embedding + TF-IDF caches."""
    print("Cache directory:", BACKUP_DIR.resolve())
    paths = [
        ("Embeddings NPZ", BACKUP_EMB),
        ("TF-IDF Vectorizer", TFIDF_PICKLE),
    ]
    for label, p in paths:
        if p.exists():
            stat = p.stat()
            size_kb = stat.st_size / 1024.0
            mtime = datetime.fromtimestamp(stat.st_mtime)
            print(f"- {label}:")
            print(f"    Path : {p.resolve()}")
            print(f"    Size : {size_kb:.1f} KB")
            print(f"    mtime: {mtime}")
        else:
            print(f"- {label}: (missing)")



# ---------- quick CLI test ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", help="Query (wrap in quotes)", default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--refresh-cache", action="store_true", help="Force recompute embeddings and TF-IDF (invalidate caches)")
    parser.add_argument("--cache-info", action="store_true", help="Show cache file info and exit")
    args = parser.parse_args()

    # if user only wants cache info
    if args.cache_info:
        cache_info()
        sys.exit(0)

    force_refresh = bool(args.refresh_cache)
    if args.query is None:
        q = input("Query: ").strip()
    else:
        q = args.query

    r = CombinedRetriever(force_refresh=force_refresh)
    res = r.retrieve(q, final_k=args.k)
    for i, r in enumerate(res, start=1):
        print(f"\n[{i}] final={r['final_score']:.4f} embed={r['embed_score']:.4f} tfidf={r['tfidf_score']:.4f} source={r['meta'].get('source')} page={r['meta'].get('page')}")
        print(r['text'][:400].replace("\n"," ") + "...")
