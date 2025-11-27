# frontend/app.py
"""
Simple FastAPI frontend for your RAG system.

Endpoints:
  GET  /        -> serves the index.html UI
  POST /ask     -> accepts JSON { "question": "..." } and returns:
                   { "answer": "...", "retrieved": [{chunk_id, meta, text, final_score}, ...] }

Run:
  uvicorn frontend.app:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import traceback
import os
from pathlib import Path

# Import your existing retriever and LLM call
# Ensure paths are correct; this assumes your project root contains rag_cli.py and retrieval/combined_retriever.py
try:
    from retrevial.combined_retriever import CombinedRetriever
except Exception as e:
    raise RuntimeError("Failed to import CombinedRetriever. Make sure retrieval/combined_retriever.py exists and is correct.") from e

try:
    # import call_llm from your rag_cli module (adjust name if different)
    from rag_cli import call_llm
except Exception:
    # try alternative location (if your CLI file is named differently)
    try:
        from retrevial.rag_cli import call_llm
    except Exception:
        raise RuntimeError("Failed to import call_llm from rag_cli.py. Ensure rag_cli.py defines call_llm(prompt).")

# setup app
app = FastAPI(title="RAG Frontend")

# Serve static folder with index.html
STATIC_DIR = Path(__file__).resolve().parent / "static"
if not STATIC_DIR.exists():
    raise RuntimeError(f"Static dir missing: {STATIC_DIR}. Create frontend/static/index.html first.")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# create retriever instance once (this will load caches)
_retriever = None

@app.on_event("startup")
def startup_event():
    global _retriever
    # Force refresh can be toggled; default False
    try:
        _retriever = CombinedRetriever(force_refresh=False)
    except Exception as e:
        print("Warning: failed to initialize CombinedRetriever at startup:", e)
        traceback.print_exc()
        _retriever = None

# Pydantic model for request
class AskRequest(BaseModel):
    question: str
    k: int = 5

@app.get("/", response_class=HTMLResponse)
def index():
    index_file = STATIC_DIR / "index.html"
    return FileResponse(index_file)

@app.post("/ask")
def ask(req: AskRequest):
    global _retriever
    if not _retriever:
        # lazy init as fallback
        try:
            _retriever = CombinedRetriever(force_refresh=False)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Retriever init failed: {e}")

    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty question")

    try:
        # 1) retrieval
        retrieved = _retriever.retrieve(q, final_k=req.k)
        # map to minimal payload
        retrieved_payload = []
        for r in retrieved:
            retrieved_payload.append({
                "chunk_id": r.get("chunk_id"),
                "meta": r.get("meta"),
                "text": r.get("text"),
                "embed_score": r.get("embed_score"),
                "tfidf_score": r.get("tfidf_score"),
                "final_score": r.get("final_score")
            })

        # 2) build prompt same as CLI's build_prompt format
        # simple prompt: include numbered passages and instruct the LLM to cite [1]...[k]
        ctx_lines = []
        for i, r in enumerate(retrieved_payload, start=1):
            meta = r.get("meta") or {}
            src = meta.get("source", "unknown")
            page = meta.get("page", "?")
            ctx_lines.append(f"[{i}] Source: {src} | Page: {page}\n{r['text']}\n")
        context_block = "\n---\n".join(ctx_lines)
        prompt = (
            "You are an assistant that answers questions using ONLY the provided context passages. "
            "If the answer is not contained in the context, respond exactly: \"I don't know based on the provided documents.\""
            + "\n\n"
            + f"Question: {q}\n\nContext:\n{context_block}\n\n"
            "Instructions:\n"
            "1) Answer concisely (max 300 words).\n"
            "2) Use only the context passages above.\n"
            "3) Cite passages using bracketed numbers, e.g., [1], [3].\n"
            "4) If not found, reply exactly: \"I don't know based on the provided documents.\"\n"
        )

        # 3) call LLM (synchronous)
        answer = call_llm(prompt)

        return JSONResponse({
            "answer": answer,
            "retrieved": retrieved_payload
        })
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
