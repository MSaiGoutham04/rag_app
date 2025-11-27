# chunker_utils.py
import uuid
from typing import List
import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")

def add_ids_and_token_counts(chunks: List[dict]) -> List[dict]:
    """
    Add deterministic chunk_id and token_count to each chunk dict.
    Input chunk dicts must contain: 'source', 'page', 'chunk_index', 'text'
    """
    out = []
    for c in chunks:
        text = c.get("text", "") or ""
        token_ids = ENC.encode(text)
        token_count = len(token_ids)
        chunk_id = f"{c.get('source','doc')}-p{c.get('page','?')}-c{c.get('chunk_index',0)}-{str(uuid.uuid4())[:8]}"
        new = dict(c)  # copy
        new.update({"chunk_id": chunk_id, "token_count": token_count})
        out.append(new)
    return out

# chunker_utils.py (append)
import json
from pathlib import Path

def save_chunks_jsonl(chunks: List[dict], out_path: str = r"D:\RAG\out"):
    """
    Save list of chunk dicts to JSONL, one chunk per line.
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return str(p.resolve())
