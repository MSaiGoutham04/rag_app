# chunker.py
"""
Minimal chunker using TokenTextSplitter from langchain_text_splitters.

Input:  docs = list of Documents (cleaned, page-wise) with:
          - doc.page_content (text)
          - doc.metadata containing at least 'page' and optionally 'source'

Output: list of dicts:
  { "source", "page", "chunk_index", "text" }
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter

def chunk_documents(
    docs: List[Document],
    chunk_size: int = 350,
    chunk_overlap: int = 80,
    encoding: str = "cl100k_base"
) -> List[dict]:
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name=encoding
    )

    chunks = []
    for doc in docs:
        source = doc.metadata.get("source")
        page = doc.metadata.get("page")

        pieces = splitter.split_text(doc.page_content or "")
        for idx, piece in enumerate(pieces):
            chunks.append({
                "source": source,
                "page": page,
                "chunk_index": idx,
                "text": piece
            })

    return chunks
