# pdfclean.py
"""
Ultra-simple PDF text cleaning:
- Remove obvious page numbers
- Remove extra blank lines
- Fix hyphenated line breaks
- Normalize spaces
"""

import re
from typing import List
from langchain_core.documents import Document


def clean_text_basic(text: str) -> str:
    """
    Basic cleaning without modifying structure.
    Keeps all content safe.
    """
    # Remove hyphen at line break: "exam-\nple" → "example"
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

    # Remove classic page numbers only (safe)
    lines = text.splitlines()
    cleaned_lines = []
    for ln in lines:
        stripped = ln.strip()
        if re.fullmatch(r"\d+", stripped):   # line is only digits (page number)
            continue
        if re.fullmatch(r"Page\s*\d+", stripped, flags=re.I):
            continue
        cleaned_lines.append(ln)

    # Rejoin safely
    text = "\n".join(cleaned_lines)

    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove extra spaces
    text = re.sub(r'[ \t]{2,}', ' ', text)

    return text.strip()


def simple_clean(docs: List[Document]) -> List[Document]:
    """
    Apply basic cleaning to each page Document.
    """
    cleaned_docs = []
    for d in docs:
        cleaned_text = clean_text_basic(d.page_content)
        cleaned_docs.append(
            Document(page_content=cleaned_text, metadata=d.metadata)
        )
    return cleaned_docs
