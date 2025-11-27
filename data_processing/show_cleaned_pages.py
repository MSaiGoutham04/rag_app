# show_cleaned_pages.py
"""
Load a PDF page-wise, apply the ultra-simple cleaner (pdfclean.simple_clean),
and print cleaned pages to console with clear page headers.

Usage:
  python show_cleaned_pages.py path/to/sample.pdf         # prints all pages
  python show_cleaned_pages.py path/to/sample.pdf 3       # prints only page 3
"""

import sys
from pathlib import Path

from langchain_community.document_loaders import PDFPlumberLoader
from pdfclean import simple_clean   # your simple cleaning module

def print_page_header(page_num: int):
    print("\n" + "=" * 12 + f" PAGE {page_num} " + "=" * 12 + "\n")

def show_cleaned_pages(pdf_path: str, single_page: int | None = None):
    pdf_path = Path(pdf_path)
    loader = PDFPlumberLoader(str(pdf_path))
    docs = loader.load()  # list of Documents, one per page

    # simple_clean expects List[Document] and returns cleaned Documents
    cleaned_docs = simple_clean(docs)

    total = len(cleaned_docs)
    if total == 0:
        print("No pages extracted.")
        return

    if single_page is not None:
        if single_page < 1 or single_page > total:
            print(f"Invalid page number {single_page}. PDF has {total} pages.")
            return
        docs_to_print = [(single_page, cleaned_docs[single_page - 1])]
    else:
        docs_to_print = list(enumerate(cleaned_docs, start=1))

    for page_num, doc in docs_to_print:
        print_page_header(page_num)
        text = doc.page_content or ""
        if not text:
            print("[No text on this page after cleaning]")
        else:
            print(text)
        print("\n")  # small gap between pages

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python show_cleaned_pages.py path/to/sample.pdf [page_number]")
        sys.exit(1)
    pdf = sys.argv[1]
    page = int(sys.argv[2]) if len(sys.argv) >= 3 else None
    show_cleaned_pages(pdf, single_page=page)