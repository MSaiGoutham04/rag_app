# from langchain_community.document_loaders import PDFPlumberLoader
# from pdfclean import simple_clean
# from chunker import chunk_documents
# pdf_path = r"D:\RAG\input\AI-102_StudyGuide_ENU_FY23Q1_7.1.pdf"

# loader = PDFPlumberLoader(pdf_path)
# docs = loader.load()   # list of Documents, one per page
# cleaned_docs = simple_clean(docs)
# chunks = chunk_documents(cleaned_docs, chunk_size=350, chunk_overlap=80)

# print(cleaned_docs[0].page_content[:1000])

# print("Total Pages Extracted:", len(docs))

# for d in docs[:3]:
#     print(f"\n--- Page {d.metadata['page']} ---")
#     print(d.page_content[:500])

### Main FIle In DATA PROCESSING LAYER ###


from chunker import chunk_documents        # your chunker
from pdfclean import simple_clean
from langchain_community.document_loaders import PDFPlumberLoader
from chunker_utils import add_ids_and_token_counts, save_chunks_jsonl

# load, clean, chunk (you already did this earlier)
loader = PDFPlumberLoader(r"D:\RAG\input\AI-102_StudyGuide_ENU_FY23Q1_7.1.pdf")
docs = loader.load()
cleaned = simple_clean(docs)
chunks = chunk_documents(cleaned, chunk_size=350, chunk_overlap=80)

# add ids & token counts
chunks = add_ids_and_token_counts(chunks)

# save
path = save_chunks_jsonl(chunks, out_path="out/chunks.jsonl")
print("Saved chunks to:", path)
print("Sample chunk meta:", chunks[0]['chunk_id'], chunks[0]['token_count'])




