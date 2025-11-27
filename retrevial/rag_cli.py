# rag_cli.py
"""
Interactive RAG CLI:
- Type a question
- Retrieves top-k chunks from Chroma
- Sends context+question to LLM
- Shows answer + sources
"""

import chromadb
from pathlib import Path
from retrevial.combined_retriever import CombinedRetriever

import traceback
# import re
# from textwrap import shorten

# # helper: map bracket citations like [1] to retrieved passages and print them
# CITE_SNIPPET_LEN = 300  # chars shown for each cited source

# def print_answer_with_sources(answer_text: str, retrieved: list):
#     """
#     answer_text: string returned by LLM (may include [1], [2] citations)
#     retrieved: list of retrieved dicts in the current order (1-based indices)
#     Behavior:
#       - If answer contains citations like [1], [2] -> show those source snippets.
#       - Otherwise show the top 3 retrieved passages as 'Sources'.
#     """
#     print("=======================================")
#     print("📘 ANSWER:")
#     print("=======================================\n")
#     print(answer_text)
#     print("\n---------------------------------------\n")

#     # find bracketed citations like [1], [2]
#     cited_idxs = [int(n) for n in re.findall(r"\[(\d+)\]", answer_text)]
#     if cited_idxs:
#         # unique and preserve order
#         seen = []
#         for i in cited_idxs:
#             if i not in seen:
#                 seen.append(i)
#         print("🔎 Cited sources (from answer):")
#         for idx in seen:
#             if 1 <= idx <= len(retrieved):
#                 r = retrieved[idx - 1]
#                 meta = r.get("meta", {})
#                 src = meta.get("source", "unknown")
#                 page = meta.get("page", "?")
#                 snippet = (r.get("text") or "").replace("\n", " ")
#                 snippet = shorten(snippet, width=CITE_SNIPPET_LEN, placeholder="...")
#                 print(f"[{idx}] {src} | page: {page} | chunk_id: {r.get('chunk_id','-')}")
#                 print(f"    {snippet}\n")
#             else:
#                 print(f"[{idx}] (citation index out of range)")
#     else:
#         # No explicit citations in answer — show top-N retrieved as sources
#         N = min(3, len(retrieved))
#         if N == 0:
#             print("No retrieved passages to show as sources.")
#             return
#         print("🔎 Top retrieved sources (no explicit citations detected):")
#         for i in range(N):
#             r = retrieved[i]
#             meta = r.get("meta", {})
#             src = meta.get("source", "unknown")
#             page = meta.get("page", "?")
#             snippet = (r.get("text") or "").replace("\n", " ")
#             snippet = shorten(snippet, width=CITE_SNIPPET_LEN, placeholder="...")
#             print(f"[{i+1}] {src} | page: {page} | chunk_id: {r.get('chunk_id','-')}")
#             print(f"    {snippet}\n")
#     print("=======================================\n")


# ---------- CONFIG ----------
CHROMA_DIR = Path(r"D:\RAG\out\chroma_db") # persistent chroma dir
COLLECTION_NAME = "pdf_chunks"
TOP_K = 5
LLM_MODEL = "gpt-oss-20b"  # Change to your preferred model
MAX_CONTEXT_CHARS = 4000   # safety limit
# ----------------------------

# ----------- LLM Call (OpenAI v1.x) -----------
from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
) # auto-reads OPENAI_API_KEY

def call_llm(prompt: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-oss-20b",    # or any model you want
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=600
    )
    return resp.choices[0].message.content


# ------------------------------------------

# ------------ CHROMA CLIENT ---------------
def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))  # point to your DB dir
    try:
        col = client.get_collection(COLLECTION_NAME)
        print(f"Using existing collection: {COLLECTION_NAME}")
    except Exception:
        col = client.create_collection(COLLECTION_NAME)
        print(f"Created collection: {COLLECTION_NAME}")
    return col

# -------------------------------------------

# ------------- RETRIEVAL -------------------
def retrieve(query: str, k: int = TOP_K):
    col = get_collection()
    res = col.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    results = []
    for d, m, dist in zip(docs, metas, dists):
        results.append({"text": d, "meta": m, "score": dist})
    return results
# -------------------------------------------

# ---------- PROMPT BUILDER -----------------
def build_prompt(question: str, retrieved):
    header = "Use ONLY the context passages to answer. If answer is not found, say: \"I don't know based on the provided documents.\"\n\n"

    ctx = []
    for i, r in enumerate(retrieved, start=1):
        src = r["meta"].get("source", "unknown")
        page = r["meta"].get("page", "?")
        ctx.append(f"[{i}] (page {page}) {src}\n{r['text']}\n")

    context_block = "\n---\n".join(ctx)

    prompt = f"""{header}
Question: {question}

Context:
{context_block}

Instructions:
- Answer in 5–10 lines.
- Cite sources like [1], [2] at end.
"""

    if len(prompt) > MAX_CONTEXT_CHARS:
        prompt = prompt[:MAX_CONTEXT_CHARS] + "\n[Context truncated]\n"

    return prompt
# -------------------------------------------


# # ============ INTERACTIVE LOOP =============
# def main():
#     print("\n===============================")
#     print("  Interactive RAG Terminal CLI ")
#     print("===============================\n")
#     print("Type your question below. Press Ctrl+C to exit.\n")

#     while True:
#         try:
#             q = input("❓ Your question: ").strip()
#             if not q:
#                 continue

#             # print("\n🔍 Retrieving context...")
#             # retrieved = retrieve(q)
            
#             # new:

#             _retriever = CombinedRetriever()   # create once at program start
#             retrieved = _retriever.retrieve(q, final_k=5)
#             # retrieved is list of dicts with keys: chunk_id, text, meta, embed_score, tfidf_score, final_score


#             print("🧠 Generating answer...\n")
#             prompt = build_prompt(q, retrieved)
#             answer = call_llm(prompt)

#             print("=======================================")
#             print("📘 ANSWER:")
#             print("=======================================\n")
#             print(answer)
#             print("\n---------------------------------------\n")
#             # show answer and linked sources
#             # print_answer_with_sources(answer, retrieved)

#         except KeyboardInterrupt:
#             print("\nExiting RAG CLI. Bye! 👋")
#             break
#         except Exception as e:
#             print("\n❌ Error:", e)
#             print("Check your environment or LLM API key.\n")
# # ===========================================

# if __name__ == "__main__":
#     main()


# ============ INTERACTIVE LOOP =============
# Wire CombinedRetriever in here (import it)
try:
    from combined_retriever import CombinedRetriever
except Exception:
    # If the import fails, we'll still allow fallback to the old retrieve() function above
    CombinedRetriever = None

def main():
    print("\n===============================")
    print("  Interactive RAG Terminal CLI ")
    print("===============================\n")
    print("Type your question below. Press Ctrl+C to exit.\n")

    # create CombinedRetriever once at program start (if available)
    _retriever = None
    if CombinedRetriever is not None:
        try:
            _retriever = CombinedRetriever()
            print("CombinedRetriever initialized and ready.")
        except Exception as e:
            print("Warning: CombinedRetriever failed to initialize, falling back to Chroma retrieve().")
            traceback.print_exc()

    while True:
        try:
            q = input("❓ Your question: ").strip()
            if not q:
                continue

            # Use CombinedRetriever if available, otherwise fallback to old retrieve()
            if _retriever is not None:
                # use high-accuracy combined retriever
                retrieved = _retriever.retrieve(q, final_k=TOP_K)
                # ensure each retrieved item matches expected keys for build_prompt:
                # Each item must have keys: 'text' and 'meta' (meta should be a dict)
                # CombinedRetriever returns 'meta' as {'source','page'} so it's compatible.
            else:
                # fallback: use the simple Chroma retrieval
                print("\n🔍 Retrieving context via fallback Chroma...")
                retrieved = retrieve(q, k=TOP_K)

            print("🧠 Generating answer...\n")
            prompt = build_prompt(q, retrieved)
            answer = call_llm(prompt)

            print("=======================================")
            print("📘 ANSWER:")
            print("=======================================\n")
            print(answer)
            print("\n---------------------------------------\n")

        except KeyboardInterrupt:
            print("\nExiting RAG CLI. Bye! 👋")
            break
        except Exception as e:
            print("\n❌ Error:", e)
            traceback.print_exc()
            print("Check your environment or LLM API key.\n")
# ===========================================

if __name__ == "__main__":
    main()