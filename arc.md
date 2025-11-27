             ┌────────────────────┐
             │       PDFs          │
             └─────────┬──────────┘
                       ▼
       ┌────────────────────────────────┐
       │       Data Processing Layer     │
       │  - Extract text (PDF parser)    │
       │  - Clean & normalize            │
       │  - Chunk into 200–500 tokens    │
       └────────────────┬───────────────┘
                        ▼
       ┌────────────────────────────────────┐
       │         Embedding Layer             │
       │   Hugging Face Models (Free)        │
       │   e.g., all-MiniLM-L6-v2, BGE, E5   │
       └────────────────┬───────────────────┘
                        ▼
       ┌────────────────────────────────┐
       │        Vector Database          │
       │  Chroma / Pinecone / Weaviate   │
       └────────────────┬───────────────┘
                        ▼
   ┌────────────────────────────────────────┐
   │        Retrieval Layer (High Accuracy) │
   │  - Dense Vector Search (HF embeddings) │
   │  - Keyword Search (BM25)               │
   │  - Reranker (optional, HF models)      │
   └───────────────────┬────────────────────┘
                       ▼
       ┌────────────────────────────────┐
       │         LLM Generator          │
       │      GPT-5.1 / GPT-4.1 etc     │
       │  (Uses retrieved context only)  │
       └────────────────┬───────────────┘
                        ▼
             ┌────────────────────┐
             │   Final Answer      │
             └────────────────────┘
