# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python RAG (Retrieval-Augmented Generation) chatbot for Bahria University Karachi Campus (BUKC). It ingests university PDF documents and answers student queries using hybrid search + Groq (Llama 3.1).

## Setup & Commands

```bash
# Install dependencies
pip install -r requirement.txt
pip install langchain-groq python-dotenv  # missing from requirement.txt

# Step 1: Ingest PDFs and build vector database (run once before the app)
python ingest.py

# Step 2: Launch the chatbot UI
streamlit run app.py

# Run evaluation
python evaluator.py
```

**Required environment variable** — set in `.env` or shell before running:
```bash
GROQ_API_KEY=your_key_here
```

No test runner, linter, or build system is configured.

## Architecture

### Pipeline Overview

```
data/*.pdf
  → ingest.py  (PyPDFLoader → chunk → BAAI/bge-small-en embeddings → ChromaDB persist)
  → vector_db/ (persistent ChromaDB store)

User query (Streamlit)
  → app.py       (loads PDFs via @st.cache_resource, calls hybrid_search())
  → retriever.py (ChromaDB vector search + BM25 keyword search → CrossEncoder rerank → top 3 docs)
  → app.py       (context + question → ChatGroq llama-3.1-8b-instant → display response)
```

### Key Design Points

- `retriever.py` loads the ChromaDB and CrossEncoder **at module import time** (globals), so both are initialized once per Streamlit session, not per query.
- `app.py` loads raw PDF docs via `@st.cache_resource` and passes them into `hybrid_search()` on each query — BM25 is rebuilt per query from these docs.
- `hybrid_search(query, docs)` takes the full doc list so BM25 can operate over the same corpus as the vector store. Vector search uses the pre-loaded `vector_retriever`; results are merged, deduplicated, then reranked.
- `ingest.py` is a standalone script (not imported anywhere). Re-run it whenever `data/` changes to rebuild `vector_db/`.
- `evaluator.py` defines `run_evaluation(dataset)` but does not generate the dataset itself — caller must supply a RAGAS-compatible `Dataset` object.

### Models Used

| Model | Purpose |
|-------|---------|
| `BAAI/bge-small-en` | Sentence embeddings (HuggingFace, runs locally) |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranking retrieved docs (runs locally) |
| `llama-3.1-8b-instant` via Groq | LLM for answer generation |
