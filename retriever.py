import os
from rank_bm25 import BM25Okapi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db")

# ----------------------------
# BM25 Search Function
# ----------------------------

def bm25_search(query, docs):
    corpus = [doc.page_content for doc in docs]
    tokenized_corpus = [doc.split() for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.split()

    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:5]

    return [docs[i] for i in top_indices]

# ----------------------------
# Hybrid Search Function
# ----------------------------

def hybrid_search(query, docs, vector_retriever, reranker):

    # 1. Vector search
    vector_docs = vector_retriever.invoke(query)
    # 2. BM25 search
    bm25_docs = bm25_search(query, docs)

    # 3. Combine results
    all_docs = vector_docs + bm25_docs

    # 4. Remove duplicates
    seen = set()
    unique_docs = []

    for doc in all_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)

    # 5. Reranking using Cross Encoder
    pairs = [(query, doc.page_content) for doc in unique_docs]
    scores = reranker.predict(pairs)

    scored_docs = list(zip(unique_docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # 6. Return top 3
    return [doc for doc, score in scored_docs[:3]]