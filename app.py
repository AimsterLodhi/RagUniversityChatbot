import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from retriever import hybrid_search, VECTOR_DB_PATH

# Load .env file
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

st.title("University RAG Chatbot")

@st.cache_resource
def load_models():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from sentence_transformers import CrossEncoder
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
    db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
    vector_retriever = db.as_retriever(search_kwargs={"k": 5})
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return vector_retriever, reranker

@st.cache_resource
def load_docs():
    docs = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_DIR, file))
            docs.extend(loader.load())
    return docs

vector_retriever, reranker = load_models()
docs = load_docs()

query = st.text_input("Ask your question:")

if query:

    top_docs = hybrid_search(query, docs, vector_retriever, reranker)

    context = "\n".join([doc.page_content for doc in top_docs])

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY
    )

    prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question: {query}

If answer is not in context, say "I don't know".
"""

    response = llm.invoke(prompt)

    st.write(response.content)