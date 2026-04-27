import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

#Load all PDFs from data folder
docs = []

for file in os.listdir("data"):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(f"data/{file}")
        docs.extend(loader.load())

#chunking (context-aware-splitting)
splitter=RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=100
)
chunks=splitter.split_documents(docs)

#craete embeddins 
embeddings=HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en"

)

#STORE IN VECTOR DB (Chroma)
db=Chroma.from_documents(
    chunks, 
    embeddings, 
    persist_directory="vector_db"
)
db.persist()
print("Vector DB created successfully!")