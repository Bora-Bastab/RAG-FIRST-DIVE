import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from embedding_utils import get_embedding_model
from langchain_community.vectorstores import FAISS

INDEX_DIR = "travel_knowledge_base"  # Directory to store the travel knowledge base
DATA_DIR = "travel_documents"

def setup_travel_knowledge_base(reset: bool = False):
    """
    Sets up or resets the travel knowledge base.
    """
    if reset:
        clear_knowledge_base()

    # Step 1: Load travel documents
    documents = load_travel_documents()
    # Step 2: Split documents into chunks
    chunks = split_travel_documents(documents)
    # Step 3: Create the FAISS index
    create_faiss_index(chunks)

def load_travel_documents():
    """
    Loads travel documents from the specified directory.
    """
    loader = PyPDFDirectoryLoader(DATA_DIR)
    documents = loader.load()
    print("Loaded Travel Documents:")
    for doc in documents:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    return documents

def split_travel_documents(documents: list[Document]):
    """
    Splits travel documents into smaller chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        if "source" not in chunk.metadata:
            chunk.metadata["source"] = "Unknown"
    return chunks

def create_faiss_index(chunks: list[Document]):
    """
    Creates a FAISS index from the travel document chunks.
    """
    embedding_model = get_embedding_model()
    travel_knowledge_base = FAISS.from_documents(chunks, embedding_model)
    travel_knowledge_base.save_local(INDEX_DIR)
    print(f"✅ Travel knowledge base saved to {INDEX_DIR}")

def clear_knowledge_base():
    """
    Clears the existing travel knowledge base.
    """
    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
        print(f"🗑️ Deleted travel knowledge base at {INDEX_DIR}")
    else:
        print(f"❌ No travel knowledge base found at {INDEX_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the travel knowledge base.")
    args = parser.parse_args()
    setup_travel_knowledge_base(reset=args.reset)
