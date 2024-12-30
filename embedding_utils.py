from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model():
    """
    Returns a HuggingFace embedding model for converting text into vectors.
    """
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
