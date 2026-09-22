from pathlib import Path
import sys

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_DIR = BASE_DIR / "chroma_db"


def get_vectorstore():
    """Load the persisted local Chroma vector store."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings,
    )

    return vectorstore


def retrieve_documents(query: str, top_k: int = 3):
    """Return the most relevant chunks for a user query."""
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=top_k)

    if not results:
        return []

    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What is Retrieval-Augmented Generation and how does it work?"

    retrieve_documents(query, top_k=3)
