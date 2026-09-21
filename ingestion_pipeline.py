from pathlib import Path
import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "doc"
VECTORSTORE_DIR = BASE_DIR / "chroma_db"


def load_documents_from_folder(folder: Path):
    """Read text files from the doc folder and convert them to LangChain documents."""
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    docs = []
    valid_extensions = {".txt", ".md", ".csv"}

    for file_path in sorted(folder.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            text = file_path.read_text(encoding="utf-8")
            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": str(file_path.relative_to(BASE_DIR))},
                    )
                )

    return docs


def build_vectorstore():
    """Create chunks from raw docs and persist them in a local Chroma DB."""
    load_dotenv(BASE_DIR / ".env")

    docs = load_documents_from_folder(DOCS_DIR)
    if not docs:
        raise ValueError(f"No raw documents were found in {DOCS_DIR}. Add .txt/.md files there.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(docs)

    print("=" * 80)
    print("INGESTION TEST SUMMARY")
    print("=" * 80)
    print(f"SOURCE_FILES={len(docs)}")
    print(f"CHUNKS_CREATED={len(chunks)}")
    print()

    for idx, chunk in enumerate(chunks[:5], start=1):
        print(f"--- CHUNK {idx} ---")
        print(f"SOURCE: {chunk.metadata.get('source')}")
        preview = chunk.page_content[:200].replace("\n", " ").strip()
        print(f"PREVIEW: {preview}")
        print("CHUNK_LENGTH:", len(chunk.page_content))
        print()

    print("=" * 80)

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        print("EMBEDDINGS=OpenAI")
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("EMBEDDINGS=LocalHF")

    print("\n" + "=" * 80)
    print("EMBEDDING TEST")
    print("=" * 80)
    sample_embedding = embeddings.embed_query(chunks[0].page_content)
    print(f"EMBEDDING_DIMENSION={len(sample_embedding)}")
    print(f"EMBEDDING_SAMPLE={sample_embedding[:10]}")
    print("=" * 80)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )

    print(f"VECTOR_DB_PATH={VECTORSTORE_DIR}")
    print("VECTOR_DB_STATUS=CREATED")

    return vectorstore


if __name__ == "__main__":
    try:
        build_vectorstore()
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        print("Create a folder named 'doc' inside this project and add your raw files there.")
    except ValueError as exc:
        print(f"Error: {exc}")
    except Exception as exc:
        print(f"Unexpected error: {exc}")
