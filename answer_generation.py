from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_DIR = BASE_DIR / "chroma_db"


def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return Chroma(
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings,
    )


def build_prompt(query: str, context_docs):
    context_text = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in context_docs
    )

    template = """
You are a helpful assistant. Answer the user's question based only on the context below.

Context:
{context}

Question:
{question}

Give a concise but accurate answer. If the answer is not in the context, say that clearly.
"""

    return PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    ).format(context=context_text, question=query)


def generate_answer(query: str, top_k: int = 3):
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(query, k=top_k)

    if not docs:
        return "I could not find relevant information in the knowledge base."

    prompt = build_prompt(query, docs)
    model_names = ["llama3.2:1b", "llama3.1", "llama3.2"]
    answer = None

    for model_name in model_names:
        try:
            llm = Ollama(model=model_name)
            answer = llm.invoke(prompt)
            break
        except Exception:
            continue

    if answer is None:
        answer = (
            "Ollama is not available or no compatible model is installed. "
            "Please run: ollama pull llama3.2:1b"
        )

    return answer


if __name__ == "__main__":
    question = "What is RAG and how does it help AI systems?"
    generate_answer(question)
