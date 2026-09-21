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


def build_prompt(history, query, context_docs):
    context_text = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in context_docs
    )

    chat_history = "\n".join(
        f"User: {h['question']}\nAssistant: {h['answer']}" for h in history
    )

    template = """
You are a helpful assistant grounded in the retrieved document context.
Use the conversation history when it helps understand the user's intent.

Conversation history:
{history}

Retrieved context:
{context}

Question:
{question}

Answer based only on the context and the conversation. If the answer is not present, say so clearly.
"""

    return PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
    ).format(history=chat_history, context=context_text, question=query)


def generate_history_aware_answer(history, query, top_k=3):
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(query, k=top_k)

    if not docs:
        return "I could not find relevant information in the knowledge base."

    prompt = build_prompt(history, query, docs)

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
        return "Ollama model could not be loaded. Please ensure a compatible model is installed."

    print("=" * 80)
    print("HISTORY-AWARE ANSWER")
    print("=" * 80)
    print(f"Question: {query}")
    print("\nRetrieved context:")
    for i, doc in enumerate(docs, start=1):
        print(f"\n--- CONTEXT {i} ---")
        print(doc.page_content[:400].replace("\n", " ").strip())
    print("\n" + "=" * 80)
    print("Answer:")
    print(answer)
    print("=" * 80)

    return answer


if __name__ == "__main__":
    history = [
        {
            "question": "What is RAG?",
            "answer": "RAG stands for Retrieval-Augmented Generation and combines retrieval with a language model.",
        }
    ]

    question = "How does it help AI systems?"
    generate_history_aware_answer(history, question)
