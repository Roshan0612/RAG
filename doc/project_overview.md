













































# Project Overview

This project is a small RAG application for processing raw documents and turning them into searchable chunks.

The ingestion pipeline reads files from the `doc` folder, splits them into smaller chunks, and stores them in a vector database for semantic search.

## Why this matters

Document retrieval works better when content is segmented into meaningful chunks. This allows a language model to retrieve the most relevant context before answering a user question.

## Data flow

1. Put raw knowledge files inside the `doc` folder.
2. The ingestion script reads all supported text files.
3. Chunks are created using a recursive text splitter.
4. Embeddings are generated and stored in Chroma.
5. The vector database is later queried in the retrieval step.
