# FullRAGSystem  
**A Modular Retrieval-Augmented Generation (RAG) Pipeline with Semantic Chunking, Pinecone, and Gemini**

---

## Overview

FullRAGSystem is a Python-based implementation of a **Retrieval-Augmented Generation (RAG)** pipeline designed for **accurate, document-grounded question answering**.

The system emphasizes:
- Semantic correctness over brute-force chunking
- Deterministic behavior over creative hallucination
- Clear separation of ingestion, retrieval, reranking, and generation

It is intended for developers building **production-grade RAG systems**, not tutorial-level demos.

---

## Key Features

- **Semantic Chunking**
  - Sentence-level chunking using embedding similarity
  - Preserves conceptual coherence across chunks

- **Vector Search with Pinecone**
  - Efficient similarity-based retrieval
  - Document-scoped filtering via metadata

- **Cross-Encoder Reranking**
  - Precise relevance scoring using query–chunk pairs
  - Reduces noise from approximate vector search

- **Strict Context Grounding**
  - LLM answers are constrained to retrieved content only
  - Explicit fallback when information is missing

- **Modular Architecture**
  - Clear separation of responsibilities
  - Easy to extend or replace individual components

---

## Architecture

Raw Text
↓
Semantic Chunking (sentence + similarity)
↓
SentenceTransformer Embeddings
↓
Pinecone Vector Database
↓
Vector Retrieval (top-k)
↓
Cross-Encoder Reranking
↓
Gemini Answer Generation (context-only)

---

## Technology Stack

| Layer | Technology |
|-----|-----------|
| Language | Python |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Vector Database | Pinecone |
| LLM | Google Gemini (gemini-2.5-flash) |
| NLP Utilities | NLTK |

---

## Installation

### Prerequisites
- Python 3.9+
- Active Pinecone account
- Google Generative AI API access
