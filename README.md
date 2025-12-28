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

### Install Dependencies

```bash
pip install torch sentence-transformers nltk pinecone-client python-dotenv google-generativeai
Configuration
Create a .env file in the project root:
GOOGLE_API_KEY=your_google_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=test
The application will fail fast if required variables are missing.
Usage
Document Ingestion
rag.ingest_document(raw_text, doc_id)
What ingestion does:
Deletes existing vectors for the document ID
Applies semantic chunking
Generates embeddings for each chunk
Uploads vectors to Pinecone with metadata
Querying Documents
rag.search(
    query="What is Machine Learning?",
    allowed_doc_ids=["ai_doc"]
)
Important notes:
Queries are restricted to the provided document IDs
There is no implicit global search
This prevents accidental cross-document leakage
Answer Generation Policy
The system enforces strict answer rules:
Answers must be derived only from retrieved chunks
External knowledge is never used
If relevant context is missing, the system responds:
The document does not specify this.
This design significantly reduces hallucination risk.
Example Queries
queries = [
    ("ai_doc", "What is Machine Learning?"),
    ("bio_doc", "What is photosynthesis?"),
    ("history_doc", "When did the French Revolution begin?")
]
