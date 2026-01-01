ContextIQ
AI-Powered Intelligent Document Intelligence Platform

🚀 GDG TechSprint Hackathon Project | GDG MNNIT

<p align="center"> <img src="./logo.jpg" alt="ContextIQ Logo" width="160"/> </p> <p align="center"> <b>Understand documents the way humans do.</b><br/> Upload, summarize, search, and ask — all in context. </p> <p align="center"> <img src="https://img.shields.io/badge/GDG-TechSprint-blue?style=for-the-badge"/> <img src="https://img.shields.io/badge/AI-RAG-success?style=for-the-badge"/> <img src="https://img.shields.io/badge/FastAPI-Backend-brightgreen?style=for-the-badge"/> <img src="https://img.shields.io/badge/Status-Hackathon%20Build-orange?style=for-the-badge"/> </p>
🌍 Problem Statement

In academia and industry, people constantly deal with large, complex documents — PDFs, scanned files, notes, reports, and images.

Existing challenges:

Reading long documents is time-consuming

Keyword search fails to capture semantic meaning

Scanned/image documents are hard to analyze

No unified way to ask questions across documents

💡 Our Solution — ContextIQ

ContextIQ is an AI-powered document intelligence system that transforms raw documents into an interactive, searchable, and queryable knowledge base using Retrieval-Augmented Generation (RAG).

It doesn’t just store documents —
👉 it understands them in context.

✨ Key Features
📤 Intelligent Document Upload

Supports PDF, TXT & image files

OCR using EasyOCR

Automatic text extraction & cleanup

🧾 AI-Generated Unified Summaries

Single coherent summary per document

Preserves technical accuracy & chronology

Clean HTML-rendered summaries for UI

🔍 Semantic Search (Meaning > Keywords)

Sentence-level vector embeddings

Context-aware retrieval

Re-ranking using cross-encoders

❓ Ask Anything (RAG-Based Q&A)

Ask natural language questions

Answers grounded in uploaded documents

Multi-document contextual reasoning

🕘 Upload History & Document Tracking

User-specific document history

Quick access to summaries

Secure document isolation

📊 User Analytics

Estimated time saved

Documents processed per user

Stored in SQLite for persistence

🔐 Authentication-Ready Interface

Firebase Google Authentication

Profile management

Secure access control

🧠 System Architecture
Frontend (HTML + Tailwind + JS)
        |
        v
FastAPI Backend
        |
        ├── OCR & Text Extraction
        ├── AI Summarization (Gemini)
        ├── Semantic Chunking
        ├── Vector Embeddings
        ├── Pinecone Vector DB
        ├── RAG Answer Generation
        |
        v
SQLite Database (PrimaryDB)

🛠️ Tech Stack
🔹 Frontend

HTML5

Tailwind CSS

JavaScript

Firebase Authentication

Responsive, glass-morphism UI

Animated video backgrounds

🔹 Backend

FastAPI

Python 3.10+

SQLite (PrimaryDB)

REST APIs

🔹 AI & ML

Google Gemini API

Sentence Transformers

Cross-Encoder Re-Ranker

Pinecone Vector Database

Retrieval-Augmented Generation (RAG)

🔹 OCR & Parsing

PyMuPDF

EasyOCR

NLTK

📁 Project Structure
📦 ContextIQ
 ┣ 📁 Backend
 ┃ ┣ 📄 main.py               # FastAPI routes
 ┃ ┣ 📄 final_rag.py          # RAG pipeline
 ┃ ┣ 📄 FileHandling.py       # OCR, parsing, DB ops
 ┃ ┣ 📄 prompts.py            # Prompt engineering
 ┃ ┣ 📄 API_key.env           # API keys
 ┃ ┗ 📁 Database
 ┃    ┗ 📄 PrimaryDB.db
 ┣ 📁 Frontend
 ┃ ┣ 📄 index.html            # Upload UI
 ┃ ┣ 📄 summary.html          # AI summary view
 ┃ ┣ 📄 history.html          # Upload history
 ┃ ┣ 📄 full-search.html      # Semantic search
 ┃ ┣ 📄 profile.html          # User profile
 ┃ ┗ 📁 assets
 ┣ 📄 README.md

⚙️ Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/your-repo/contextiq.git
cd contextiq

2️⃣ Backend Setup
pip install -r requirements.txt


Create API_key.env:

GEMINI_API_KEY=your_gemini_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=your_index_name

3️⃣ Run Backend
uvicorn main:app --reload

4️⃣ Run Frontend

Open index.html using Live Server or any static server.

🔌 API Endpoints Overview
Endpoint	Method	Description
/preprocess	POST	Upload & process document
/document/{doc_id}	GET	Fetch document summary
/download/{doc_id}	GET	Download original file
/ask	GET	RAG-based Q&A
/history	GET	User document history
/analysis	GET	User analytics
🏆 Why ContextIQ Stands Out

✅ True semantic understanding
✅ Handles scanned & image documents
✅ Production-grade UI & backend
✅ Modular, scalable architecture
✅ Proper RAG pipeline design
✅ Built for real-world use cases

🚀 Future Scope

Multi-language OCR & summarization

Collaborative workspaces

Voice-based Q&A

Document comparison engine

Cloud storage integration

Role-based access control

👥 Team

Built during GDG TechSprint Hackathon
by passionate developers pushing the boundaries of AI-powered knowledge systems.

🏁 Final Words

ContextIQ doesn’t just read documents —
it understands them, connects them, and reasons over them.

⭐ If you like this project, don’t forget to star the repository!