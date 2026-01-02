<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
<<<<<<< HEAD
=======
  <title>ContextIQ – AI Document Intelligence</title>
>>>>>>> d45a367698d376337058376c33039e92fe80a1b9
</head>

<body>

<h1>🧠 ContextIQ</h1>
<h2>AI-Powered Intelligent Document Intelligence Platform</h2>

<p><strong>GDG TechSprint Hackathon Project | GDG MNNIT</strong></p>

<p>
ContextIQ is an AI-powered document intelligence system that enables users to upload documents,
generate accurate summaries, perform semantic search, and ask natural language questions —
all with deep contextual understanding using Retrieval-Augmented Generation (RAG).
</p>

<hr/>

<h2>🌍 Problem Statement</h2>

<p>
In academic, technical, and professional environments, users frequently work with long,
complex documents such as PDFs, scanned files, reports, and images.
</p>

<ul>
  <li>Manual reading is time-consuming</li>
  <li>Keyword-based search fails to capture meaning</li>
  <li>Scanned documents are difficult to analyze</li>
  <li>No unified system to ask questions across documents</li>
</ul>

<hr/>

<h2>💡 Our Solution — ContextIQ</h2>

<p>
ContextIQ transforms raw documents into an intelligent, searchable knowledge system.
Instead of simply storing files, it understands documents at a semantic level and allows
users to interact with them naturally.
</p>

<p>
The platform is built using a full Retrieval-Augmented Generation (RAG) pipeline
combined with modern AI models.
</p>

<hr/>

<h2>✨ Key Features</h2>

<h3>📤 Intelligent Document Upload</h3>
<ul>
  <li>Supports PDF, TXT, and image files</li>
  <li>OCR-based text extraction using EasyOCR</li>
  <li>Automatic preprocessing and cleanup</li>
</ul>

<h3>🧾 AI-Generated Unified Summaries</h3>
<ul>
  <li>Single coherent summary per document</li>
  <li>Preserves technical details and chronology</li>
  <li>HTML-rendered summaries for clean UI display</li>
</ul>

<h3>🔍 Semantic Search</h3>
<ul>
  <li>Meaning-based search instead of keyword matching</li>
  <li>Sentence-level embeddings</li>
  <li>Cross-encoder reranking for relevance</li>
</ul>

<h3>❓ RAG-Based Question Answering</h3>
<ul>
  <li>Ask questions in natural language</li>
  <li>Context-aware answers grounded in documents</li>
  <li>Supports multiple document contexts</li>
</ul>

<h3>🕘 Document History</h3>
<ul>
  <li>User-specific upload history</li>
  <li>Quick access to past summaries</li>
  <li>Secure document isolation</li>
</ul>

<h3>📊 User Analytics</h3>
<ul>
  <li>Estimated time saved per document</li>
  <li>Total documents processed</li>
  <li>Stored persistently in SQLite</li>
</ul>

<h3>🔐 Authentication-Ready UI</h3>
<ul>
  <li>Google authentication via Firebase</li>
  <li>User profile management</li>
  <li>Secure access control</li>
</ul>

<hr/>

<h2>🧠 System Architecture</h2>

<pre>
Frontend (HTML + Tailwind + JS)
        |
        v
FastAPI Backend
        |
        ├── OCR & Text Extraction
        ├── AI Summarization (Gemini)
        ├── Semantic Chunking
        ├── Vector Embeddings
        ├── Pinecone Vector Database
        ├── RAG Answer Generation
        |
        v
SQLite Primary Database
</pre>

<hr/>

<h2>🛠️ Tech Stack</h2>

<h3>Frontend</h3>
<ul>
  <li>HTML5</li>
  <li>Tailwind CSS</li>
  <li>JavaScript</li>
  <li>Firebase Authentication</li>
</ul>

<h3>Backend</h3>
<ul>
  <li>FastAPI</li>
  <li>Python 3.10+</li>
  <li>SQLite</li>
</ul>

<h3>AI & Machine Learning</h3>
<ul>
  <li>Google Gemini API</li>
  <li>Sentence Transformers</li>
  <li>Cross-Encoder Re-Ranker</li>
  <li>Pinecone Vector Database</li>
  <li>Retrieval-Augmented Generation (RAG)</li>
</ul>

<h3>OCR & Parsing</h3>
<ul>
  <li>PyMuPDF</li>
  <li>EasyOCR</li>
  <li>NLTK</li>
</ul>

<hr/>

<h2>📁 Project Structure</h2>

<pre>
ContextIQ/
 ┣ Backend/
 ┃ ┣ main.py
 ┃ ┣ final_rag.py
 ┃ ┣ FileHandling.py
 ┃ ┣ prompts.py
 ┃ ┣ API_key.env
 ┃ ┗ Database/
 ┃   ┗ PrimaryDB.db
 ┣ Frontend/
 ┃ ┣ index.html
 ┃ ┣ summary.html
 ┃ ┣ history.html
 ┃ ┣ full-search.html
 ┃ ┣ profile.html
 ┃ ┗ assets/
 ┣ README.html
</pre>

<hr/>

<h2>⚙️ Installation & Setup</h2>

<h3>1. Clone Repository</h3>
<pre>
<<<<<<< HEAD
git clone https://github.com/Aayansh207/GDG-25.git
=======
git clone https://github.com/your-repo/contextiq.git
>>>>>>> d45a367698d376337058376c33039e92fe80a1b9
cd contextiq
</pre>

<h3>2. Install Dependencies</h3>
<pre>
pip install -r requirements.txt
</pre>

<h3>3. Configure Environment</h3>
<pre>
GEMINI_API_KEY=your_gemini_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=your_index_name
</pre>

<h3>4. Run Backend</h3>
<pre>
uvicorn main:app --reload
</pre>

<h3>5. Run Frontend</h3>
<p>
Open <strong>index.html</strong> using Live Server or any static web server.
</p>

<hr/>

<h2>🔌 API Endpoints</h2>

<table border="1" cellpadding="6">
  <tr>
    <th>Endpoint</th>
    <th>Method</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>/preprocess</td>
    <td>POST</td>
    <td>Upload and process document</td>
  </tr>
  <tr>
    <td>/document/{doc_id}</td>
    <td>GET</td>
    <td>Fetch document summary</td>
  </tr>
  <tr>
    <td>/download/{doc_id}</td>
    <td>GET</td>
    <td>Download original document</td>
  </tr>
  <tr>
    <td>/ask</td>
    <td>GET</td>
    <td>RAG-based question answering</td>
  </tr>
  <tr>
    <td>/history</td>
    <td>GET</td>
    <td>User document history</td>
  </tr>
  <tr>
    <td>/analysis</td>
    <td>GET</td>
    <td>User analytics</td>
  </tr>
</table>

<hr/>

<h2>🏆 Why ContextIQ?</h2>

<ul>
  <li>True semantic understanding</li>
  <li>Handles scanned and image documents</li>
  <li>Clean, production-grade architecture</li>
  <li>Proper RAG pipeline implementation</li>
  <li>Built for real-world scalability</li>
</ul>

<hr/>

<h2>🚀 Future Enhancements</h2>

<ul>
  <li>Multi-language OCR and summarization</li>
  <li>Collaborative document spaces</li>
  <li>Voice-based question answering</li>
  <li>Document comparison engine</li>
  <li>Cloud storage integration</li>
</ul>

<hr/>

<h2>👥 Team</h2>

<p>
Built during the <strong>GDG TechSprint Hackathon</strong> by a team passionate about
AI-powered knowledge systems.
</p>

<hr/>

<h2>🏁 Final Note</h2>

<p>
<strong>ContextIQ</strong> does not just read documents —
it understands, connects, and reasons over them.
</p>

<p>⭐ If you like this project, consider starring the repository.</p>

</body>
</html>
