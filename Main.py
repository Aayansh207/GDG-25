import os
import shutil
import sqlite3
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import provided backend components
from FileHandling import FileHandler
from final_rag import FullRAGSystem

app = FastAPI(title="GDG-25 AI Document Assistant")

# --- CONFIGURATION ---
STORAGE_DIR = "document_storage"
DB_PATH = "Database/PrimaryDB.db"
os.makedirs(STORAGE_DIR, exist_ok=True)

# Initialize global systems
file_handler = FileHandler()
rag_system = FullRAGSystem()

# CORS setup for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- REQUEST MODELS ---
class SearchRequest(BaseModel):
    user_id: str
    query: str
    doc_ids: List[str]

class SummaryRequest(BaseModel):
    user_id: str
    doc_ids: List[str]

# --- API ENDPOINTS ---

@app.post("/preprocess")
async def preprocess_and_upload(
    user_id: str = Form(...), 
    files: List[UploadFile] = File(...)
):
    """
    Handles file upload: extracts text, saves to SQLite, moves original 
    to storage named by doc_id, and indexes in Pinecone.
    """
    results = []
    for file in files:
        # Save initially to a temp path for extraction
        temp_path = os.path.join(STORAGE_DIR, f"temp_{file.filename}")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            # 1. Process via FileHandler (Extraction, DB Entry, and Summary)
            extracted_data = file_handler.addFiles(temp_path, user_id)
            doc_id = list(extracted_data.keys())[0]
            raw_text = extracted_data[doc_id]["text"]

            # 2. Persist Original File (Rename to doc_id)
            extension = os.path.splitext(file.filename)[1]
            permanent_path = os.path.join(STORAGE_DIR, f"{doc_id}{extension}")
            os.rename(temp_path, permanent_path)

            # 3. Ingest into RAG for Semantic Search
            rag_system.ingest_document(raw_text, doc_id)
            
            results.append({"filename": file.filename, "doc_id": doc_id, "status": "success"})
        
        except Exception as e:
            if os.path.exists(temp_path): os.remove(temp_path)
            results.append({"filename": file.filename, "status": "error", "detail": str(e)})

    return {"results": results}

@app.post("/summarize")
async def get_summary_or_comparison(request: SummaryRequest):
    """Fetches a single summary or generates an HTML comparison for multiple docs."""
    try:
        if len(request.doc_ids) == 1:
            data = file_handler.getFiles(request.doc_ids[0])
            if not data: raise HTTPException(status_code=404, detail="Doc not found")
            return {"type": "summary", "content": data[request.doc_ids[0]]["summary"]}
        
        # Compare multiple summaries into one HTML synthesis
        comparison_html = file_handler.compareDocs(request.doc_ids)
        return {"type": "comparison", "content": comparison_html}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ask")
async def ask_question(query: str, user_id: str, doc_ids: str):
    """
    Performs RAG search. doc_ids should be comma-separated in the URL.
    Example: /ask?query=what_is_ai&user_id=123&doc_ids=DOC_001,DOC_002
    """
    try:
        target_ids = doc_ids.split(",")
        # RAG search includes retrieval, reranking, and Gemini generation
        answer = rag_system.search(query=query, allowed_doc_ids=target_ids)
        return {"query": query, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{user_id}")
async def get_user_history(user_id: str):
    """Fetches metadata for all documents uploaded by a specific user."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT doc_id, filename, date, summary FROM primarydb WHERE user_id = ?", (user_id,))
        rows = cursor.fetchall()
        conn.close()
        
        return {
            "documents": [{"doc_id": r[0], "filename": r[1], "date": r[2], "summary": r[3]} for r in rows]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))