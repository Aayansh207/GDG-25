import os
import shutil
import sqlite3
from typing import List, Optional
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

# --- API ENDPOINTS ---

# Updated main.py snippet for consistency
@app.post("/preprocess")
async def preprocess_and_upload(user_id: str = Form("default_user"), file: UploadFile = File(...)):
    os.makedirs(STORAGE_DIR, exist_ok=True)
    
    temp_path = os.path.join(STORAGE_DIR, file.filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # file_handler.addFiles now stores the correct original filename in DB
        extracted_data = file_handler.addFiles(temp_path, user_id)
        doc_id = list(extracted_data.keys())[0]
        raw_text = extracted_data[doc_id]["text"]

        # Rename physical file to doc_id to avoid OS-level naming conflicts
        ext = os.path.splitext(file.filename)[1]
        final_path = os.path.join(STORAGE_DIR, f"{doc_id}{ext}")
        
        # Check if file exists before renaming to prevent errors
        if os.path.exists(temp_path):
            os.rename(temp_path, final_path)

        rag_system.ingest_document(raw_text, doc_id)
        
        return {"doc_id": doc_id, "filename": file.filename, "status": "success"}
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/summary")
@app.get("/document/{doc_id}")
async def get_summary(doc_id: Optional[str] = None):
    """Fetches summary for summary.html. If no doc_id, gets the latest entry."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        if doc_id:
            cursor.execute("SELECT doc_id, filename, date, summary FROM primarydb WHERE doc_id = ?", (doc_id,))
        else:
            cursor.execute("SELECT doc_id, filename, date, summary FROM primarydb ORDER BY id DESC LIMIT 1")
        
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Document not found")

        return {
            "doc_id": row[0],
            "filename": row[1],
            "date": row[2],
            "summary": row[3],
            "category": "General" # Placeholder as per frontend expectation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ask")
async def ask_question(query: str, doc_ids: Optional[str] = None):
    """Used by summary.html for document-specific Q&A."""
    try:
        # If no doc_ids provided, find the most recent one
        if not doc_ids:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT doc_id FROM primarydb ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            conn.close()
            target_ids = [row[0]] if row else []
        else:
            target_ids = doc_ids.split(",")

        answer = rag_system.search(query=query, allowed_doc_ids=target_ids)
        return {"query": query, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history():
    """Fetches all documents for history.html."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT doc_id, filename, date, summary FROM primarydb ORDER BY id DESC")
        rows = cursor.fetchall()
        conn.close()
        
        return {row["doc_id"]: dict(row) for row in rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis")
async def get_analysis():
    """Provides stats for index.html dashboard."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM primarydb")
        total = cursor.fetchone()[0]
        conn.close()
        return {"total_documents": total, "time_saved": total * 0.5} # Mock logic
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/search")
async def global_search(query: str, top_k: int = 15):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT doc_id, filename FROM primarydb")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return {"results": []}

        all_ids = [row[0] for row in rows]
        answer = rag_system.search(query=query, allowed_doc_ids=all_ids)
        
        # If no answer is found, return an empty list so the frontend shows "No content found"
        if "No relevant documents found" in answer:
            return {"results": []}

        return {
            "results": [
                {
                    "id": all_ids[-1], # Use the most recent ID for the button link
                    "filename": "Cross-Document AI Analysis",
                    "score": 0.95,
                    "snippet": answer
                }
            ]
        }
    except Exception as e:
        print(f"Search Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))