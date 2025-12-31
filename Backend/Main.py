import os
import shutil
import sqlite3
import pymupdf 
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse 
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import provided backend components
from FileHandling import FileHandler
from final_rag import FullRAGSystem

app = FastAPI(title="GDG-25 AI Document Assistant")

# --- CONFIGURATION ---
STORAGE_DIR = "document_storage"
# FileHandling.py creates the DB at Database/PrimaryDB.db. We must match that.
DB_PATH = "Database/PrimaryDB.db" 
os.makedirs(STORAGE_DIR, exist_ok=True)

# Initialize RAG System (Global is fine as it manages its own connections)
rag_system = FullRAGSystem()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- HELPER TO FIND FILE ---
def find_physical_file(doc_id: str):
    """Searches the storage directory for any file starting with the doc_id."""
    for filename in os.listdir(STORAGE_DIR):
        if filename.startswith(doc_id):
            return os.path.join(STORAGE_DIR, filename)
    return None

# --- API ENDPOINTS ---

@app.post("/preprocess")
async def preprocess_and_upload(user_id: str = Form("default_user"), file: UploadFile = File(...)):
    os.makedirs(STORAGE_DIR, exist_ok=True)
    
    # Instantiate FileHandler per request to avoid state collision (self.text)
    # FileHandling.py stores state in the instance, so global instance is not thread-safe.
    request_file_handler = FileHandler() 

    # Save with temporary name to process
    temp_path = os.path.join(STORAGE_DIR, file.filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 1. Process via FileHandler (Extracts text and saves to DB)
        # addFiles returns a dict: {doc_id: {metadata...}}
        extracted_data = request_file_handler.addFiles(temp_path, user_id)
        
        # Get the new Doc ID
        doc_id = list(extracted_data.keys())[0]
        
        # Get text from the handler instance
        raw_text = request_file_handler.text

        # 2. Rename physical file to doc_id (keeping extension)
        ext = os.path.splitext(file.filename)[1]
        final_path = os.path.join(STORAGE_DIR, f"{doc_id}{ext}")
        
        # Handle overwrite if exists (rare due to UUID)
        if os.path.exists(final_path):
            os.remove(final_path)
        os.rename(temp_path, final_path)

        # 3. Ingest into RAG
        rag_system.ingest_document(raw_text, doc_id)
        
        return {"doc_id": doc_id, "filename": file.filename, "status": "success"}
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        print(f"Error in preprocess: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document/{doc_id}")
async def get_summary(doc_id: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # Schema: doc_id, user_id, filename, date, summary
        cursor.execute("SELECT doc_id, filename, date, summary FROM primarydb WHERE doc_id = ?", (doc_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Document not found")

        # Calculate metadata from physical file
        file_path = find_physical_file(doc_id)
        file_type = "Unknown"
        file_size = "—"
        
        if file_path and os.path.exists(file_path):
            ext = os.path.splitext(file_path)[1].upper().replace(".", "")
            file_type = ext
            
            # Use pages for PDF, size for others
            if ext == "PDF":
                try:
                    with pymupdf.open(file_path) as doc:
                        file_size = f"{len(doc)} pages"
                except:
                    file_size = "Unknown"
            else:
                size_kb = os.path.getsize(file_path) / 1024
                file_size = f"{size_kb:.1f} KB"

        return {
            "doc_id": row[0],
            "filename": row[1],
            "date": row[2],
            "summary": row[3],
            "filetype": file_type,
            "pages": file_size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{doc_id}")
async def download_file(doc_id: str):
    """Endpoint to download the file using its doc_id."""
    file_path = find_physical_file(doc_id)
    
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on server")
    
    # Get original filename from DB to restore it for the user
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT filename FROM primarydb WHERE doc_id = ?", (doc_id,))
        row = cursor.fetchone()
        conn.close()
        
        original_name = row[0] if row else "document"
        
        return FileResponse(
            path=file_path, 
            filename=original_name, 
            media_type='application/octet-stream'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ask")
async def ask_question(query: str, doc_ids: Optional[str] = None):
    """Used by summary.html for document-specific Q&A."""
    try:
        target_ids = []
        if not doc_ids:
            # Fallback to latest if no ID passed using SQLite rowid for ordering
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            # Changed 'ORDER BY id' to 'ORDER BY rowid' as 'id' column does not exist
            cursor.execute("SELECT doc_id FROM primarydb ORDER BY rowid DESC LIMIT 1")
            row = cursor.fetchone()
            conn.close()
            target_ids = [row[0]] if row else []
        else:
            # Split the comma-separated string from the URL
            target_ids = doc_ids.split(",")

        # Call RAG search with the specific allowed doc IDs
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
        # FIX: The table 'primarydb' in FileHandling.py does not have an 'id' column.
        # We must use 'rowid' or sort by 'date'/'doc_id'. Using rowid for insertion order.
        cursor.execute("SELECT doc_id, filename, date, summary FROM primarydb ORDER BY rowid DESC")
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
        
        # Get total documents
        cursor.execute("SELECT COUNT(*) FROM primarydb")
        total_docs = cursor.fetchone()[0]
        
        # Get actual time saved from user_analytics table created in FileHandling.py
        # Summing up stats for all users (global view)
        cursor.execute("SELECT SUM(total_time_saved_minutes) FROM user_analytics")
        row = cursor.fetchone()
        total_time = row[0] if row and row[0] is not None else 0
        
        conn.close()
        
        return {"total_documents": total_docs, "time_saved": round(total_time, 2)}
    except Exception as e:
        # Fallback if table doesn't exist yet
        return {"total_documents": 0, "time_saved": 0}

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
                    "id": all_ids[-1], # Link to the most recent doc for context, or improve UI to link generic
                    "filename": "Cross-Document AI Analysis",
                    "score": 0.95,
                    "snippet": answer
                }
            ]
        }
    except Exception as e:
        print(f"Search Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))