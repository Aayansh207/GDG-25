import os
import shutil
import sqlite3
import pymupdf
import zipfile
from fastapi import BackgroundTasks 
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Body
from fastapi.responses import FileResponse 
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback

# Import provided backend components
from FileHandling import FileHandler
from final_rag import FullRAGSystem

app = FastAPI(title="GDG-25 AI Document Assistant")

# --- CONFIGURATION ---
STORAGE_DIR = "document_storage"
DB_PATH = "Database/PrimaryDB.db" 
os.makedirs(STORAGE_DIR, exist_ok=True)

# Initialize RAG System
rag_system = FullRAGSystem()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class BatchDeleteRequest(BaseModel):
    doc_ids: List[str]
    user_id: str

class BatchDownloadRequest(BaseModel):
    doc_ids: List[str]
    user_id: str

# --- HELPER ---
def find_physical_file(doc_id: str):
    for filename in os.listdir(STORAGE_DIR):
        if filename.startswith(doc_id):
            return os.path.join(STORAGE_DIR, filename)
    return None

def cleanup_file(path: str):
    if os.path.exists(path):
        os.remove(path)

# --- API ENDPOINTS ---

@app.post("/preprocess")
async def preprocess_and_upload(user_id: str = Form(...), file: UploadFile = File(...)):
    """
    Uploads a file for a specific user with enhanced error reporting.
    """
    os.makedirs(STORAGE_DIR, exist_ok=True)
    request_file_handler = FileHandler() 

    temp_path = os.path.join(STORAGE_DIR, file.filename)
    
    # Save temp file
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save temporary file: {str(e)}")
    
    try:
        # Pass user_id to addFiles
        extracted_data = request_file_handler.addFiles(temp_path, user_id)
        doc_id = list(extracted_data.keys())[0]
        raw_text = request_file_handler.text

        ext = os.path.splitext(file.filename)[1]
        final_path = os.path.join(STORAGE_DIR, f"{doc_id}{ext}")
        
        if os.path.exists(final_path):
            os.remove(final_path)
        os.rename(temp_path, final_path)

        rag_system.ingest_document(raw_text, doc_id)
        
        return {"doc_id": doc_id, "filename": file.filename, "status": "success"}

    except Exception as e:
        # Cleanup temp file if it exists
        if os.path.exists(temp_path): 
            os.remove(temp_path)
        
        # 1. Log the full traceback to the server console (Critical for debugging)
        print("!! PREPROCESS ERROR TRACEBACK START !!")
        traceback.print_exc()
        print("!! PREPROCESS ERROR TRACEBACK END !!")

        # 2. Extract specific error details for the client
        error_msg = str(e)
        
        # If the error has a chained cause (e.g. RuntimeError from FileHandling), append it
        if hasattr(e, '__cause__') and e.__cause__:
            error_msg += f" -> Root Cause: {str(e.__cause__)}"

        # 3. Return appropriate status code
        # ValueError usually comes from FileHandling (e.g., "Unsupported pdf file!")
        status_code = 400 if isinstance(e, ValueError) else 500

        raise HTTPException(status_code=status_code, detail=error_msg)

@app.post("/download_multiple")
async def download_multiple_files(payload: BatchDownloadRequest, background_tasks: BackgroundTasks):
    """
    Creates a ZIP file containing the requested documents and returns it.
    """
    user_id = payload.user_id
    doc_ids = payload.doc_ids
    
    if not doc_ids:
        raise HTTPException(status_code=400, detail="No document IDs provided")

    try:
        # 1. Verify ownership and get filenames from DB
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create a placeholder string for the query (e.g., "?,?,?")
        placeholders = ','.join('?' * len(doc_ids))
        query = f"SELECT doc_id, filename, user_id FROM primarydb WHERE doc_id IN ({placeholders})"
        
        cursor.execute(query, doc_ids)
        rows = cursor.fetchall()
        conn.close()

        # Filter valid documents belonging to this user
        valid_docs = [
            {"doc_id": row[0], "filename": row[1]} 
            for row in rows if row[2] == user_id
        ]
        
        if not valid_docs:
            raise HTTPException(status_code=404, detail="No valid documents found for this user.")

        # 2. Create a temporary ZIP file
        zip_filename = f"documents_export_{user_id}.zip"
        zip_path = os.path.join(STORAGE_DIR, zip_filename)

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            files_added = 0
            for doc in valid_docs:
                physical_path = find_physical_file(doc["doc_id"])
                
                if physical_path and os.path.exists(physical_path):
                    # Add file to zip (using original filename)
                    # Handle duplicate filenames in zip by prepending doc_id if needed
                    zipf.write(physical_path, arcname=doc["filename"])
                    files_added += 1
        
        if files_added == 0:
            raise HTTPException(status_code=404, detail="Physical files not found on server.")

        # 3. Schedule cleanup of the zip file after sending
        background_tasks.add_task(cleanup_file, zip_path)

        # 4. Return the ZIP file
        return FileResponse(
            path=zip_path, 
            filename=zip_filename, 
            media_type='application/zip'
        )

    except Exception as e:
        print(f"Batch Download Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document/{doc_id}")
async def get_summary(doc_id: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT doc_id, filename, date, summary FROM primarydb WHERE doc_id = ?", (doc_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Document not found")

        file_path = find_physical_file(doc_id)
        file_type = "Unknown"
        file_size = "—"
        
        if file_path and os.path.exists(file_path):
            ext = os.path.splitext(file_path)[1].upper().replace(".", "")
            file_type = ext
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
    file_path = find_physical_file(doc_id)
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on server")
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT filename FROM primarydb WHERE doc_id = ?", (doc_id,))
        row = cursor.fetchone()
        conn.close()
        original_name = row[0] if row else "document"
        
        return FileResponse(path=file_path, filename=original_name, media_type='application/octet-stream')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ask")
async def ask_question(query: str, doc_ids: Optional[str] = None):
    try:
        target_ids = []
        if not doc_ids:
            return {"query": query, "answer": "Please select a document context."}
        else:
            target_ids = doc_ids.split(",")

        rag_result = rag_system.search(query=query, allowed_doc_ids=target_ids)
        
        return {
            "query": query, 
            "answer": rag_result["answer"],
            "sources": rag_result.get("sources", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history(user_id: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT doc_id, filename, date, summary FROM primarydb WHERE user_id = ? ORDER BY rowid DESC", (user_id,))
        rows = cursor.fetchall()
        conn.close()
        
        return {row["doc_id"]: dict(row) for row in rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis")
async def get_analysis(user_id: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM primarydb WHERE user_id = ?", (user_id,))
        total_docs = cursor.fetchone()[0]
        
        cursor.execute("SELECT total_time_saved_minutes FROM user_analytics WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        total_time_mins = row[0] if row else 0
        conn.close()
        
        total_time_hrs = total_time_mins / 60
        return {"total_documents": total_docs, "time_saved": round(total_time_hrs, 2)}
    except Exception as e:
        return {"total_documents": 0, "time_saved": 0}

@app.get("/profile-stats")
async def get_profile_stats(user_id: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM primarydb WHERE user_id = ?", (user_id,))
        count = cursor.fetchone()[0]
        cursor.execute("SELECT date FROM primarydb WHERE user_id = ? ORDER BY rowid ASC LIMIT 1", (user_id,))
        row = cursor.fetchone()
        member_since = row[0] if row else "New Member"
        conn.close()
        return {"doc_count": count, "member_since": member_since}
    except Exception as e:
        return {"doc_count": 0, "member_since": "Unknown"}

@app.delete("/document/{doc_id}")
async def delete_document(doc_id: str, user_id: str = Form(...)):
    """
    Deletes a single document.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM primarydb WHERE doc_id = ?", (doc_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if row[0] != user_id:
            raise HTTPException(status_code=403, detail="Unauthorized action")

        rag_system.delete_document(doc_id)
        file_handler = FileHandler()
        success = file_handler.deleteFile(doc_id, STORAGE_DIR)

        if success:
            return {"status": "success", "message": f"Document {doc_id} deleted"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete from storage")

    except Exception as e:
        print(f"Delete Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete_multiple")
async def delete_multiple_documents(payload: BatchDeleteRequest):
    """
    New Endpoint: Deletes multiple documents for a specific user.
    """
    try:
        user_id = payload.user_id
        doc_ids = payload.doc_ids
        
        if not doc_ids:
            return {"status": "success", "message": "No documents to delete"}

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Verify ownership for all docs first
        placeholders = ','.join('?' * len(doc_ids))
        cursor.execute(f"SELECT doc_id, user_id FROM primarydb WHERE doc_id IN ({placeholders})", doc_ids)
        rows = cursor.fetchall()
        conn.close()

        valid_docs = [row[0] for row in rows if row[1] == user_id]
        
        if not valid_docs:
             return {"status": "success", "message": "No valid documents found for deletion"}

        file_handler = FileHandler()
        deleted_count = 0

        for doc_id in valid_docs:
            # 1. Delete from Vector DB
            rag_system.delete_document(doc_id)
            # 2. Delete from Local Storage & SQLite
            if file_handler.deleteFile(doc_id, STORAGE_DIR):
                deleted_count += 1

        return {"status": "success", "message": f"Deleted {deleted_count} documents"}

    except Exception as e:
        print(f"Batch Delete Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def global_search(user_id: str, query: str, top_k: int = 15):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT doc_id, filename FROM primarydb WHERE user_id = ?", (user_id,))
        rows = cursor.fetchall()
        user_docs_map = {row[0]: row[1] for row in rows}
        conn.close()

        if not user_docs_map:
            return {"results": []}

        user_doc_ids = list(user_docs_map.keys())
        rag_output = rag_system.search(query=query, allowed_doc_ids=user_doc_ids)
        
        answer = rag_output["answer"]
        source_chunks = rag_output["sources"]
        
        if "No relevant documents found" in answer:
            return {"results": []}

        unique_sources = {}
        for chunk in source_chunks:
            did = chunk.get("doc_id")
            if did and did not in unique_sources and did in user_docs_map:
                unique_sources[did] = {
                    "doc_id": did,
                    "filename": user_docs_map[did]
                }
        
        final_sources_list = list(unique_sources.values())

        return {
            "results": [
                {
                    "id": "result", 
                    "filename": "Your Documents",
                    "score": 1.0,
                    "snippet": answer,
                    "sources": final_sources_list
                }
            ]
        }
    except Exception as e:
        print(f"Search Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))