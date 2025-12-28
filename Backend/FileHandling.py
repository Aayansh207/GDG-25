'''It deals with text extraction from uploaded images/documents and the integration primary database'''

import pymupdf
import os
import easyocr
import json
import sqlite3

from google import genai
from pathlib import Path
from datetime import date
from dotenv import load_dotenv

class GenerationModel:

    def __init__(self) -> None:

        env_path = Path(__file__).parent / "API_key.env"
        load_dotenv(env_path)

        self.api_key = os.getenv("GEMINI_API_KEY")

        self.client = genai.Client(api_key=self.api_key)


class FileHandler:


    def __init__(self) -> None:

        self.model = GenerationModel()
        self.connection = sqlite3.connect("Database/PrimaryDB.db")
        self.cursor = self.connection.cursor()

        self.cursor.execute("""CREATE TABLE IF NOT EXISTS primary(
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            doc_id TEXT UNIQUE NOT NULL,
                            user_id TEXT NOT NULL,
                            filename TEXT NOT NULL,
                            date TEXT NOT NULL,
                            summary TEXT NOT NULL
                            )""")
        self.connection.commit()
        self.connection.close()



    def extract_PDF_text(self, filePath:str, pagelevel:bool = False) -> str: # | list:
        
        if not pagelevel:
            with pymupdf.open(filePath) as doc:
                self.file_metadata = doc.metadata
                text = chr(12).join([page.get_text() for page in doc])
            
            return text

        # with pymupdf.open(filePath) as doc:
        #     self.file_metadata = doc.metadata
        #     text = []    
        #     pages = doc.page_count    
        #     for i in pages:

        #         page = doc[i-1]
        #         text.append(chr(12).join([page.get_text()]))
        
        # return text
    
    
    def image_OCR(self, imagePath:str) -> str:
        
        reader = easyocr.Reader(['en'])
        text = ""

        # for image in imagePath:
        text = reader.readtext(imagePath)
        lines = [entry[1] for entry in text]

        text += "\n".join(lines)

        return text
    

    def addFiles(self, files:list[str], user_id:str) -> dict:
        
        extracted_content = {}

        for file in files:

            if file.lower().endswith(".pdf"):
                self.text = self.extract_PDF_text(file)
            
            elif file.lower().endswith(".txt"):
                
                with open(file, "rt",encoding="utf-8", errors="ignore") as f:
                    return f.read()

            else:
                self.text = self.image_OCR(file)

            self.connection = sqlite3.connect("PrimaryDB.db")
            self.cursor = self.connection.cursor()
            self.cursor.execute("SELECT COUNT(*) FROM primary")

            total_entries = self.cursor.fetchone()[0]
            index = "DOC_" + "0"*(3-len(str((total_entries+1)))) + str(total_entries+1)

            self.cursor.execute(f"INSERT INTO primary VALUES ('{index}', '{user_id}', '{os.path.basename(file)}', '{date.today()}', '{self.summary}')")
            self.connection.commit() 
            self.connection.close()

            extracted_content.update({index : {"index":index, "text" : self.text, "filename": os.path.basename(file), "summary":self.summary}})


        return extracted_content


    def getFiles(self, doc_ids:list[str]) -> dict:

        retrieved_content = {}

        for doc_id in doc_ids:
            self.connection = sqlite3.connect("Database/PrimaryDB.db")
            self.cursor = self.connection.cursor()
            self.cursor.execute(f"SELECT * FROM primary WHERE doc_id = {doc_id}")
            self.retrieved_content = self.cursor.fetchall()
            self.connection.close()

            if self.retrieved_content != []:

                retrieved_content.update({
                    doc_id:{
                    "index": self.retrieved_content[1],
                    "user_id": self.retrieved_content[2],
                    "filename": self.retrieved_content[3],
                    "upload_date": self.retrieved_content[4],
                    "summary":self.retrieved_content[5]
                }})
        
        return retrieved_content

    def handleFiles(self, addFiles:bool = True, getFiles:bool = False):
        ...
