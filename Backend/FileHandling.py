# =================================================================================================== #
"""It deals with text extraction from uploaded images/documents and the integration primary database"""
# =================================================================================================== #

# Importing required modules #

import pymupdf
import os
import easyocr
import sqlite3
import re
import prompts
import uuid

from google import genai
from pathlib import Path
from datetime import date
from dotenv import load_dotenv
from dataclasses import dataclass
from nltk.tokenize import sent_tokenize


# Custom typecasting classes


@dataclass
class UserAnalytics:
    user_id: str
    time_saved_for_this_doc: float


@dataclass
class PrimarydbData:
    index: str
    user_id: str
    filename: str
    summary: str


# Main Classes #


class GenerationModel:
    """Class that handles the initialization of the gemini api client."""

    def __init__(self) -> None:

        env_path = Path(__file__).parent / "API_key.env"
        load_dotenv(env_path)  # adding the .env file to path

        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)


class DatabaseManager:
    """This class handles all the database related operations."""

    def __init__(self) -> None:
        """Initializing the database and creating the tables if not present"""

        with sqlite3.connect("Database/PrimaryDB.db") as connection:
            cursor = connection.cursor()
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS primarydb (
                                doc_id TEXT UNIQUE NOT NULL,
                                user_id TEXT NOT NULL,
                                filename TEXT NOT NULL,
                                date TEXT NOT NULL,
                                summary TEXT NOT NULL
                                )"""
            )

            connection.commit()

            cursor.execute(
                """CREATE TABLE IF NOT EXISTS user_analytics (
                                user_id TEXT UNIQUE,
                                total_time_saved_minutes REAL DEFAULT 0,
                                documents_processed INTEGER DEFAULT 0      
                                )"""
            )

            connection.commit()

    def add_data(
        self,
        user_analytics: UserAnalytics | None = None,
        primarydb: PrimarydbData | None = None,
    ) -> None:
        """Function responsible for the operations related to addition of data into the database"""

        if not user_analytics and not primarydb:
            raise ValueError(
                "Both user_analytics and primarydb parameters can't be None at the same time!"
            )

        with sqlite3.connect("Database/PrimaryDB.db") as connection:
            cursor = connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")

            if user_analytics:

                cursor.execute(
                    """
                INSERT INTO user_analytics (user_id, total_time_saved_minutes, documents_processed)
                VALUES (?, ?, 1)
                ON CONFLICT(user_id) DO UPDATE SET
                    total_time_saved_minutes =
                        total_time_saved_minutes + excluded.total_time_saved_minutes,
                    documents_processed =
                        documents_processed + 1
                """,
                    (
                        user_analytics.user_id,
                        float(user_analytics.time_saved_for_this_doc),
                    ),
                )

            if primarydb:

                # Use original_filename in the INSERT statement
                cursor.execute(
                    """
                    INSERT INTO primarydb (doc_id, user_id, filename, date, summary)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        primarydb.index,
                        primarydb.user_id,
                        primarydb.filename,
                        str(date.today()),
                        primarydb.summary,
                    ),
                )
    def delete_data(self, doc_id: str) -> bool:
        """Removes a document entry from the primary database."""
        try:
            with sqlite3.connect("Database/PrimaryDB.db") as connection:
                cursor = connection.cursor()
                cursor.execute("DELETE FROM primarydb WHERE doc_id = ?", (doc_id,))
                connection.commit()
                return cursor.rowcount > 0
        except Exception as e:
            print(f"Database deletion error: {e}")
            return False
        
    def get_data(self, doc_id: str) -> dict:
        """Function responsible for extracting the data about a file from the database from its doc_id"""

        retrieved_content = {}

        # Initiating database connection
        with sqlite3.connect("Database/PrimaryDB.db") as connection:
            cursor = connection.cursor()

            # retrieving the data
            cursor.execute("SELECT * FROM primarydb WHERE doc_id = ?", (doc_id,))
            retrieved_data = cursor.fetchall()

        if retrieved_data != []:
            retrieved_content.update(
                {
                    doc_id: {
                        "index": retrieved_data[0][0],
                        "user_id": retrieved_data[0][1],
                        "filename": retrieved_data[0][2],
                        "upload_date": retrieved_data[0][3],
                        "summary": retrieved_data[0][4],
                    }
                }
            )

        return retrieved_content  # returning the retrieved content


class FileHandler:
    """The class that handles all the main operations."""

    def __init__(self) -> None:
        """Creating an instance of the GenerationModel class and chunking model."""

        self.model = GenerationModel()
        self.reader = None
        self.database_manager = DatabaseManager()

    def deleteFile(self, doc_id: str, storage_dir: str) -> bool:
        """
        Orchestrates the deletion of physical files and database entries.
        """
        db_success = self.database_manager.delete_data(doc_id)
        file_deleted = False
        if os.path.exists(storage_dir):
            for filename in os.listdir(storage_dir):
                if filename.startswith(doc_id):
                    try:
                        os.remove(os.path.join(storage_dir, filename))
                        file_deleted = True
                        break
                    except OSError as e:
                        print(f"Error deleting physical file: {e}")

        return db_success or file_deleted
    
    def extract_PDF_text(self, filePath: str) -> list:
        """Function responsible for extracting the text from PDF files."""

        extracted_text = []

        with pymupdf.open(filePath) as doc:

            self.file_metadata = doc.metadata  # getting the pdf file metadata
            # reading the file

            for page in doc:
                text = page.get_text()

                # removing excess line breaks
                text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
                text = re.sub(r"\n{2,}", "\n\n", text)
                extracted_text.append(text)

        if extracted_text:

            return (
                extracted_text
            )  # returning the text by removing the trailing white spaces, if any

        else:
            raise ValueError("Unsupported pdf file!")

    def image_OCR(self, imagePath: str) -> list:
        """Function responsible for performing OCR on images to extract text"""

        if not self.reader:
            self.reader = easyocr.Reader(
                ["en"]
            )  # Creating the easyocr reader class instance, if doesn't already exist.

        text = ""
        text = self.reader.readtext(imagePath)
        lines = [entry[1] for entry in text]
        text = ["\n".join(lines)]

        if text:
            return text
        else:
            raise ValueError("Unsupported image file!")

    def addFiles(self, file: str, user_id: str) -> dict:
        """Function handling all the processes followed by the uploading of any new file."""

        # Extracting the file's text
        extracted_content = {}

        if file.lower().endswith(".pdf"):
            self.text = self.extract_PDF_text(file)

        elif file.lower().endswith(".txt"):

            with open(file, "rt", encoding="utf-8", errors="ignore") as f:
                text = f.read()
                sentences = sent_tokenize(text)
            
            self.text = []
            current_page = ""

            for sentence in sentences:
                if current_page and len(current_page.split()) + len(sentence.split()) > 300:
                    self.text.append(current_page.strip())
                    current_page = ""

                if current_page:
                    current_page += " " + sentence
                
                else:
                    current_page = sentence
            
            if current_page.strip():
                self.text.append(current_page)
                current_page = ""


            if not self.text.strip():
                raise ValueError("Unsupported/empty text file!")

        else:
            self.text = self.image_OCR(file)

        # Clean ID generation

        index = f"DOC_{uuid.uuid4().hex[:8]}"

        try:
            self.summary = self.compareAndSummarize(
                docs_summaries_compare=None, doc_text="\n\n".join(self.text)
            )

        except Exception as e:
            raise RuntimeError("Summary generation failed!") from e

        # Getting the estimated time saved
        self.time_saved_for_this_doc = self.estimatedTimeSaved(
            text_words=(len(("\n\n".join(self.text)).split())), summary_words=len(self.summary.split())
        )

        self.database_manager.add_data(
            user_analytics=UserAnalytics(
                user_id=user_id, time_saved_for_this_doc=self.time_saved_for_this_doc
            ),
            primarydb=PrimarydbData(
                index=index,
                user_id=user_id,
                filename=os.path.basename(file),
                summary=self.summary,
            ),
        )

        extracted_content.update(
            {
                index: {
                    "index": index,
                    "filename": os.path.basename(file),
                    "upload_date": str(date.today()),
                    "summary": self.summary,
                }
            }
        )

        return extracted_content  # returning the data for quick retrieval

    def handleFiles(
        self,
        doc_ids: list[str] | None,
        files: list[str] | None,
        user_id: str | None,
        addFiles: bool = True,
    ) -> dict:
        """This functions brings together the functionality of both addFiles and get_data (DatabaseManager class) function."""

        if addFiles:

            if not files or not user_id:
                raise ValueError("files and user_id are required when addFiles=True")

            self.extracted_content = {}
            for file in files:
                self.extracted_content.update(self.addFiles(file=file, user_id=user_id))

            return self.extracted_content

        else:

            if not doc_ids:
                raise ValueError("doc_ids are required when addFiles=False")

            self.retrieved_content = {}
            for doc_id in doc_ids:

                self.retrieved_content.update(self.database_manager.get_data(doc_id))

            return self.retrieved_content

    def compareDocs(self, doc_ids: list[str]) -> str:
        """Function that is responsible for generating the comparison between any two or more documents"""

        docs_summaries = []

        for doc_id in doc_ids:
            retrieved_content = self.database_manager.get_data(doc_id)
            docs_summaries.append((retrieved_content[doc_id])["summary"])

        try:
            self.comparison = self.compareAndSummarize(
                docs_summaries_compare=docs_summaries,
                doc_text=None,
                summarize=False,
            )

        except Exception as e:
            raise RuntimeError("Comparision generation failed!") from e 

        return self.comparison  # returning the generated comparison

    def compareAndSummarize(
        self,
        docs_summaries_compare: list[str] | None,
        doc_text: str | None,
        summarize: bool = True,
    ):
        """Behind the scenes logic for summary and comparison generation."""

        if summarize:
            """Summarization logic."""

            # generating the main summary
            summary = self.model.client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompts.prompt_summary_new + "\n\n" + doc_text,
            )
            return summary.text  # returning the summary text

        else:
            """Comparison logic"""

            comparison = self.model.client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompts.prompt_comparison
                + "\n\n"
                + "\n\n".join(docs_summaries_compare),
            )
            return comparison.text

    def estimatedTimeSaved(
        self,
        text_words: int,
        summary_words: int,
        reading_speed: int = 180,
        summary_read_speed: int = 210,
        semantic_search_time_min: int = 1,
    ):
        """Function responsible for estimating the amount of time saved per document."""

        if text_words <= 800:
            summary_gen_time_min = 0.5
        
        elif text_words <=4000:
            summary_gen_time_min = 1.5
        
        elif text_words <=9000:
            summary_gen_time_min = 2.5

        else:
            summary_gen_time_min = 3.5

        manual_time = text_words / reading_speed

        summary_read_time = (
            summary_words / summary_read_speed
            + summary_gen_time_min
            + semantic_search_time_min
        )

        time_saved = manual_time - summary_read_time
        time_saved = min(time_saved, 0.85*manual_time)

        return max(time_saved, 0)