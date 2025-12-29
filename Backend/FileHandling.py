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

from google import genai
from pathlib import Path
from datetime import date
from dotenv import load_dotenv

# Main Classes #


class GenerationModel:
    """Class that handles the initialization of the gemini api client."""

    def __init__(self) -> None:

        env_path = Path(__file__).parent / "API_key.env"
        load_dotenv(env_path)  # adding the .env file to path

        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)


class FileHandler:
    """The class that handles all the main operations."""

    def __init__(self) -> None:
        """Setting up the database and creating an instance of the GenerationModel class and chunking model."""

        self.model = GenerationModel()

        self.connection = sqlite3.connect("Database/PrimaryDB.db")
        self.cursor = self.connection.cursor()
        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS primarydb (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            doc_id TEXT UNIQUE NOT NULL,
                            user_id TEXT NOT NULL,
                            filename TEXT NOT NULL,
                            date TEXT NOT NULL,
                            summary TEXT NOT NULL
                            )"""
        )
        self.connection.commit()
        self.connection.close()

    def extract_PDF_text(self, filePath: str) -> str:
        """Function responsible for extracting the text from PDF files."""

        with pymupdf.open(filePath) as doc:

            self.file_metadata = doc.metadata  # getting the pdf file metadata
            text = chr(12).join([page.get_text() for page in doc])  # reading the file

        # removing excess line breaks

        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        text = re.sub(r"\n{2,}", "\n\n", text)

        return (
            text.strip()
        )  # returning the text by removing the trailing white spaces, if any

    def image_OCR(self, imagePath: str) -> str:
        """Function responsible for performing OCR on images to extract text"""

        reader = easyocr.Reader(["en"])  # Creating the easyocr reader class instance.

        text = ""
        text = reader.readtext(imagePath)
        lines = [entry[1] for entry in text]
        text += "\n".join(lines)

        return text

    def addFiles(self, file: str, user_id: str) -> dict:
        """Function handling all the processes followed by the uploading of any new file."""
        original_filename = os.path.basename(file)
        # Extracting the file's text
        extracted_content = {}

        if file.lower().endswith(".pdf"):
            self.text = self.extract_PDF_text(file)

        elif file.lower().endswith(".txt"):

            with open(file, "rt", encoding="utf-8", errors="ignore") as f:
                self.text = f.read()

        else:
            self.text = self.image_OCR(file)

        # Initiating database connection
        self.connection = sqlite3.connect("Database/PrimaryDB.db")
        self.cursor = self.connection.cursor()
        self.cursor.execute("SELECT COUNT(*) FROM primarydb")
        total_entries = self.cursor.fetchone()[0]

        # Clean ID generation
        index = f"DOC_{total_entries + 1:03d}" 

        self.summary = self.compareAndSummarize(docs_summaries_compare=None, doc_text=self.text)

        # Use original_filename in the INSERT statement
        self.cursor.execute(
            "INSERT INTO primarydb (doc_id, user_id, filename, date, summary) VALUES (?, ?, ?, ?, ?)",
            (index, user_id, original_filename, str(date.today()), self.summary)
        )

        self.connection.commit()
        self.connection.close()

        extracted_content.update(
            {
                index: {
                    "index": index,
                    "filename": os.path.basename(file),
                    "upload_date": date.today(),
                    "text": self.text,
                    "summary": self.summary,
                }
            }
        )

        return extracted_content  # returning the data for quick retrieval

    def getFiles(self, doc_id: str) -> dict:
        """Function responsible for extracting the data about a file from the database from its doc_id"""

        retrieved_content = {}

        # Initiating database connection
        self.connection = sqlite3.connect("Database/PrimaryDB.db")
        self.cursor = self.connection.cursor()

        # retrieving the data
        self.cursor.execute("SELECT * FROM primarydb WHERE doc_id = ?", (doc_id,))
        retrieved_data = self.cursor.fetchall()
        self.connection.close()

        if retrieved_data != []:
            retrieved_content.update(
                {
                    doc_id: {
                        "index": retrieved_data[1],
                        "user_id": retrieved_data[2],
                        "filename": retrieved_data[3],
                        "upload_date": retrieved_data[4],
                        "summary": retrieved_data[5],
                    }
                }
            )

        return retrieved_content  # returning the retrieved content

    def handleFiles(
        self,
        doc_ids: list[str] | None,
        files: list[str] | None,
        user_id: str | None,
        addFiles: bool = True,
    ):
        """This functions brings together the functionality of both addFiles and getFiles function."""

        if addFiles:
            self.extracted_content = {}
            for file in files:
                self.extracted_content.update(self.addFiles(file=file, user_id=user_id))

            return self.extracted_content

        self.retrieved_content = {}
        for doc_id in doc_ids:

            self.retrieved_content.update(self.getFiles(doc_id))

        return self.retrieved_content

    def compareDocs(self, doc_ids: list[str]) -> str:
        """Function that is responsible for generating the comparison between any two or more documents"""

        docs_summaries = []

        for doc_id in doc_ids:
            retrieved_content = self.getFiles(doc_id)
            docs_summaries.append((retrieved_content[doc_id])["summary"])

        self.comparison = self.compareAndSummarize(
            docs_summaries_compare=docs_summaries,
            doc_text=None,
            summarize=False,
        )

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
                + "\n\n".join(docs_summaries_compare),
            )
            return comparison.text


# Testing

if __name__ == "__main__":

    import time

    start1 = time.time()
    handler = FileHandler()
    end1 = time.time()
    print("elapsed 1:", end1 - start1)
    summary = handler.compareAndSummarize(
        docs_summaries_compare=None, doc_text=handler.extract_PDF_text(filePath="")
    )
    print(summary)
    ed2 = time.time()
    print("final elapsed ", ed2 - start1)
