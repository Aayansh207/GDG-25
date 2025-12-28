# =================================================================================================== #
"""It deals with text extraction from uploaded images/documents and the integration primary database"""
# =================================================================================================== #

# Importing required modules #

import pymupdf
import os
import easyocr
import torch
import sqlite3
import re
import prompts

from google import genai
from pathlib import Path
from datetime import date
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer


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

        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

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
        self.connection = sqlite3.connect("PrimaryDB.db")
        self.cursor = self.connection.cursor()
        self.cursor.execute("SELECT COUNT(*) FROM primarydb")

        total_entries = self.cursor.fetchone()[0]

        index = (
            "DOC_" + "0" * (3 - len(str((total_entries + 1)))) + str(total_entries + 1)
        )

        # Generating the summary
        self.summary = self.compareAndSummarize(
            docs_summaries_compare=None, doc_text=self.text
        )

        # Inserting the content into the database.
        self.cursor.execute(
            f"INSERT INTO primarydb VALUES ('{index}', '{user_id}', '{os.path.basename(file)}', '{date.today()}', '{self.summary}')"
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
        self.cursor.execute(f"SELECT * FROM primarydb WHERE doc_id = {doc_id}")
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
        getFiles: bool = False,
    ):
        """This functions brings together the functionality of both addFiles and getFiles function."""

        if addFiles:
            self.extracted_content = {}
            for file in files:
                self.extracted_content.update(self.addFiles(file))

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
            compare=True,
            summarize=False,
        )

        return self.comparison  # returning the generated comparison

    def compareAndSummarize(
        self,
        docs_summaries_compare: list[str] | None,
        doc_text: str | None,
        summarize: bool = True,
        compare: bool = False,
    ):
        """Behind the scenes logic for summary and comparison generation."""

        if summarize:
            """Summarization logic."""

            chunks = self.semantic_chunk(
                text=doc_text
            )  # The raw text is broken into smaller chunks to prevent token exhaustion in case of bigger documents.
            chunk_summaries = []

            for chunk in chunks:
                """Generating short summary for each indivisual chunk"""

                chunk_summary = self.model.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=(prompts.prompt_chunk_summary + "\n\n" + chunk),
                )
                chunk_summaries.append(
                    chunk_summary.text
                )  # appending all generated summaries for indivisual chunks into one list.

            # generating the main summary
            summary = self.model.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompts.prompt_summary + "\n\n".join(chunk_summaries),
            )
            return summary.text  # returning the summary text

        else:
            """Comparison logic"""

            comparison = self.model.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompts.prompt_comparison
                + "\n\n".join(docs_summaries_compare),
            )
            return comparison.text

    def semantic_chunk(
        self,
        text: str,
        max_tokens: int = 1800,
        similarity_threshold: float = 0.18,
    ) -> list[str]:
        """Function responsible for semantic chunking of the raw extracted text. (Taken from final_rag.py)"""

        sentences = sent_tokenize(text)
        if not sentences:
            return []

        chunks = []
        current_chunk = []
        current_embeddings = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = int(len(sent.split()) * 1.3)  # better token estimate
            sent_emb = torch.tensor(
                self.embed_model.encode(sent, normalize_embeddings=True)
            )

            if not current_chunk:
                current_chunk.append(sent)
                current_embeddings.append(sent_emb)
                current_tokens = sent_tokens
                continue

            chunk_mean = torch.stack(current_embeddings).mean(dim=0)
            sim = torch.nn.functional.cosine_similarity(
                sent_emb, chunk_mean, dim=0
            ).item()

            if sim < similarity_threshold or current_tokens + sent_tokens > max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sent]
                current_embeddings = [sent_emb]
                current_tokens = sent_tokens
            else:
                current_chunk.append(sent)
                current_embeddings.append(sent_emb)
                current_tokens += sent_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks  # returns the generated chunks


# Testing

if __name__ == "__main__":

    handler = FileHandler()
    summary = handler.compareAndSummarize(
        docs_summaries_compare=None, doc_text=handler.extract_PDF_text(filePath="1.pdf")
    )
    print(summary)
