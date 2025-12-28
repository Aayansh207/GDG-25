'''It deals with text extraction from uploaded images/documents and the integration primary database'''

import pymupdf
import os
import easyocr
import torch
import sqlite3
import re

from google import genai
from pathlib import Path
from datetime import date
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

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

        self.cursor.execute("""CREATE TABLE IF NOT EXISTS primarydb (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            doc_id TEXT UNIQUE NOT NULL,
                            user_id TEXT NOT NULL,
                            filename TEXT NOT NULL,
                            date TEXT NOT NULL,
                            summary TEXT NOT NULL
                            )""")
        self.connection.commit()
        self.connection.close()

        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")


    def extract_PDF_text(self, filePath:str, pagelevel:bool = False) -> str: # | list:
        
        if not pagelevel:
            with pymupdf.open(filePath) as doc:
                self.file_metadata = doc.metadata
                text = chr(12).join([page.get_text() for page in doc])
            
            text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
            text = re.sub(r"\n{2,}", "\n\n", text)
            
            return text.strip()

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
    

    def addFiles(self, file:str, user_id:str) -> dict:
        
        extracted_content = {}

        if file.lower().endswith(".pdf"):
            self.text = self.extract_PDF_text(file)
        
        elif file.lower().endswith(".txt"):
            
            with open(file, "rt",encoding="utf-8", errors="ignore") as f:
                self.text = f.read()

        else:
            self.text = self.image_OCR(file)

        self.connection = sqlite3.connect("PrimaryDB.db")
        self.cursor = self.connection.cursor()
        self.cursor.execute("SELECT COUNT(*) FROM primarydb")

        total_entries = self.cursor.fetchone()[0]
        index = "DOC_" + "0"*(3-len(str((total_entries+1)))) + str(total_entries+1)
        
        self.summary = self.compareAndSummarize(docs_summaries_compare=None, doc_text=self.text)

        self.cursor.execute(f"INSERT INTO primarydb VALUES ('{index}', '{user_id}', '{os.path.basename(file)}', '{date.today()}', '{self.summary}')")
        self.connection.commit() 
        self.connection.close()

        extracted_content.update({index : {"index":index, "filename": os.path.basename(file), "upload_date": date.today(), "text" : self.text, "summary":self.summary}})


        return extracted_content


    def getFiles(self, doc_id:str) -> dict:

        retrieved_content = {}


        self.connection = sqlite3.connect("Database/PrimaryDB.db")
        self.cursor = self.connection.cursor()
        self.cursor.execute(f"SELECT * FROM primarydb WHERE doc_id = {doc_id}")
        retrieved_data= self.cursor.fetchall()
        self.connection.close()

        if retrieved_data != []:

            retrieved_content.update({
                doc_id:{
                "index": retrieved_data[1],
                "user_id": retrieved_data[2],
                "filename": retrieved_data[3],
                "upload_date": retrieved_data[4],
                "summary":retrieved_data[5]
            }})
        
        return retrieved_content


    def handleFiles(self, doc_ids: list[str] | None, files:list[str] | None, user_id:str | None, addFiles:bool = True, getFiles:bool = False):  
        
        if addFiles:

            self.extracted_content = {}
            for file in files:
                self.extracted_content.update(self.addFiles(file))
            
            return self.extracted_content
        
        self.retrieved_content = {}
        for doc_id in doc_ids:

            self.retrieved_content.update(self.getFiles(doc_id))
        
        return self.retrieved_content             


    def compareDocs(self, doc_ids:list[str]) ->str:
        
        docs_summaries = []

        for doc_id in doc_ids:
            retrieved_content = self.getFiles(doc_id)
            docs_summaries.append((retrieved_content[doc_id])["summary"])
        
        self.comparison = self.compareAndSummarize(docs_summaries_compare=docs_summaries, doc_text=None,compare=True, summarize=False)
        return self.comparison


    def compareAndSummarize(self, docs_summaries_compare:list[str] | None, doc_text:str | None ,summarize:bool = True, compare:bool = False):

        if summarize:

            chunks = self.semantic_chunk(text=doc_text, max_tokens=1800, similarity_threshold=0.18)
            chunk_summaries = []
            for chunk in chunks:
                prompt_summary_chunk = f"""You are given a chunk of a larger document.

Task:
Summarize this chunk clearly and concisely while preserving:
- Key facts and observations
- Important timelines or sequences
- Notable entities, actions, or decisions
- Any conclusions or implications explicitly stated
- Do not any salutations greetings or anything else in your response

Rules:
- Do NOT add new information or interpretations
- Do NOT repeat sentences verbatim
- Maintain a neutral, factual tone
- Prefer clarity over verbosity

Output format:
- Use short bullet points
- Group related points together
- Keep the summary under 120 words

Document chunk:
{chunk}
""" 
                

                chunk_summary = self.model.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt_summary_chunk
            )
                chunk_summaries.append(chunk_summary.text)
            
            prompt_summary = """You are provided with multiple summaries, each corresponding to different sections of the same source document.

Task
Synthesize these summaries into a single, unified final summary that accurately represents the entire document as a coherent whole.

Strict rules (must follow exactly):

Do NOT include salutations, greetings, or introductions

Do NOT include concluding phrases such as “In conclusion,” “Overall,” or similar

Do NOT use buzzwords, marketing language, or vague wording

Do NOT repeat the same information unless necessary for clarity

Do NOT add interpretations, opinions, assumptions, or external context

Do NOT reference or mention section summaries or the summarization process

Content requirements:

Preserve all key events, timelines, and chronological sequences

Preserve critical observations, technical findings, and analyses

Preserve official assessments, decisions, and recommendations

Maintain a neutral, factual, report-style tone throughout

Formatting requirements (STRICT):

Output MUST be valid HTML only

Use <h3> tags for major themes only if clearly distinguishable

Use <ul> and <li> for structured points

Do NOT use markdown

Do NOT include explanations, comments, or text outside HTML tags

Do NOT wrap output in <html>, <head>, or <body> tags

Do NOT use code blocks

Length constraint:

Target total length: 180–500 words depending on the length and number of the indivisual summaries.

Keep the summary concise and information-dense

Input:
Multiple section summaries from the same document

Output:
A single, unified HTML-formatted final summary following all rules above:

""" + "\n\n".join(chunk_summaries)
            
            summary = self.model.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_summary
        )
            return summary.text

                

        
        else:

            prompt_comparison = """
You are an expert analyst and technical reviewer.

I will provide you with a list of summaries, where each summary represents a different document.
Your task is to compare and analyze these documents based ONLY on the information present in the summaries.

STRICT RULES:
- Use ONLY plain HTML tags in your output (no Markdown, no code blocks)
- Do NOT include <html>, <head>, or <body> tags
- Do NOT use inline styles, CSS, JavaScript, or emojis
- Do NOT hallucinate or assume missing information
- If information is insufficient, explicitly state that
- Treat each document equally
- Do NOT add interpretations or new information.
- Do not add your opinions or salutations in your response.
- Reply to the point and do not add unnecessary buzz words

Your output MUST be directly renderable inside a webpage container.

----------------------------------

STRUCTURE YOUR OUTPUT EXACTLY AS FOLLOWS:

<h2>Document Overview</h2>
<ul>
  <li><strong>Doc 1:</strong> Short 1–2 line description</li>
  <li><strong>Doc 2:</strong> Short 1–2 line description</li>
  <!-- Continue for all documents -->
</ul>

<h2>Core Similarities</h2>
<ul>
  <li>Similarity description (mention which documents share this)</li>
</ul>

<h2>Core Differences</h2>
<ul>
  <li><strong>Focus / Scope:</strong> Explanation</li>
  <li><strong>Approach / Methodology:</strong> Explanation</li>
  <li><strong>Assumptions / Perspective:</strong> Explanation</li>
  <li><strong>Conclusions / Outcomes:</strong> Explanation</li>
</ul>

<h2>Unique Contributions</h2>
<ul>
  <li><strong>Doc 1:</strong> Unique contribution</li>
  <li><strong>Doc 2:</strong> Unique contribution</li>
</ul>

<h2>Contradictions or Tensions</h2>
<p>
State any conflicting viewpoints or explicitly say "No direct contradictions identified."
</p>

<h2>Overall Synthesis</h2>
<p>
Concise synthesis describing how these documents relate to each other as a whole.
</p>

----------------------------------

Here are the document summaries:

                                """ + "\n\n".join(docs_summaries_compare)
            
            comparison = self.model.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_comparison
        )
            return comparison.text
        
    def semantic_chunk(
        self,
        text: str,
        max_tokens: int = 1800,
        similarity_threshold: float = 0.75,
    ) -> list[str]:

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

        return chunks

            

if __name__ == '__main__':

    handler = FileHandler()

    summary = handler.compareAndSummarize(docs_summaries_compare=None, doc_text=handler.extract_PDF_text(filePath="1.pdf"))
    print(summary)