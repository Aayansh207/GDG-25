import os
from typing import List
from dotenv import load_dotenv

from pinecone import Pinecone

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_experimental.text_splitter import SemanticChunker


load_dotenv()


class FullRAGSystem:
    def __init__(self, index_name: str | None = None):

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        pinecone_key = os.getenv("PINECONE_API_KEY")
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME")
        self.pc = Pinecone(api_key=pinecone_key)
        self.index = self.pc.Index(self.index_name)

        self.vectorstore = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings,
            text_key="text"
        )

        google_key = os.getenv("GOOGLE_API_KEY")
        if not google_key:
            raise RuntimeError("GOOGLE_API_KEY missing")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=google_key,
            temperature=0
        )

        # ---------- Prompt ----------
        self.answer_prompt = PromptTemplate.from_template("""
You must answer the question using ONLY the context below.
If the answer is not present, say "Not found in document".

Return valid HTML only.

Question:
{question}

Context:
{context}
""")

        self.answer_chain = (
            self.answer_prompt
            | self.llm
            | StrOutputParser()
        )

        # ---------- Retriever (semantic-only) ----------
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 10}
        )

    # ======================================================
    # INGESTION
    # ======================================================
    def ingest_document(self, pages: List[str], doc_id: str) -> None:

        if not pages:
            raise ValueError("No pages provided for ingestion")

        semantic_splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95
        )

        documents: List[Document] = []

        for page_no, page_text in enumerate(pages):
            if not page_text.strip():
                continue

            chunks = semantic_splitter.split_text(page_text)

            for chunk in chunks:
                if not chunk.strip():
                    continue

                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "doc_id": doc_id,
                            "page_no": page_no + 1
                        }
                    )
                )

        if not documents:
            raise RuntimeError("No valid chunks generated")

        # Remove old chunks (idempotent ingestion)
        try:
            self.vectorstore.delete(filter={"doc_id": doc_id})
        except Exception:
            pass

        self.vectorstore.add_documents(documents)

    # ======================================================
    # SEARCH (SEMANTIC ONLY)
    # ======================================================
    def search(
        self,
        query: str,
        allowed_doc_ids: List[str]
    ) -> str:

        if not query.strip():
            raise ValueError("Empty query")

        docs = self.vectorstore.similarity_search(
            query,
            k=10,
            filter={"doc_id": {"$in": allowed_doc_ids}}
        )

        if not docs:
            return "No relevant context found."

        context = "\n---\n".join(d.page_content for d in docs)

        return self.answer_chain.invoke({
            "question": query,
            "context": context
        })


import os
import json
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai


# ===================== ENV =====================
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY missing")

genai.configure(api_key=GOOGLE_API_KEY)


# =========================================================
# QUERY ROUTER + EXECUTION ENGINE
# =========================================================
class QueryEngine:
    def __init__(self, semantic_rag, window_size: int = 5):
        self.semantic_rag = semantic_rag
        self.llm = genai.GenerativeModel("gemini-2.5-flash")
        self.memory = WindowMemory(window_size)

    # ======================================================
    # QUERY CLASSIFICATION + EXPANSION
    # ======================================================
    def _route(self, user_query: str) -> List[dict]:
        memory_block = self.memory.render()

        prompt = f""" #Change the prompt later on
Conversation so far:
{memory_block}

You are a QUERY ROUTER AND QUERY EXPANSION ENGINE for a RAG system.
...
USER INPUT:
{user_query}
"""
        response = self.llm.generate_content(prompt)
        raw = response.text.strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON from Gemini:\n{raw}")

        return parsed["queries"]

    # ======================================================
    # SEMANTIC
    # ======================================================
    def query(
        self,
        query: str,
        allowed_doc_ids: List[str],
        memory: str = ""
    ) -> str:

        docs = self.vectorstore.similarity_search(
            query,
            k=10,
            filter={"doc_id": {"$in": allowed_doc_ids}}
        )

        if not docs:
            return "<p>No relevant context found.</p>"

        context = "\n---\n".join(d.page_content for d in docs)

        prompt = f"""
    Conversation so far (for reference only):
    {memory}

    You must answer the CURRENT question using ONLY the document context below.
    Do NOT use the conversation history as a source of facts.

    Current Question:
    {query}

    Document Context:
    {context}

    Return valid HTML only.
    """

        return self.llm.generate_content(prompt).text

    # ======================================================
    # TRANSFORM
    # ======================================================
    def _transform(self, q, doc_ids):
        context = self.semantic_rag.query(
            q["text"],
            allowed_doc_ids=doc_ids
        )

        memory_block = self.memory.render()

        prompt = f"""
Conversation so far:
{memory_block}

Apply the following transformation.

Instruction:
{q["text"]}

Content:
{context}

Return valid HTML only.
"""

        return self.llm.generate_content(prompt).text

    # ======================================================
    # LEXICAL (keyword scan)
    # ======================================================
    def _lexical(self, q, all_chunks):
        keyword = q["expanded_text"].lower()
        hits = []

        for c in all_chunks:
            if keyword in c.page_content.lower():
                hits.append(
                    f"[Doc: {c.metadata['doc_id']} | Page: {c.metadata['page_no']}]\n"
                    f"{c.page_content}"
                )

        if not hits:
            return "No lexical matches found."

        return "<br/><br/>".join(hits[:10])

    # ======================================================
    # STRUCTURAL (locations only)
    # ======================================================
    def _structural(self, q, all_chunks):
        keyword = q["text"].lower()
        locations = []

        for c in all_chunks:
            if keyword in c.page_content.lower():
                locations.append({
                    "doc_id": c.metadata["doc_id"],
                    "page_no": c.metadata["page_no"]
                })

        if not locations:
            return "Term not found in selected documents."

        return json.dumps(locations)

    # ======================================================
    # METADATA (document stats)
    # ======================================================
    def _metadata(self, all_chunks):
        pages_per_doc = {}

        for c in all_chunks:
            doc = c.metadata["doc_id"]
            pages_per_doc.setdefault(doc, set()).add(c.metadata["page_no"])

        return json.dumps({
            doc: len(pages)
            for doc, pages in pages_per_doc.items()
        })

    # ======================================================
    # EXECUTION ORCHESTRATOR
    # ======================================================
    def run(self, user_query, allowed_doc_ids, all_chunks):

    # store user turn
        self.memory.add_user(user_query)

        plans = self._route(user_query)
        answers = []

        for q in plans:
            intent = q["intent"]

            if intent == "SEMANTIC":
                answers.append(self._semantic(q, allowed_doc_ids))

            elif intent == "TRANSFORM":
                answers.append(self._transform(q, allowed_doc_ids))

            elif intent == "LEXICAL":
                answers.append(self._lexical(q, all_chunks))

            elif intent == "STRUCTURAL":
                answers.append(self._structural(q, all_chunks))

            elif intent == "METADATA":
                answers.append(self._metadata(all_chunks))

        final_answer = "\n<hr/>\n".join(map(str, answers))

        self.memory.add_assistant(final_answer)

        return final_answer

class WindowMemory:
    def __init__(self, window_size: int = 5):
        """
        window_size = number of (user, assistant) turns to keep
        """
        self.window_size = window_size
        self.messages = []  # list of dicts: {"role": "...", "content": "..."}

    def add_user(self, text: str):
        self.messages.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str):
        self.messages.append({"role": "assistant", "content": text})
        self._trim()

    def _trim(self):
        # keep only last 2 * window_size messages
        max_len = self.window_size * 2
        if len(self.messages) > max_len:
            self.messages = self.messages[-max_len:]

    def render(self) -> str:
        """
        Convert memory to plain text for Gemini prompt
        """
        if not self.messages:
            return ""

        lines = []
        for m in self.messages:
            prefix = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {m['content']}")

        return "\n".join(lines)

#Testing
if __name__ == "__main__":

    semantic_rag = SemanticRAGSystem()

    # 2 Initialize query engine
    query_engine = QueryEngine(semantic_rag)

    # 3 Load or define ALL chunks (needed for lexical/structural/metadata)
    # IMPORTANT: these must be langchain Document objects
    from langchain_core.documents import Document

    ALL_CHUNKS = [
        Document(
            page_content="LangChain is a framework for building LLM-powered applications.",
            metadata={"doc_id": "doc_1", "page_no": 1}
        ),
        Document(
            page_content="Pinecone is a vector database used for semantic search.",
            metadata={"doc_id": "doc_1", "page_no": 2}
        ),
    ]

    # 4️⃣ Define allowed documents
    allowed_doc_ids = ["doc_1"]

    # 5️⃣ Test query
    user_query = "Where is LangChain mentioned and summarize it"

    # 6️⃣ Run
    response = query_engine.run(
        user_query=user_query,
        allowed_doc_ids=allowed_doc_ids,
        all_chunks=ALL_CHUNKS
    )

    print("\n===== FINAL ANSWER =====\n")
    print(response)
