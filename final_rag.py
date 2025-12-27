"""
Semantic Chunking + Embedding + Pinecone Upload + Retrieval + Rerank + Gemini Answer
"""
#Google API_KEy and Pinecone API_KEY are required to run this code , they should be stored in .env file
#update this file to store the comments of what each function is doing

#ONLY CALL MASTER FUNCTION #ONLY CALL MASTER FUNCTION  #ONLY CALL MASTER FUNCTION  #ONLY CALL MASTER FUNCTION  #ONLY CALL MASTER FUNCTION  #ONLY CALL MASTER FUNCTION #ONLY CALL MASTER FUNCTION 

# ===================== IMPORTS =====================
import os
import nltk
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv
from pinecone import Pinecone
from google import genai

# ===================== LOAD ENV =====================
load_dotenv()

# ===================== NLTK SETUP =====================
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


# ===================== CLASS =====================
class FullRAGSystem: 
    def __init__(self, index_name: str | None = None): # index name is the pinecone index name

        # ---------- Embedding Model ----------
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        # ---------- Cross Encoder ----------
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # ---------- Gemini ----------
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise RuntimeError("GOOGLE_API_KEY missing in .env")

        self.client = genai.Client(api_key=google_api_key)

        # ---------- Pinecone ----------
        pinecone_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_key:
            raise RuntimeError("PINECONE_API_KEY missing in .env")

        self.pc = Pinecone(api_key=pinecone_key)
        index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "test")
        self.index = self.pc.Index(index_name)

    # ===================== QUERY EXPANSION =====================
    def expand_query(self, query: str) -> list[str]: #Expand the query for better retrieval
        return [query]  # placeholder, intentional

    # ===================== SEMANTIC CHUNKING =====================
    """split text into semantically coherent chunks(it doesn't just split by size but also by meaning)"""
    def semantic_chunk( 
        self,
        text: str,
        max_tokens: int = 200,
        similarity_threshold: float = 0.18,
    ) -> list[str]:

        sentences = sent_tokenize(text)
        if not sentences:
            return []

        chunks = []
        current_chunk = []
        current_embeddings = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = len(sent.split())
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

        chunks.append(" ".join(current_chunk))
        return chunks

    # ===================== EMBEDDING =====================
    """Get embedding vector for a given text chunks"""
    def embedding(self, text: str) -> list[float]:
        vec = self.embed_model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        return vec.tolist()

    # ===================== PINECONE UPLOAD =====================
    """Upload raw text to Pinecone after chunking and embedding , chunking and embedding function is called here"""
    def upload_raw_text(self, raw_text: str, doc_id: str):

        chunks = self.semantic_chunk(raw_text)
        vectors = []

        for idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            vectors.append(
                {
                    "id": f"{doc_id}-chunk-{idx}",
                    "values": self.embedding(chunk),
                    "metadata": {
                        "doc_id": doc_id,
                        "chunk_index": idx,
                        "text": chunk,
                    },
                }
            )

        if vectors:
            self.index.upsert(vectors=vectors)

        print(f"[UPLOAD] Uploaded {len(vectors)} chunks for doc_id={doc_id}")

    # ===================== RETRIEVAL =====================
    """Retrieve candidate chunks from Pinecone based on the query"""
    def retrieve_candidates_from_pinecone(
        self, query: str, doc_id: str, k: int = 20, all: bool = False
    ) -> list[dict]:

        candidates = {}

        for q in self.expand_query(query): # expand query function is called here , enhances the query for better retrieval
            q_vec = self.embedding(q) # embedding function is called here to get vector of the query

            if all: #(MAIN)- check for global search across all documents
                res = self.index.query( #query is in-built pinecone function to get relevant chunks from pinecone
                    vector=q_vec,
                    top_k=k,
                    include_metadata=True
                )
            else: #check for document-specific search
                res = self.index.query(
                    vector=q_vec,
                    top_k=k,
                    filter={"doc_id": {"$eq": doc_id}}, #filtering by doc_id
                    include_metadata=True
                )

            for match in res.matches: 
                text = match.metadata.get("text")
                if not text:
                    continue

                candidates[text] = { # store candidate chunk details in a dictionary
                    "text": text,
                    "pinecone_score": float(match.score),
                    "doc_id": match.metadata.get("doc_id"),
                    "chunk_index": match.metadata.get("chunk_index"),
                }

        return list(candidates.values())

    # ===================== RERANK =====================
    """#after retrieval, rerank the candidates based on relevance to the query"""
    def rerank_candidates(
        self, query: str, candidates: list, top_n: int = 5
    ) -> list: 

        if not candidates:
            return []

        pairs = [[query, c["text"]] for c in candidates]
        scores = self.reranker.predict(pairs)

        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_n]

    # ===================== SEARCH =====================
    """main search function that combines retrieval and reranking"""
    def semantic_search_pinecone(self, query: str, doc_id: str, k: int = 20, all: bool = False):
        candidates = self.retrieve_candidates_from_pinecone(query, doc_id, k, all)
        return self.rerank_candidates(query, candidates)

    # ===================== ANSWER GENERATION =====================
    def generate_answer(self, query: str, retrieved_chunks: list) -> str:

        if not retrieved_chunks:
            return "The document does not specify this."

        context = "\n\n".join(c["text"] for c in retrieved_chunks)

        prompt = f"""
You are an assistant. Use ONLY the following CONTEXT (from a document) to answer the QUESTION.

- If the answer is present in the context, answer clearly.
- If the document gives another name / synonym, mention it explicitly.
- If the answer is NOT in the context, say: "The document does not specify this."

CONTEXT:
{context}

QUESTION:
{query}
"""

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            return response.text
        except Exception as e:
            print("[LLM ERROR]", e)
            return "There was an error generating the answer."

    """SOME PINECONE DATABASE MANAGEMENT FUNCTIONS"""
    # ===================== DELETE ALL =====================
    def delete_all(self):
        self.index.delete(delete_all=True)
        print("Deleted ALL vectors from the index")

    # ===================== DELETE DOCUMENT =====================
    def delete_document(self, doc_id): #Run reindex_document to update the index
        self.index.delete(
            filter={"doc_id": {"$eq": doc_id}}
        )
        print(f"Deleted document: {doc_id}")

    # ===================== REINDEX DOCUMENT =====================
    def reindex_document(self, raw_text, doc_id):
        self.delete_document(doc_id)
        self.upload_raw_text(raw_text, doc_id)
        print(f"Reindexed document: {doc_id}")

    # ===================== MASTER FUNCTION =====================
    """if RAW file is empy no need to upload PineconeDB just search and generate answer
    if RAW file is present upload it to PineconeDB and then search and generate answer
    Also check do the user only wants to ingest the document without querying """
    def MasterFunction(self, RAWfile: str | None, query: str | None, doc_id: str, all: bool = False):
        """
        End-to-end RAG pipeline for a single document.
        """

        if RAWfile:
            self.index.delete(
                filter={"doc_id": {"$eq": doc_id}}
            )
            self.upload_raw_text(RAWfile, doc_id=doc_id)

        if not query:
            return "No query provided."

        candidates = self.retrieve_candidates_from_pinecone(
            query=query,
            doc_id=doc_id,
            k=20,
            all=all
        )

        if not candidates:
            if all:
                return "No data found across all documents."
            return f"No data found for document: {doc_id}"

        #  Rerank
        top_chunks = self.rerank_candidates(
            query=query,
            candidates=candidates,
            top_n=5
        )

        # Generate answer
        answer = self.generate_answer(query, top_chunks)
        return answer


# ===================== USAGE =====================
if __name__ == "__main__":

    #NOW TIME FOR IMPLEMENTATION
    rag = FullRAGSystem()
    
    docs = {
        "ai_doc": open("ai.txt").read(),
        "history_doc": open("history.txt").read(),
        "bio_doc": open("bio.txt").read(),
    }

    # Ingest once
    for doc_id, text in docs.items():
        print(
            rag.MasterFunction(
                RAWfile=text,
                query=None,
                doc_id=doc_id #DOC ID IS NECESSARY FOR UPLOADING
            )
        )

    # Now ONLY query
    queries = [
        ("ai_doc", "What is Machine Learning?"),
        ("bio_doc", "What is photosynthesis?"),
        ("history_doc", "When did the French Revolution begin?")
    ]

    for doc_id, q in queries:
        ans = rag.MasterFunction(
            RAWfile=None,
            query=q,
            doc_id=doc_id, #put doc_id anything if all = true , i.e.,  to search globally
            all=False   # default behavior
        )
        print(f"\n[{doc_id}] {q}\n{ans}")
