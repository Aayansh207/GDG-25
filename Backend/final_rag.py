"""
Semantic Chunking + Embedding + Pinecone Upload + Retrieval + Rerank + Gemini Answer
"""
#Google API_KEy and Pinecone API_KEY are required to run this code , they should be stored in .env file
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
    overlap_sentences: int = 1,
    decay: float = 0.7,
) -> list[str]:

        sentences = sent_tokenize(text)
        if not sentences:
            return []

        # ---------- AUTO-TUNE SIMILARITY THRESHOLD ----------
        sent_embeddings = self.embed_model.encode(
            sentences, normalize_embeddings=True
        )
        sims = []
        for i in range(1, len(sent_embeddings)):
            sim = torch.nn.functional.cosine_similarity(
                torch.tensor(sent_embeddings[i]),
                torch.tensor(sent_embeddings[i - 1]),
                dim=0
            ).item()
            sims.append(sim)

        if sims:
            mean_sim = sum(sims) / len(sims)
            std_sim = (sum((s - mean_sim) ** 2 for s in sims) / len(sims)) ** 0.5
            similarity_threshold = max(0.1, min(0.4, mean_sim - 0.5 * std_sim))
        else:
            similarity_threshold = 0.2

        # ---------- SEMANTIC CHUNKING ----------
        chunks = []
        current_chunk = []
        current_tokens = 0
        centroid = None

        for sent, sent_emb in zip(sentences, sent_embeddings):
            sent_tokens = len(sent.split())
            sent_emb = torch.tensor(sent_emb)

            if centroid is None:
                centroid = sent_emb
                current_chunk.append(sent)
                current_tokens = sent_tokens
                continue

            sim = torch.nn.functional.cosine_similarity(
                sent_emb, centroid, dim=0
            ).item()

            if sim < similarity_threshold or current_tokens + sent_tokens > max_tokens:
                chunks.append(" ".join(current_chunk))

                # ---------- SEMANTIC OVERLAP ----------
                overlap = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []

                current_chunk = overlap + [sent]
                current_tokens = sum(len(s.split()) for s in current_chunk)

                # recompute centroid from overlap
                overlap_embs = [
                    torch.tensor(self.embed_model.encode(s, normalize_embeddings=True))
                    for s in current_chunk
                ]
                centroid = torch.stack(overlap_embs).mean(dim=0)
            else:
                current_chunk.append(sent)
                current_tokens += sent_tokens
                centroid = decay * sent_emb + (1 - decay) * centroid

        if current_chunk:
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
        self, query: str, allowed_doc_ids: list[str], k: int = 20
    ) -> list[dict]:

        candidates = {}

        for q in self.expand_query(query):  # query expansion
            q_vec = self.embedding(q)

            res = self.index.query(
                vector=q_vec,
                top_k=k,
                filter={"doc_id": {"$in": allowed_doc_ids}},
                include_metadata=True
            )

            for match in res.matches:
                text = match.metadata.get("text")
                if not text:
                    continue

                # keep best pinecone score if duplicate appears
                if text in candidates:
                    candidates[text]["pinecone_score"] = max(
                        candidates[text]["pinecone_score"],
                        float(match.score)
                    )
                    continue

                candidates[text] = {
                    "text": text,
                    "pinecone_score": float(match.score),
                    "doc_id": match.metadata.get("doc_id"),
                    "chunk_index": match.metadata.get("chunk_index"),
                }

        return list(candidates.values())

    # ===================== RERANK =====================
    """After retrieval, rerank the candidates based on relevance to the query"""
    def rerank_candidates(
        self,
        query: str,
        candidates: list,
        top_n: int = 5,
        alpha: float = 0.7,   # weight for rerank score
        beta: float = 0.3    # weight for pinecone score
    ) -> list:

        if not candidates:
            return []

        # ---------- Cross-encoder reranking ----------
        pairs = [[query, c["text"]] for c in candidates]
        rerank_scores = self.reranker.predict(pairs)

        for c, s in zip(candidates, rerank_scores):
            c["rerank_score"] = float(s)

        # ---------- Normalize Pinecone scores ----------
        pinecone_scores = [c["pinecone_score"] for c in candidates]
        min_p, max_p = min(pinecone_scores), max(pinecone_scores)

        for c in candidates:
            if max_p > min_p:
                c["pinecone_norm"] = (c["pinecone_score"] - min_p) / (max_p - min_p)
            else:
                c["pinecone_norm"] = 0.0

        # ---------- Normalize rerank scores ----------
        rerank_vals = [c["rerank_score"] for c in candidates]
        min_r, max_r = min(rerank_vals), max(rerank_vals)

        for c in candidates:
            if max_r > min_r:
                c["rerank_norm"] = (c["rerank_score"] - min_r) / (max_r - min_r)
            else:
                c["rerank_norm"] = 0.0

        # ---------- Final fused score ----------
        for c in candidates:
            c["final_score"] = (
                alpha * c["rerank_norm"] +
                beta * c["pinecone_norm"]
            )

        # ---------- Sort and return ----------
        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        return candidates[:top_n]

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

{context}

QUESTION:
{query}
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

    # ===================== INGESTION FUNCTION =====================
    """Ingest a document into Pinecone using raw text and document ID"""
    def ingest_document(self, raw_text: str, doc_id: str):
        self.delete_document(doc_id)
        self.upload_raw_text(raw_text, doc_id)

    # ===================== SEARCH FUNCTION =====================
    """Search over allowed document IDs and generate an answer"""
    def search(self, query: str, allowed_doc_ids: list[str]) -> str:

        candidates = self.retrieve_candidates_from_pinecone(
            query=query,
            allowed_doc_ids=allowed_doc_ids,
            k=20
        )

        if not candidates:
            return "No data found for the specified documents."

        top_chunks = self.rerank_candidates(
            query=query,
            candidates=candidates,
            top_n=5
        )

        return self.generate_answer(query, top_chunks)


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
        rag.ingest_document(text, doc_id)

    # Now ONLY query
    queries = [
        ("ai_doc", "What is Artificial Intelligence?"),
        ("bio_doc", "What is photosynthesis?"),
        ("history_doc", "When did the French Revolution begin?")
    ]

    for doc_id, q in queries:
        ans = rag.search(
            query=q,
            allowed_doc_ids=[doc_id]
        )
        print(f"\n[{doc_id}] {q}\n{ans}")
