import os
import nltk
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from google import genai
from pathlib import Path  # Add this import

# --- FIX: Explicitly load the correct .env file ---
# Get the absolute path to the directory containing this script
base_dir = Path(__file__).parent

# Try loading 'API_key.env' (from FileHandling) OR standard '.env'
# If your file is named 'API_key.env', use that. If it's '.env', use that.
env_path = base_dir / "API_key.env" 

if not env_path.exists():
    env_path = base_dir / ".env" # Fallback to standard .env

load_dotenv(env_path)
# --------------------------------------------------

# ===================== NLTK SETUP =====================
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

class FullRAGSystem: 
    def __init__(self, index_name: str | None = None):
        # 1. Models
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # 2. Gemini Setup
        google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not google_api_key:
            # Debugging help: print where it looked
            print(f"DEBUG: Looking for .env at: {env_path}")
            print(f"DEBUG: File exists? {env_path.exists()}")
            raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) missing in .env")
        
        self.client = genai.Client(api_key=google_api_key)
        self.llm_model_id = "gemini-2.5-flash-lite"
        # 3. Pinecone Setup
        pinecone_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_key:
            raise RuntimeError("PINECONE_API_KEY missing in .env")
        
        self.pc = Pinecone(api_key=pinecone_key)
        index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "test")

        # ✅ Ensure index exists
        existing_indexes = self.pc.list_indexes().names()

        if index_name not in existing_indexes:
            self.pc.create_index(
                name=index_name,
                dimension=384,  # MUST match all-MiniLM-L6-v2
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",      # change if your console shows gcp
                    region="us-east-1"  # MUST match Pinecone console
                )
            )

        self.index = self.pc.Index(index_name)

    def expand_query(self, query: str) -> list[str]:
        return [query]

    def semantic_chunk(self, text: str, max_tokens: int = 200, overlap_sentences: int = 1, decay: float = 0.7) -> list[str]:
        sentences = sent_tokenize(text)
        if not sentences: return []

        sent_embeddings = self.embed_model.encode(sentences, normalize_embeddings=True)
        sims = []
        for i in range(1, len(sent_embeddings)):
            sim = torch.nn.functional.cosine_similarity(
                torch.tensor(sent_embeddings[i]),
                torch.tensor(sent_embeddings[i - 1]),
                dim=0
            ).item()
            sims.append(sim)

        threshold = max(0.1, min(0.4, (sum(sims)/len(sims)) - 0.5 * 0.1)) if sims else 0.2

        chunks, current_chunk, current_tokens, centroid = [], [], 0, None

        for sent, sent_emb in zip(sentences, sent_embeddings):
            sent_tokens = len(sent.split())
            sent_emb = torch.tensor(sent_emb)

            if centroid is None:
                centroid, current_chunk, current_tokens = sent_emb, [sent], sent_tokens
                continue

            sim = torch.nn.functional.cosine_similarity(sent_emb, centroid, dim=0).item()

            if sim < threshold or current_tokens + sent_tokens > max_tokens:
                chunks.append(" ".join(current_chunk))
                overlap = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
                current_chunk = overlap + [sent]
                current_tokens = sum(len(s.split()) for s in current_chunk)
                overlap_embs = [torch.tensor(self.embed_model.encode(s)) for s in current_chunk]
                centroid = torch.stack(overlap_embs).mean(dim=0)
            else:
                current_chunk.append(sent)
                current_tokens += sent_tokens
                centroid = decay * sent_emb + (1 - decay) * centroid

        if current_chunk: chunks.append(" ".join(current_chunk))
        return chunks

    def embedding(self, text: str) -> list[float]:
        return self.embed_model.encode(text, normalize_embeddings=True).tolist()

    def upload_raw_text(self, raw_text: list, doc_id: str):
        
        for page_no, text in enumerate(raw_text):
            chunks = self.semantic_chunk(text)
            vectors = []
            for idx, chunk in enumerate(chunks):
                if not chunk.strip(): continue
                vectors.append({
                    "id": f"{doc_id}-chunk{idx}-page_no{page_no+1}",
                    "values": self.embedding(chunk),
                    "metadata": {"doc_id": doc_id, "text": chunk, "page_no" : page_no+1},
                })
            if vectors:
                # Upsert in batches if vectors are many
                self.index.upsert(vectors=vectors)
            print(f"[UPLOAD] Success: doc_id={doc_id}")

    def retrieve_candidates_from_pinecone(self, query: str, allowed_doc_ids: list[str], k: int = 10) -> list[dict]:
        sources = []
        q_vec = self.embedding(query)
        res = self.index.query(
            vector=q_vec,
            top_k=k,
            filter={"doc_id": {"$in": allowed_doc_ids}},
            include_metadata=True
        )
        
        candidates = []
        for match in res.matches:
            candidates.append({
                "text": match.metadata["text"],
                "pinecone_score": float(match.score),
                "doc_id": match.metadata["doc_id"]
            })
            sources.append({"doc_id":match.metadata["doc_id"], "page_no":match.metadata["page_no"]})
        return candidates

    def rerank_candidates(self, query: str, candidates: list, top_n: int = 5) -> list:
        if not candidates: return []
        pairs = [[query, c["text"]] for c in candidates]
        rerank_scores = self.reranker.predict(pairs)
        
        for c, s in zip(candidates, rerank_scores):
            c["final_score"] = float(s)
            
        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        return candidates[:top_n]

    def generate_answer(self, query: str, retrieved_chunks: list) -> str:
        if not retrieved_chunks: return "No context found."
        context = "\n---\n".join(c["text"] for c in retrieved_chunks)
        
        prompt = f"""
You are an assistant answering questions using retrieved document context.

Strict rules (must follow exactly):

- Use ONLY the information present in the provided context
- Do NOT add external knowledge, assumptions, or interpretations
- If the context does not contain enough information, clearly state that
- Do NOT invent lists, steps, or details that are not explicitly mentioned
- Preserve factual accuracy over fluency

Formatting rules (MANDATORY):

- Output MUST be valid HTML only
- Use <p>, <ul>, <li>, <strong>, <h3>, <h4>, <h5> and <br> tags where appropriate
- Do NOT use markdown
- Do NOT use **, *, bullet symbols, or numbering characters
- Do NOT include explanations outside HTML
- Do NOT include code blocks

Answer behavior:

- If the question asks for a list, output a proper <ul> with <li> items
- If order or sequence is implied in the context, preserve it
- Keep the answer concise, complete, and directly relevant to the query

Question:
{query}

Context:
{context}
"""
        
        try:
            # FIXED: Corrected call for google-genai library
            response = self.client.models.generate_content(
                model=self.llm_model_id,
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"LLM Error: {str(e)}"

    def search(self, query: str, allowed_doc_ids: list[str]) -> str:
        candidates = self.retrieve_candidates_from_pinecone(query, allowed_doc_ids, k = 15)
        if not candidates: return "No relevant documents found."
        top_chunks = self.rerank_candidates(query, candidates)
        ans = self.generate_answer(query, top_chunks)
        print(ans)
        return ans

    def ingest_document(self, raw_text: list, doc_id: str):
        # Pinecone doesn't have a "delete by metadata" in all index types 
        # without a specialized setup, but this works for most:
        try:
            self.index.delete(filter={"doc_id": {"$eq": doc_id}})
        except:
            pass 
        self.upload_raw_text(raw_text, doc_id)