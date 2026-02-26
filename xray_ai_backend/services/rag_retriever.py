import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

KNOWLEDGE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "database", "medical_knowledge.txt"
)
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_FINAL  = 3
MIN_SCORE    = 0.25   

SYNONYMS = {
    "sob":                  "shortness of breath dyspnea",
    "shortness of breath":  "dyspnea breathlessness",
    "heart failure":        "cardiac failure cardiomegaly pulmonary edema",
    "copd":                 "emphysema chronic obstructive pulmonary disease",
    "tb":                   "tuberculosis mycobacterium",
    "xray":                 "chest x-ray radiograph imaging",
    "x-ray":                "chest x-ray radiograph imaging",
    "covid":                "covid-19 coronavirus pneumonia",
    "pe":                   "pulmonary embolism",
    "chf":                  "congestive heart failure cardiomegaly edema",
    "ards":                 "acute respiratory distress syndrome",
    "pneumo":               "pneumothorax lung collapse",
    "effusion":             "pleural effusion fluid lung",
    "ca":                   "cancer carcinoma malignancy",
    "mets":                 "metastasis metastatic cancer",
    "spo2":                 "oxygen saturation hypoxia",
    "pft":                  "pulmonary function test spirometry",
    "fvc":                  "forced vital capacity lung function",
    "fev1":                 "forced expiratory volume spirometry",
    "hrct":                 "high resolution ct chest",
    "medxscan":             "MedXScan AI chest xray analysis system",
}


def expand_query(query: str) -> str:
    """Expands medical abbreviations and adds synonyms for better retrieval."""
    q          = query.lower()
    expansions = [exp for term, exp in SYNONYMS.items() if term in q]
    return (query + " " + " ".join(expansions)).strip() if expansions else query


def parse_knowledge_chunks(filepath: str) -> list:
    """
    Splits the flat text knowledge base into individual disease chunks.
    Each section separated by dashes becomes one chunk.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()
    raw_chunks = re.split(r"-{10,}", raw)
    return [c.strip() for c in raw_chunks if len(c.strip()) > 50]


class MedicalRetriever:
    """
    Singleton FAISS retriever.
    Embeddings are built once at server startup — never reloaded per request.
    Uses cosine similarity (L2-normalized vectors + IndexFlatIP).
    """
    _instance = None

    def __init__(self):
        print("[Retriever] Initializing — loading knowledge base...")

        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.chunks   = parse_knowledge_chunks(KNOWLEDGE_PATH)

        print(f"[Retriever] {len(self.chunks)} chunks loaded")

        embeddings = self.embedder.encode(
            self.chunks,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True  
        ).astype(np.float32)

        self.dim   = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)

        print(f"[Retriever] FAISS index ready — {self.index.ntotal} vectors @ {self.dim}d")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def retrieve(self, query: str, top_k: int = TOP_K_FINAL) -> list:
        """
        Returns list of dicts: [{ chunk, score, rank }]
        Filters out chunks below MIN_SCORE cosine similarity.
        """
        expanded = expand_query(query)
        q_vec    = self.embedder.encode(
            [expanded],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        n_search        = min(top_k * 3, len(self.chunks))
        scores, indices = self.index.search(q_vec, n_search)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or float(score) < MIN_SCORE:
                continue
            results.append({
                "chunk": self.chunks[idx],
                "score": round(float(score), 4),
                "rank":  len(results) + 1
            })
            if len(results) >= top_k:
                break

        if results:
            print(f"[Retriever] '{query[:55]}' → {len(results)} chunks, "
                  f"top score={results[0]['score']:.3f}")
        else:
            print(f"[Retriever] No relevant chunks for: '{query[:55]}'")

        return results

def retrieve_context(query: str, top_k: int = TOP_K_FINAL) -> list:
    """Returns list[str] of relevant knowledge chunks."""
    return [r["chunk"] for r in MedicalRetriever.get_instance().retrieve(query, top_k)]


def retrieve_with_scores(query: str, top_k: int = TOP_K_FINAL) -> list:
    """Returns list[dict] with chunk + score."""
    return MedicalRetriever.get_instance().retrieve(query, top_k)