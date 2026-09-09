"""
Retrieval engine for the Patient & Clinical Content Assistant prototype.

This is a "RAG-lite" system: no external LLM API and no downloaded neural
embedding model. Retrieval is done with classic ML (TF-IDF + cosine
similarity, scikit-learn), and "generation" is answer synthesis: we return
the most relevant matched Q&A entries verbatim, with their source, so every
word is traceable to a real NIH document (this is even stricter grounding
than an LLM-generation approach, since there is zero risk of hallucinated
phrasing).

Swap-in upgrade path (for when you're outside this sandbox and can reach
Hugging Face):
  - Replace TfidfVectorizer with sentence-transformers embeddings
    (e.g. 'all-MiniLM-L6-v2') + cosine similarity on dense vectors, OR
    build a FAISS/Chroma index over those embeddings.
  - Optionally, plug in a local generative model
    (e.g. google/flan-t5-base, or a small local LLM via llama.cpp) to
    paraphrase/synthesize the top retrieved answers instead of returning
    them verbatim.
The retrieval pipeline’s "shape" (embed query -> similarity search -> return
grounded, cited answer) is the same either way -- that’s the RAG pattern
from the HCA architecture doc, just running on local classic-ML components.
"""

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "medquad_clean.csv")
INDEX_PATH = os.path.join(BASE_DIR, "tfidf_index.pkl")

# Simple out-of-scope / emergency guardrail keywords, mirroring the
# HCA doc's "redirect symptom-like queries" guardrail.
EMERGENCY_PATTERNS = [
    r"\bchest pain\b", r"\bcan'?t breathe\b", r"\bsuicid", r"\bself.?harm\b",
    r"\bsevere bleeding\b", r"\boverdose\b", r"\bstroke\b", r"\bunconscious\b",
    r"\bheart attack\b", r"\bemergency\b",
]


class MedicalRAGEngine:
    def __init__(self, data_path=DATA_PATH):
        self.df = pd.read_csv(data_path)
        # Combine focus + question for a richer retrieval signal
        self.df["retrieval_text"] = (
            self.df["focus"].fillna("") + " " + self.df["question"].fillna("")
        )
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=50000,
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(self.df["retrieval_text"])

    def _is_emergency(self, query: str) -> bool:
        q = query.lower()
        return any(re.search(p, q) for p in EMERGENCY_PATTERNS)

    def search(self, query: str, top_k: int = 5):
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix).flatten()
        top_idx = sims.argsort()[::-1][:top_k]
        results = []
        for idx in top_idx:
            score = float(sims[idx])
            if score <= 0.0:
                continue
            row = self.df.iloc[idx]
            results.append({
                "score": round(score, 4),
                "focus": row["focus"],
                "question": row["question"],
                "answer": row["answer"],
                "source": row["source"],
                "url": row["url"],
            })
        return results

    def answer(self, query: str, top_k: int = 3, min_score: float = 0.12):
        """
        Mimics the HCA pipeline's guardrails:
        1. Emergency/symptom-like query -> redirect, don't answer.
        2. No confident match -> honest "don't know" fallback.
        3. Otherwise -> grounded answer with citation(s), synthesized from
           the top retrieved entries (concatenated + deduplicated).
        """
        if self._is_emergency(query):
            return {
                "type": "guardrail_redirect",
                "message": (
                    "I'm not able to assess symptoms like that. If this is "
                    "urgent, please call 999 (or 911) immediately, or 111 "
                    "for non-emergency medical advice. I can help you find "
                    "general health information instead."
                ),
                "sources": [],
            }

        results = self.search(query, top_k=top_k)
        results = [r for r in results if r["score"] >= min_score]

        if not results:
            return {
                "type": "no_answer",
                "message": (
                    "I don't have reliable information on that in my current "
                    "knowledge base. I'd recommend checking with a licensed "
                    "medical provider or a trusted source like MedlinePlus."
                ),
                "sources": [],
            }

        # "Generation" = pick the best-matching answer as the primary answer,
        # and surface any other distinct top matches as related info.
        primary = results[0]
        related = [r for r in results[1:] if r["focus"] != primary["focus"]]

        return {
            "type": "answer",
            "message": primary["answer"],
            "matched_question": primary["question"],
            "topic": primary["focus"],
            "confidence": primary["score"],
            "sources": [{
                "source": primary["source"],
                "url": primary["url"],
            }],
            "related": [
                {"question": r["question"], "topic": r["focus"], "score": r["score"]}
                for r in related[:2]
            ],
        }

    def save_index(self, path=INDEX_PATH):
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "matrix": self.matrix}, f)


if __name__ == "__main__":
    engine = MedicalRAGEngine()
    print(f"Indexed {len(engine.df)} QA pairs.\n")

    test_queries = [
        "What are the symptoms of diabetes?",
        "How is high blood pressure treated?",
        "I have sharp chest pain, what should I do?",
        "What causes Parkinson's disease?",
        "What is the treatment for asthma in children?",
    ]

    for q in test_queries:
        print("=" * 70)
        print(f"Q: {q}")
        result = engine.answer(q)
        print(f"Type: {result['type']}")
        if result["type"] == "answer":
            print(f"Topic matched: {result['topic']} (confidence {result['confidence']})")
            print(f"Answer: {result['message'][:400]}...")
            print(f"Source: {result['sources']}")
        else:
            print(f"Message: {result['message']}")
        print()
