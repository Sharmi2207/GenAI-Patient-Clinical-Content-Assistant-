# Health Information Assistant — Retrieval-based Medical Q&A Prototype

A working, local, no-API-key MVP inspired by the HCA Patient & Clinical Content
Assistant architecture. Instead of consultant/pricing content, it's grounded
in the **MedQuAD** dataset (16,406 cleaned medical Q&A pairs from 9 NIH
sources: cancer.gov, MedlinePlus, GARD, GHR, NIDDK, NINDS, NIHSeniorHealth,
NHLBI, CDC). CC BY 4.0 licensed, from https://github.com/abachaa/MedQuAD.

## How it works (the RAG pattern, without an LLM)

1. **Ingest**: `parse_medquad.py` flattens the raw MedQuAD XML files into one
   clean CSV (`medquad_clean.csv`) — this mirrors the "ingestion pipeline"
   step in the original architecture doc, minus S3/Textract/PHI-redaction
   (not needed since the source is already clean public NIH text).
2. **Index**: `rag_engine.py` builds a TF-IDF vector index (scikit-learn)
   over each entry's topic + question text — the classic-ML stand-in for
   the Titan/OpenSearch embedding + vector-index step.
3. **Retrieve**: a user query is vectorized and compared via cosine
   similarity against the index — the stand-in for OpenSearch hybrid
   retrieval + reranking.
4. **"Generate" (grounded, no hallucination risk)**: instead of an LLM
   paraphrasing the retrieved text, the prototype returns the top-matched
   NIH answer verbatim with its source and a confidence score. This is
   actually *stricter* grounding than LLM generation — zero risk of
   invented facts, at the cost of less conversational phrasing.
5. **Guardrail**: emergency/symptom-like queries (chest pain, suicidal
   ideation, etc.) are pattern-matched and redirected to 999/911/111
   instead of answered — mirroring the HCA doc's clinical-safety guardrail.

## Run it

```bash
pip install pandas scikit-learn flask
python3 app.py
```

Then open http://localhost:5000 in your browser.

Or test the engine directly without the web UI:

```bash
python3 rag_engine.py
```

## Files

- `parse_medquad.py` — one-time script that builds `medquad_clean.csv` from
  the raw MedQuAD GitHub repo (already run; the CSV is included).
- `medquad_clean.csv` — the flattened dataset (16,406 rows: doc_id, source,
  focus, url, question, question_type, answer).
- `rag_engine.py` — the retrieval + guardrail engine (`MedicalRAGEngine`
  class). Run directly for a CLI smoke test.
- `app.py` — Flask backend exposing `/api/ask` and `/api/stats`.
- `static/index.html` — the chat UI.

## Upgrade path (once you're outside a sandboxed/restricted network)

This prototype avoids downloading any pretrained neural model so it runs
anywhere. To make it noticeably smarter, swap in:

1. **Dense embeddings** instead of TF-IDF: replace `TfidfVectorizer` in
   `rag_engine.py` with `sentence-transformers` (e.g. `all-MiniLM-L6-v2`).
   This catches semantic matches TF-IDF misses (e.g. "my knee hurts when I
   climb stairs" → osteoarthritis) since it doesn't need shared keywords.
2. **A real vector store**: FAISS or Chroma instead of an in-memory sklearn
   matrix — matters once you're past ~50k+ documents.
3. **A local generative model** (optional): a small local LLM (e.g.
   `flan-t5-base`, or a quantized model via `llama.cpp`/`ollama`) to
   paraphrase/synthesize across the top-k retrieved answers into a single
   conversational response with inline citations — this is the step that
   turns "retrieval" into full "RAG" in the sense of the original HCA doc.
4. **Hybrid search** (BM25 + dense vectors) — combine `rank_bm25` with the
   dense embeddings above for the best of both (exact term matches +
   semantic matches), same rationale as the OpenSearch hybrid design in the
   original architecture.
5. **Evaluation**: write ~20-30 held-out test questions with known-correct
   MedQuAD answers and measure retrieval hit-rate — a tiny version of the
   RAGAS evaluation gate in the original design.

## Scope note

This is a general health-information demo, not the patient-facing HCA
assistant itself — MedQuAD has no consultants, pricing, or hospital data.
The pipeline shape (ingest → chunk/index → retrieve → guardrail → grounded,
cited answer) is what carries over directly to a real deployment on your
own content.
