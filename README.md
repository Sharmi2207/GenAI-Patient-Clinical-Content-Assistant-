# Patient & Clinical Content Assistant 🩺

A grounded, retrieval-augmented medical question-answering assistant designed to provide reliable, verifiable health information without hallucination risks.

🔗 **Live Demo:** [https://genai-patient-clinical-content-assistant.streamlit.app/](https://genai-patient-clinical-content-assistant.streamlit.app/)

---

## 📌 Project Overview

The **Patient & Clinical Content Assistant** is an AI-powered health information prototype that uses Retrieval-Augmented Generation (RAG) principles to answer patient and clinician queries. Instead of ungrounded generative text, every response is retrieved directly from official National Institutes of Health (NIH) medical publications, ensuring strict factual accuracy and traceability.

### 🎯 Key Highlights
- **100% Grounded Answers**: Eliminates medical hallucinations by matching queries to verified NIH documentation with transparent confidence scores.
- **Direct Source Citations**: Every answer includes full provenance (NIH institute name and official reference URL).
- **Clinical Safety Guardrails**: Automatically detects high-risk or emergency queries (e.g., chest pain, breathing distress, stroke symptoms) and immediately provides emergency redirect guidance (999/911/111).
- **Comprehensive Knowledge Base**: Built on the **MedQuAD** dataset featuring **16,406** cleaned medical Q&A pairs across 9 NIH institutes (*MedlinePlus, CDC, Cancer.gov, NIDDK, NINDS, NHLBI, GARD, GHR, NIHSeniorHealth*).

---

## 🏗️ Architecture & Workflow

```
[ User Query ]
       │
       ▼
[ Clinical Safety Guardrail ] ──(Emergency / Acute Symptom)──► [ Emergency Redirection (999/911) ]
       │
       ▼ (Informational Query)
[ TF-IDF / Semantic Retrieval ]
       │
       ▼
[ MedQuAD Knowledge Base (16,406 Q&A pairs) ]
       │
       ▼
[ Grounded Answer Synthesis + NIH Citation + Confidence Score ]
```

---

## 🚀 Quickstart (Local Run)

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```

### 2. Launch the Streamlit App
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

*(Optional)* Run the Flask REST API & Web UI:
```bash
python flask_app.py
```

---

## 📁 Repository Structure

```
├── app.py              # Streamlit web application & interactive chat UI
├── flask_app.py        # Flask REST API backend
├── rag_engine.py       # Core medical retrieval engine & safety guardrails
├── parse_medquad.py    # MedQuAD dataset ingestion & preprocessing script
├── medquad_clean.csv   # Cleaned NIH medical Q&A knowledge base (16,406 records)
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

---

## ⚖️ Disclaimer

*This assistant is developed for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.*
