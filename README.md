# 📄 Enterprise AI Resume & Cover Letter Generator

**[🔗 Try the live demo] - https://airesumegeneratoragent.streamlit.app/**

An AI-powered multi-agent system that reads your resume and a job description, then generates a tailored resume, a matching cover letter, and an ATS match score — all in one pass.

## What it does

- 📤 Upload your resume (PDF, DOCX, or TXT) or paste it directly
- 🎯 Paste a job description to get semantically-matched, tailored output
- ✅ Get a rewritten, ATS-optimized resume with a clear match score and gap analysis
- ✉️ Get a matching cover letter generated automatically
- 🔍 Review AI-suggested fixes with one-click "Apply" — no manual copy-pasting
- ⬇️ Export everything as TXT, DOCX, or PDF

## How it works

A 7-agent pipeline built on FastAPI:

1. **Profile Analyzer** — reads candidate background and experience level
2. **ATS Optimizer** — scores the resume against the job description using RAG-based semantic matching + deterministic keyword matching
3. **Resume Writer** — drafts a tailored version using the full resume as ground truth
4. **Reviewer** — checks grammar, formatting, and consistency, returning structured, applicable fixes
5. **Human Optimizer** — polishes the final version to read naturally
6. **Cover Letter Writer** — drafts a matching cover letter
7. **Completeness Checker** — a guardrail that flags any content that may have been dropped during generation

Job-description matching is powered by a **RAG pipeline** (Sentence-Transformers embeddings + Qdrant vector search), so the agents can identify and emphasize the most relevant parts of a resume for a specific role.

## Tech Stack

**Backend:** Python, FastAPI, Pydantic
**AI/LLM:** Groq API (primary), OpenAI API (fallback), Sentence-Transformers, Qdrant (vector database)
**Frontend:** Streamlit
**File handling:** pypdf, python-docx, fpdf2
**Deployment:** Streamlit Community Cloud


## Running Locally

```bash
git clone <this-repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_groq_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
LLM_PROVIDER=groq
API_KEY_HERE=any_value
```

Run it:
```bash
streamlit run streamlit_app.py
```

## Author

Built by Madhav G
