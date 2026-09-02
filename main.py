from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
import time
import uuid
import os

# NEW: optional free LLM provider (Groq)
try:
    from groq import Groq
except ImportError:
    Groq = None

#----------------------
# APP Initialization
#----------------------

app = FastAPI(
    title="Enterprise AI Resume Generator Agent",
    description="An API for generating professional resumes using AI, with RAG-based job-description matching",
    version="2.0.0",
)

#----------------------
# SECURITY CONFIG
#----------------------

valid_api_keys = {os.getenv("API_KEY_HERE")}
api_key_value = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key_value) if api_key_value else None

# NEW: LLM provider selection — defaults to Groq (free) if configured, falls back to OpenAI
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key) if (Groq and groq_api_key) else None

#----------------------
# RAG CONFIG (NEW)
#----------------------

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "resume_chunks"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output size

embedding_model = None
qdrant_client = None

# Only load the heavy embedding/vector libraries if Qdrant is actually configured.
# This keeps the app lightweight (and working) until you set up the RAG phase.
if QDRANT_URL:
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    existing_collections = [c.name for c in qdrant_client.get_collections().collections]
    if COLLECTION_NAME not in existing_collections:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

#----------------------
# REQUEST MODEL
#----------------------

class ResumeRequest(BaseModel):
    api_key: str
    full_name: str
    current_role: str
    skills: list[str]
    experience_years: int

    resume_text: str | None = Field(None, max_length=90000)
    resume_file: str | None = None   # File path or uploaded file name
    job_description: str | None = Field(None, max_length=20000)  # NEW: enables RAG matching

    @model_validator(mode="after")
    def validate_resume(self):
        # User must provide either resume text or a file
        if not self.resume_text and not self.resume_file:
            raise ValueError("Provide either resume text or upload a resume.")

        # Validate uploaded file type
        if self.resume_file:
            allowed = (".pdf", ".docx", ".txt")
            if not self.resume_file.lower().endswith(allowed):
                raise ValueError(
                    "Invalid file format. Only PDF, DOCX, and TXT are supported."
                )

        return self

#----------------------
# LOGGING CONFIG
#----------------------
logs = []
def log_event(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    logs.append(log_message)
    print(log_message)

    with open("resume_generator_logs_LLM.txt", "a") as log_file:
        log_file.write(log_message + "\n")

#----------------------
# SECURITY VALIDATION
#----------------------

def verify_api_key(api_key):
    if api_key not in valid_api_keys:
        log_event(f"Unauthorized access attempt with API key: {api_key}")
        raise HTTPException(status_code=401, detail="Invalid API Key")
    log_event(f"API key verified: {api_key}")

#----------------------
# LLM HELPER
#----------------------

def call_llm(prompt):
    # NEW: Groq path (free) — used when LLM_PROVIDER=groq and a key is configured
    if LLM_PROVIDER == "groq" and groq_client is not None:
        try:
            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as exc:
            log_event(f"Groq call failed, falling back to OpenAI: {exc}")

    if client is None:
        return f"Simulated response for: {prompt}"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as exc:
        print(f"OpenAI call failed: {exc}")
        return f"Simulated response for: {prompt}"

#--------------------------------
# RAG HELPERS (NEW)
#--------------------------------

def chunk_resume(resume_text: str) -> list[str]:
    """Split resume text into chunks (by line/bullet) for embedding."""
    if not resume_text:
        return []
    raw_lines = [line.strip() for line in resume_text.split("\n")]
    chunks = [line for line in raw_lines if len(line) > 15]  # drop empty/trivial lines
    return chunks


def embed_text(text: str) -> list[float]:
    if embedding_model is None:
        return []
    return embedding_model.encode(text).tolist()


def store_resume_chunks(request_id: str, chunks: list[str]):
    """Embed and upsert resume chunks into Qdrant, tagged with request_id."""
    if qdrant_client is None or not chunks:
        log_event("Qdrant not configured or no chunks to store — skipping vector storage")
        return

    points = []
    for i, chunk in enumerate(chunks):
        vector = embed_text(chunk)
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"request_id": request_id, "text": chunk},
            )
        )

    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    log_event(f"{request_id}: stored {len(points)} resume chunks in Qdrant")


def retrieve_relevant_chunks(request_id: str, job_description: str, top_k: int = 5) -> list[str]:
    """Semantic search: find resume chunks most relevant to the job description."""
    if qdrant_client is None or not job_description:
        return []

    query_vector = embed_text(job_description)

    results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter={"must": [{"key": "request_id", "match": {"value": request_id}}]},
        limit=top_k,
    )

    matched_chunks = [hit.payload["text"] for hit in results]
    log_event(f"{request_id}: retrieved {len(matched_chunks)} relevant chunks via vector search")
    return matched_chunks


def semantic_ats_score(job_description: str, matched_chunks: list[str]) -> int:
    """
    Real similarity-based ATS score (0-100), replacing pure keyword matching.
    Uses cosine similarity between the JD embedding and each matched chunk.
    """
    if not job_description or not matched_chunks or embedding_model is None:
        return 0

    jd_vector = embedding_model.encode(job_description)
    chunk_vectors = embedding_model.encode(matched_chunks)

    import numpy as np
    similarities = [
        float(np.dot(jd_vector, cv) / (np.linalg.norm(jd_vector) * np.linalg.norm(cv)))
        for cv in chunk_vectors
    ]
    avg_similarity = sum(similarities) / len(similarities)
    score = round(avg_similarity * 100)
    return max(0, min(score, 100))

#----------------------
# PROFILE ANALYZER AGENT
#----------------------

def analyzer_agent(user_request, request_id):

    log_event(f"{request_id}: Analyzer agent received request")

    prompt = f"""
    Analyze the following candidate profile.

    Name: {user_request['full_name']}
    Current Role: {user_request['current_role']}
    Skills: {', '.join(user_request['skills'])}
    Experience: {user_request['experience_years']} years

    Return only JSON with:
    - candidate_level
    - primary_domain
    - years_experience
    """

    output = call_llm(prompt)

    log_event(f"Analyzer output: {output}")

    return output

#----------------------
# ATS SCORE CALCULATOR
#----------------------

def calculate_ats_score(resume_text: str, skills: list[str]) -> int:
    if not resume_text:
        return 0

    score = 50
    lower_text = resume_text.lower()
    for skill in skills:
        # Match if every word in the skill appears somewhere in the resume
        # (handles "REST API" vs "RESTful APIs" style mismatches)
        skill_words = skill.lower().split()
        if skill_words and all(word in lower_text for word in skill_words):
            score += 10
        else:
            score -= 5

    return max(0, min(score, 100))

#------------------------
# ATS OPTIMIZATION AGENT (UPDATED — now RAG-aware)
#------------------------

def ats_agent(user_request, matched_chunks=None, semantic_score=None):

    log_event("ATS Agent Started")

    # Fall back to keyword scoring if no JD/vector match was available
    keyword_score = calculate_ats_score(user_request.get("resume_text", ""), user_request["skills"])
    final_score = semantic_score if semantic_score is not None else keyword_score

    relevant_context = "\n".join(matched_chunks) if matched_chunks else user_request.get("resume_text", "")

    prompt = f"""
    Analyze this resume content against the skills list below and identify any important skills that seem missing or under-represented in the resume text.

    Most relevant resume content (retrieved via semantic search):
    {relevant_context}

    Skills:
    {', '.join(user_request['skills'])}

    Return only JSON:

    {{
      "missing_keywords": []
    }}
    """

    output = call_llm(prompt)

    log_event(f"ATS Output: {output} | keyword_score={keyword_score} semantic_score={semantic_score}")

    return {"llm_feedback": output, "ats_score": final_score}

#----------------------
# RESUME WRITER AGENT (UPDATED — uses RAG-matched content when available)
#----------------------

def resume_writer_agent(user_request, matched_chunks=None):

    log_event("Resume Writer Agent Started")

    resume_content = "\n".join(matched_chunks) if matched_chunks else user_request.get("resume_text", "")

    prompt = f"""
    Generate a professional ATS-friendly resume.

    Name:
    {user_request['full_name']}

    Current Role:
    {user_request['current_role']}

    Skills:
    {', '.join(user_request['skills'])}

    Experience:
    {user_request['experience_years']} years

    Most relevant resume content for this job (retrieved via semantic search):
    {resume_content}
    """

    resume = call_llm(prompt)

    output = {
        "generated_resume": resume
    }

    log_event("Resume generated successfully")

    return output


#----------------------
# HUMAN OPTIMIZER AGENT
#----------------------

def human_optimizer_agent(user_request, resume_text):

    log_event("Human Optimizer Agent Started")

    prompt = f"""
    Rewrite the resume below.

    Make it:

    - Natural
    - Human sounding
    - Remove AI generated patterns
    - Professional
    - ATS Friendly

    Resume:

    {resume_text}
    """

    optimized_resume = call_llm(prompt)

    output = {
        "human_friendly_resume": optimized_resume
    }

    log_event("Human optimization completed")

    return output

#----------------------
# REVIEWER AGENT
#----------------------

def reviewer_agent(user_request):

    log_event("Reviewer Agent Started")

    prompt = f"""
    Review this resume.

    Resume:

    {user_request.get('resume_text','')}

    Check:

    - Grammar
    - Formatting
    - Professionalism
    - Consistency

    Return concise feedback.
    """

    feedback = call_llm(prompt)

    output = {
        "review_feedback": feedback
    }

    log_event("Reviewer completed")

    return output

#----------------------
# COVER LETTER AGENT (NEW)
#----------------------

def cover_letter_agent(user_request, matched_chunks=None):

    log_event("Cover Letter Agent Started")

    job_description = user_request.get("job_description") or ""
    relevant_context = "\n".join(matched_chunks) if matched_chunks else user_request.get("resume_text", "")

    if job_description:
        jd_instruction = f"Tailor it specifically to this job description:\n{job_description}"
    else:
        jd_instruction = "No specific job description was provided — write a strong, general-purpose cover letter highlighting the candidate's background."

    prompt = f"""
    Write a professional, concise cover letter (3-4 short paragraphs) for the candidate below.

    Candidate Name: {user_request['full_name']}
    Current Role: {user_request['current_role']}
    Years of Experience: {user_request['experience_years']}
    Key Skills: {', '.join(user_request['skills'])}

    Most relevant experience from their resume:
    {relevant_context}

    {jd_instruction}

    Write in first person, professional but not stiff. If no company name is given, address it "Dear Hiring Manager,". Do not include placeholder brackets like [Company Name] unless a real company name was provided in the job description.
    """

    letter = call_llm(prompt)

    output = {
        "cover_letter": letter
    }

    log_event("Cover letter generated successfully")

    return output

#----------------------
# ORCHESTRATOR (UPDATED — adds RAG step, cover letter, and progress callback)
#----------------------

def orchestrator(user_request, request_id, progress_callback=None):
    log_event(f"Orchestrator received request: {user_request}")
    start = time.time()

    def notify(message):
        log_event(f"{request_id}: {message}")
        if progress_callback:
            progress_callback(message)

    # RAG step — chunk + store resume, then retrieve JD-relevant chunks
    matched_chunks = []
    semantic_score = None
    resume_text = user_request.get("resume_text")
    job_description = user_request.get("job_description")

    notify("Reading and chunking resume...")
    if resume_text:
        chunks = chunk_resume(resume_text)
        store_resume_chunks(request_id, chunks)

        if job_description:
            notify("Matching resume to job description...")
            matched_chunks = retrieve_relevant_chunks(request_id, job_description)
            semantic_score = semantic_ats_score(job_description, matched_chunks)

    notify("Analyzing candidate profile...")
    analyzer_output = analyzer_agent(user_request, request_id)

    notify("Scoring ATS match...")
    ats_output = ats_agent(user_request, matched_chunks=matched_chunks, semantic_score=semantic_score)

    notify("Writing first draft of resume...")
    resume_writer_output = resume_writer_agent(user_request, matched_chunks=matched_chunks)

    notify("Reviewing for grammar and consistency...")
    reviewer_output = reviewer_agent(user_request)

    notify("Polishing into a human-friendly final version...")
    human_optimizer_output = human_optimizer_agent(user_request, resume_writer_output["generated_resume"])

    notify("Drafting a matching cover letter...")
    cover_letter_output = cover_letter_agent(user_request, matched_chunks=matched_chunks)

    notify("Finalizing...")

    final_output = {
        "analyzer": analyzer_output,
        "ats_optimization": ats_output,
        "resume_writer": resume_writer_output,
        "human_optimizer": human_optimizer_output,
        "reviewer": reviewer_output,
        "cover_letter": cover_letter_output,
    }
    end = time.time()
    execution_time = round(end - start, 2)
    log_event(f"Orchestrator execution time: {execution_time}")

    return {
        "status": "success",
        "request_id": request_id,
        "execution_time": execution_time,
        "rag_matches_used": len(matched_chunks),
        "workflow": {
            "analyzer": analyzer_output,
            "ats_optimization": ats_output,
            "resume_writer": resume_writer_output,
            "human_optimizer": human_optimizer_output,
            "reviewer": reviewer_output,
            "cover_letter": cover_letter_output,
        },
        "final_report": final_output,
        "logs": logs,
    }

#----------------------
# ROOT ENDPOINT
#----------------------
@app.get("/")
def home():
    return {"message": "Welcome to the Enterprise AI Resume Generator Agent API!"}

#----------------------
# MAIN API ENDPOINT
#----------------------
@app.post("/generate_resume")
def generate_resume(request: ResumeRequest):
    request_id = str(uuid.uuid4())

    # SECURITY CHECK
    verify_api_key(request.api_key)

    # WORKFLOW EXECUTION
    result = orchestrator(request.model_dump(), request_id)

    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
