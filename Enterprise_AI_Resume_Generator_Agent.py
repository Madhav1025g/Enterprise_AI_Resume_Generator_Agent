from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
import time
import uuid
import os

#----------------------
# APP Initialization
#----------------------

app = FastAPI(
    title="Enterprise AI Resume Generator Agent",
    description="An API for generating professional resumes using AI",
    version="1.0.0",
)

#----------------------
# SECURITY CONFIG
#----------------------

valid_api_keys = {os.getenv("API_KEY_HERE")}
api_key_value = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key_value) if api_key_value else None

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
        if skill.lower() in lower_text:
            score += 10
        else:
            score -= 5

    return max(0, min(score, 100))

#------------------------
# ATS OPTIMIZATION AGENT
#------------------------

def ats_agent(user_request):

    log_event("ATS Agent Started")

    prompt = f"""
    Analyze this resume for ATS.

    Resume:
    {user_request.get('resume_text','')}

    Skills:
    {', '.join(user_request['skills'])}

    Return only JSON:

    {{
      "missing_keywords": [],
      "ats_score": 0
    }}
    """

    output = call_llm(prompt)

    log_event(f"ATS Output: {output}")

    return output

#----------------------
# RESUME WRITER AGENT
#----------------------

def resume_writer_agent(user_request):

    log_event("Resume Writer Agent Started")

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

    Resume Content:
    {user_request.get('resume_text','')}
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
# ORCHESTRATOR
#----------------------

def orchestrator(user_request, request_id):
    log_event(f"Orchestrator received request: {user_request}")
    start = time.time()
    analyzer_output = analyzer_agent(user_request, request_id)
    ats_output = ats_agent(user_request)
    resume_writer_output = resume_writer_agent(user_request)
    reviewer_output = reviewer_agent(user_request)
    human_optimizer_output = human_optimizer_agent(user_request, resume_writer_output["generated_resume"])

    final_output = {
        "analyzer": analyzer_output,
        "ats_optimization": ats_output,
        "resume_writer": resume_writer_output,
        "human_optimizer": human_optimizer_output,
        "reviewer": reviewer_output
    }
    end = time.time()
    execution_time = round(end - start, 2)
    log_event(f"Orchestrator execution time: {execution_time}")

    return {
        "status": "success",
        "request_id": request_id,
        "execution_time": execution_time,
        "workflow": {
            "analyzer": analyzer_output,
            "ats_optimization": ats_output,
            "resume_writer": resume_writer_output,
            "human_optimizer": human_optimizer_output,
            "reviewer": reviewer_output
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
