import os
import streamlit as st

# ------------------------------------------------------------------
# STEP A: Load secrets (API keys) from Streamlit Cloud's secrets manager
# and make them available the same way main.py already expects
# (main.py uses os.getenv, so we copy the secrets into the environment)
# ------------------------------------------------------------------
for key in ["OPENAI_API_KEY", "GROQ_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "LLM_PROVIDER", "API_KEY_HERE"]:
    if key in st.secrets:
        os.environ[key] = st.secrets[key]

# ------------------------------------------------------------------
# STEP B: Import your existing agent logic (no changes needed to main.py)
# ------------------------------------------------------------------
from main import orchestrator
import uuid

# ------------------------------------------------------------------
# STEP C: Page setup
# ------------------------------------------------------------------
st.set_page_config(page_title="AI Resume Generator", page_icon="📄", layout="centered")

st.title("📄 AI Resume Generator")
st.write("Paste your resume and (optionally) a job description — the AI agents will analyze, tailor, and polish it for you.")

# ------------------------------------------------------------------
# STEP D: Input form
# ------------------------------------------------------------------
with st.form("resume_form"):
    full_name = st.text_input("Full Name")
    current_role = st.text_input("Current Role (e.g. Software Engineer)")
    skills_input = st.text_input("Skills (comma-separated, e.g. Python, FastAPI, AWS)")
    experience_years = st.number_input("Years of Experience", min_value=0, max_value=50, step=1)
    resume_text = st.text_area("Paste your resume text here", height=250)
    job_description = st.text_area("Paste a job description (optional, enables smart matching)", height=150)

    submitted = st.form_submit_button("Generate Resume")

# ------------------------------------------------------------------
# STEP E: Run the agent pipeline when the form is submitted
# ------------------------------------------------------------------
if submitted:
    if not full_name or not resume_text:
        st.error("Please fill in at least your name and resume text.")
    else:
        with st.spinner("Running AI agents... this can take 10-30 seconds"):
            user_request = {
                "api_key": "internal",  # not used for direct calls, kept for compatibility
                "full_name": full_name,
                "current_role": current_role,
                "skills": [s.strip() for s in skills_input.split(",") if s.strip()],
                "experience_years": int(experience_years),
                "resume_text": resume_text,
                "resume_file": None,
                "job_description": job_description or None,
            }
            request_id = str(uuid.uuid4())
            result = orchestrator(user_request, request_id)

        st.success(f"Done! (took {result['execution_time']}s)")

        st.subheader("Generated Resume")
        st.write(result["workflow"]["resume_writer"]["generated_resume"])

        st.subheader("Human-Optimized Version")
        st.write(result["workflow"]["human_optimizer"]["human_friendly_resume"])

        with st.expander("ATS Score & Feedback"):
            st.json(result["workflow"]["ats_optimization"])

        with st.expander("Reviewer Feedback"):
            st.write(result["workflow"]["reviewer"]["review_feedback"])

# ------------------------------------------------------------------
# STEP F: Feedback / review link (Google Form — see setup guide)
# ------------------------------------------------------------------
st.divider()
st.markdown(
    "### Found this useful? \n"
    "[⭐ Leave a quick review here](PASTE_YOUR_GOOGLE_FORM_LINK_HERE) — it takes 30 seconds and helps a lot!"
)
