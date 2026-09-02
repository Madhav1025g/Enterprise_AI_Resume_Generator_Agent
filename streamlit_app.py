import os
import io
import uuid
import requests
import streamlit as st

# ------------------------------------------------------------------
# Load secrets (API keys) from Streamlit Cloud's secrets manager
# ------------------------------------------------------------------
for key in ["OPENAI_API_KEY", "GROQ_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "LLM_PROVIDER", "API_KEY_HERE"]:
    if key in st.secrets:
        os.environ[key] = st.secrets[key]

from main import orchestrator

# ------------------------------------------------------------------
# Page setup + custom styling
# ------------------------------------------------------------------
st.set_page_config(page_title="AI Resume & Cover Letter Generator", page_icon="📄", layout="centered")

st.markdown("""
<style>
.hero {
    background: linear-gradient(90deg, #4F46E5 0%, #9333EA 100%);
    padding: 2rem 1.5rem 1.5rem 1.5rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 0.75rem;
}
.hero h1 { margin: 0; font-size: 2.2rem; }
.hero p { margin-top: 0.4rem; opacity: 0.9; }
.badge-row { margin-bottom: 1.2rem; }
.badge {
    display: inline-block;
    background: #F3F0FF;
    color: #4F46E5;
    border-radius: 999px;
    padding: 0.25rem 0.8rem;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
}
.privacy-note {
    font-size: 0.85rem;
    color: #6B7280;
    text-align: center;
    margin-top: 0.5rem;
}
div.stButton > button, div.stDownloadButton > button {
    background: linear-gradient(90deg, #4F46E5 0%, #9333EA 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.4rem;
    font-weight: 600;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
    opacity: 0.9;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>📄 AI Resume & Cover Letter Generator</h1>
    <p>Upload your resume and a job description — a multi-agent AI pipeline tailors your resume, scores it for ATS match, and drafts a matching cover letter, all in one pass.</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="badge-row">'
    '<span class="badge">⚡ Groq LLM</span>'
    '<span class="badge">🧠 Multi-Agent</span>'
    '<span class="badge">🔍 RAG Matching</span>'
    '<span class="badge">🐍 FastAPI</span>'
    '</div>',
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Usage counter (free, external, persists across app restarts)
# ------------------------------------------------------------------
COUNTER_NAMESPACE = "ai-resume-generator-demo"
COUNTER_KEY = "resumes-generated"

def get_counter_count():
    try:
        resp = requests.get(f"https://api.counterapi.dev/v1/{COUNTER_NAMESPACE}/{COUNTER_KEY}", timeout=3)
        return resp.json().get("count", None)
    except Exception:
        return None

def increment_counter():
    try:
        resp = requests.get(f"https://api.counterapi.dev/v1/{COUNTER_NAMESPACE}/{COUNTER_KEY}/up", timeout=3)
        return resp.json().get("count", None)
    except Exception:
        return None

current_count = get_counter_count()
if current_count is not None:
    st.caption(f"✨ {current_count} resumes generated so far by people using this tool")

# ------------------------------------------------------------------
# How it works
# ------------------------------------------------------------------
with st.expander("ℹ️ How this works (5-agent pipeline)"):
    st.markdown("""
1. **Profile Analyzer** — reads your background and experience level
2. **ATS Optimizer** — scores your resume against the job description using semantic + keyword matching
3. **Resume Writer** — drafts a tailored first version
4. **Reviewer** — checks grammar, formatting, and consistency
5. **Human Optimizer** — polishes the final version to sound natural, not AI-generated
6. **Cover Letter Writer** — drafts a matching cover letter using the same context

All steps run automatically in sequence — you'll see live progress below once you click Generate.
    """)

# ------------------------------------------------------------------
# File parsing helpers
# ------------------------------------------------------------------
def extract_text_from_upload(uploaded_file):
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()

    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        import pypdf
        reader = pypdf.PdfReader(uploaded_file)
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    if name.endswith(".docx"):
        import docx
        document = docx.Document(uploaded_file)
        return "\n".join(p.text for p in document.paragraphs)

    return None


def build_docx_bytes(text: str) -> bytes:
    import docx
    document = docx.Document()
    for line in text.split("\n"):
        document.add_paragraph(line)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


SAMPLE_RESUME = """Jordan Lee
Software Engineer
jordan.lee@email.com | (555) 123-4567 | Boston, MA

PROFESSIONAL SUMMARY
Backend Software Engineer with 5 years of experience building REST APIs and cloud-native
applications. Strong background in Python, FastAPI, and AWS.

EXPERIENCE
TechCorp Inc. — Remote | Jan 2021 - Present
Software Engineer
- Built and maintained REST APIs serving 1M+ daily requests using Python and FastAPI
- Deployed microservices on AWS Lambda and ECS, reducing infra costs by 20%
- Automated CI/CD pipelines using GitHub Actions and Docker

SKILLS
Python, FastAPI, AWS, Docker, PostgreSQL, REST APIs, Git
"""

SAMPLE_JD = """We are hiring a Backend Software Engineer with experience in Python, FastAPI, and AWS.
The ideal candidate has built and scaled REST APIs, has experience with Docker and CI/CD,
and is comfortable working in an Agile team environment.
"""

# ------------------------------------------------------------------
# Sample data button
# ------------------------------------------------------------------
if "full_name" not in st.session_state:
    st.session_state.full_name = ""
    st.session_state.current_role = ""
    st.session_state.skills_input = ""
    st.session_state.experience_years = 0
    st.session_state.resume_text_area = ""
    st.session_state.job_description_area = ""

if st.button("🎬 Try with sample data instead"):
    st.session_state.full_name = "Jordan Lee"
    st.session_state.current_role = "Software Engineer"
    st.session_state.skills_input = "Python, FastAPI, AWS, Docker, PostgreSQL, REST APIs, Git"
    st.session_state.experience_years = 5
    st.session_state.resume_text_area = SAMPLE_RESUME
    st.session_state.job_description_area = SAMPLE_JD
    st.rerun()

# ------------------------------------------------------------------
# Input section
# ------------------------------------------------------------------
with st.container(border=True):
    st.subheader("1. Your Details")
    col1, col2 = st.columns(2)
    with col1:
        full_name = st.text_input("Full Name", key="full_name")
        experience_years = st.number_input("Years of Experience", min_value=0, max_value=50, step=1, key="experience_years")
    with col2:
        current_role = st.text_input("Current Role", placeholder="e.g. Software Engineer", key="current_role")
        skills_input = st.text_input("Skills (comma-separated)", placeholder="Python, FastAPI, AWS", key="skills_input")

with st.container(border=True):
    st.subheader("2. Your Resume")
    input_mode = st.radio("How would you like to provide your resume?", ["Upload a file", "Paste text"], horizontal=True)

    resume_text = None
    if input_mode == "Upload a file":
        uploaded_file = st.file_uploader("Upload your resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])
        if uploaded_file is not None:
            resume_text = extract_text_from_upload(uploaded_file)
            if resume_text:
                st.success(f"Loaded {uploaded_file.name} ({len(resume_text)} characters)")
            else:
                st.error("Couldn't read that file — try TXT or DOCX instead.")
        elif st.session_state.resume_text_area:
            resume_text = st.session_state.resume_text_area
            st.info("Using sample resume text (switch to 'Paste text' to view/edit it).")
    else:
        resume_text = st.text_area("Paste your resume text here", height=220, key="resume_text_area")

with st.container(border=True):
    st.subheader("3. Job Description (optional, enables smart matching + tailored cover letter)")
    job_description = st.text_area("Paste the job description here", height=150, key="job_description_area")

st.write("")
generate_clicked = st.button("✨ Generate Resume + Cover Letter", use_container_width=True)

st.markdown('<p class="privacy-note">🔒 Your resume text isn\'t stored anywhere — it only exists for the duration of this session.</p>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# Run pipeline with live progress
# ------------------------------------------------------------------
if generate_clicked:
    if not full_name or not resume_text:
        st.error("Please provide at least your name and a resume (uploaded or pasted).")
    else:
        with st.status("Starting AI agents...", expanded=True) as status:
            def update(msg):
                status.update(label=msg)
                st.write(f"→ {msg}")

            user_request = {
                "api_key": "internal",
                "full_name": full_name,
                "current_role": current_role,
                "skills": [s.strip() for s in skills_input.split(",") if s.strip()],
                "experience_years": int(experience_years),
                "resume_text": resume_text,
                "resume_file": None,
                "job_description": job_description or None,
            }
            request_id = str(uuid.uuid4())
            result = orchestrator(user_request, request_id, progress_callback=update)
            status.update(label="All agents finished!", state="complete")

        st.success(f"Done! (took {result['execution_time']}s)")
        increment_counter()

        final_resume = result["workflow"]["human_optimizer"]["human_friendly_resume"]
        cover_letter = result["workflow"]["cover_letter"]["cover_letter"]

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["✅ Final Resume", "✉️ Cover Letter", "📝 First Draft", "📊 ATS Score", "🔍 Reviewer Notes"]
        )

        with tab1:
            st.write(final_resume)
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    "⬇️ Download Resume (TXT)",
                    data=final_resume,
                    file_name=f"{full_name.replace(' ', '_')}_resume.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with dl_col2:
                st.download_button(
                    "⬇️ Download Resume (DOCX)",
                    data=build_docx_bytes(final_resume),
                    file_name=f"{full_name.replace(' ', '_')}_resume.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )

        with tab2:
            st.write(cover_letter)
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    "⬇️ Download Cover Letter (TXT)",
                    data=cover_letter,
                    file_name=f"{full_name.replace(' ', '_')}_cover_letter.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with dl_col2:
                st.download_button(
                    "⬇️ Download Cover Letter (DOCX)",
                    data=build_docx_bytes(cover_letter),
                    file_name=f"{full_name.replace(' ', '_')}_cover_letter.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )

        with tab3:
            st.write(result["workflow"]["resume_writer"]["generated_resume"])

        with tab4:
            ats_data = result["workflow"]["ats_optimization"]
            score = ats_data.get("ats_score", 0)
            st.metric("ATS Match Score", f"{score}/100")
            st.progress(score / 100)
            with st.expander("Detailed feedback"):
                st.write(ats_data.get("llm_feedback", ""))

        with tab5:
            st.write(result["workflow"]["reviewer"]["review_feedback"])
)
