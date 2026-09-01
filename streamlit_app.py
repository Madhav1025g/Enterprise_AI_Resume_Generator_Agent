import os
import io
import uuid
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
st.set_page_config(page_title="AI Resume Generator", page_icon="📄", layout="centered")

st.markdown("""
<style>
.hero {
    background: linear-gradient(90deg, #4F46E5 0%, #9333EA 100%);
    padding: 2rem 1.5rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 1.5rem;
}
.hero h1 { margin: 0; font-size: 2.2rem; }
.hero p { margin-top: 0.4rem; opacity: 0.9; }
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
    <h1>📄 AI Resume Generator</h1>
    <p>Upload your resume, paste a job description, and let AI agents tailor and polish it for you — free, instant, ATS-ready.</p>
</div>
""", unsafe_allow_html=True)

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

# ------------------------------------------------------------------
# Input section
# ------------------------------------------------------------------
with st.container(border=True):
    st.subheader("1. Your Details")
    col1, col2 = st.columns(2)
    with col1:
        full_name = st.text_input("Full Name")
        experience_years = st.number_input("Years of Experience", min_value=0, max_value=50, step=1)
    with col2:
        current_role = st.text_input("Current Role", placeholder="e.g. Software Engineer")
        skills_input = st.text_input("Skills (comma-separated)", placeholder="Python, FastAPI, AWS")

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
    else:
        resume_text = st.text_area("Paste your resume text here", height=220)

with st.container(border=True):
    st.subheader("3. Job Description (optional, enables smart matching)")
    job_description = st.text_area("Paste the job description here", height=150)

st.write("")
generate_clicked = st.button("✨ Generate My Resume", use_container_width=True)

# ------------------------------------------------------------------
# Run pipeline
# ------------------------------------------------------------------
if generate_clicked:
    if not full_name or not resume_text:
        st.error("Please provide at least your name and a resume (uploaded or pasted).")
    else:
        with st.spinner("Running AI agents... this can take 10-30 seconds"):
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
            result = orchestrator(user_request, request_id)

        st.balloons()
        st.success(f"Done! (took {result['execution_time']}s)")

        final_resume = result["workflow"]["human_optimizer"]["human_friendly_resume"]

        tab1, tab2, tab3, tab4 = st.tabs(["✅ Final Resume", "📝 First Draft", "📊 ATS Score", "🔍 Reviewer Notes"])

        with tab1:
            st.write(final_resume)
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    "⬇️ Download as TXT",
                    data=final_resume,
                    file_name=f"{full_name.replace(' ', '_')}_resume.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with dl_col2:
                st.download_button(
                    "⬇️ Download as DOCX",
                    data=build_docx_bytes(final_resume),
                    file_name=f"{full_name.replace(' ', '_')}_resume.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )

        with tab2:
            st.write(result["workflow"]["resume_writer"]["generated_resume"])

        with tab3:
            st.json(result["workflow"]["ats_optimization"])

        with tab4:
            st.write(result["workflow"]["reviewer"]["review_feedback"])

# ------------------------------------------------------------------
# Feedback footer
# ------------------------------------------------------------------
st.divider()
st.markdown(
    "### Found this useful? \n"
    "[⭐ Leave a quick review here](PASTE_YOUR_GOOGLE_FORM_LINK_HERE) — it takes 30 seconds and helps a lot!"
)
